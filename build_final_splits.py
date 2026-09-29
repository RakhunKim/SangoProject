# =====================================================================
# build_final_splits.py — v2.4
#
# v1 대비 수정 사항 (코드 리뷰 반영, blocking issue 5개 + 안전성 보완):
#   1. datasets>=4.0.0에서 legacy loading script(trust_remote_code)가 완전히 제거됨
#      (실측 확인: RuntimeError "Dataset scripts are no longer supported") -> 버전 가드 추가,
#      repo SHA를 로드 "후"가 아니라 "먼저" 조회해서 revision으로 고정
#   2. 전체 리스트를 여러 번 메모리에 복제하던 구조 제거 -> LASER 필터는 Dataset.filter()로
#      lazy 처리, 배치 단위 스트리밍 write로 전환, laser_score를 별도 lookup dict 대신
#      레코드에 그대로 실어서 이동 (중복 사전 제거)
#   3. French tokenization을 정규식 기반으로 통일 (콤마로 구분된 MEYNG gloss가 토큰에
#      punctuation을 포함한 채 들어가던 문제 수정) + NFC 정규화 + casefold
#   4. 층화 quota를 largest-remainder allocation으로 정확히 CANDIDATE_COUNT에 맞춤,
#      최종 선택을 seeded_rank로 재정렬해 strata 순회 순서에 의존하지 않게 함
#   5. Direct-9K 추출 시 candidate의 pair_id뿐 아니라 canonical EN/SG도 함께 제외
#      (같은 문장이 다른 번역과 짝지어져 다른 pair_id를 갖는 경우까지 차단)
#   + atomic write(.tmp -> os.replace), audit에 count mismatch도 hard failure로 포함
#
# v2 -> v2.1 수정 사항 (2차 코드 리뷰 반영):
#   6. Sango 정규화를 canonicalization(pair_id/overlap 비교용)과 저장용 표기 정규화로
#      분리 -> auto_clean_pairs_stream()이 이제 정규화된 sg로 script validity를 검사하고,
#      정규화하면 ë/ö가 될 ɛ/ɔ 표기 문장이 script-invalid로 오제거되던 문제 수정.
#      최종 레코드에 정규화 sg + 원문 sg_raw를 함께 저장.
#   7. DOWNSTREAM_OUTPUTS에 빠져있던 direct9k_train_lineage.json 추가, --force일 때만이
#      아니라 stage가 (불완전 상태로 인해) 자연 재실행될 때도 항상 downstream 무효화
#   8. 모든 JSON 사이드카(provenance/stats/lineage/audit/manifest)를 save_json_atomic()으로
#      통일 -> 중간에 끊겨도 outputs_complete()가 partial JSON을 완료로 오판하지 않음
#   9. stage_manifest()가 audit hard_failures/lineage_consistent를 확인해서, audit 실패
#      상태에서는 manifest 생성을 하드 블록
#
# v2.1 -> v2.2 수정 사항 (3차 코드 리뷰 반영):
#  10. check_script_validity()가 문자별 검사 시 lower()를 적용하지 않아서, 정규화로 정상
#      생성된 대문자 발음구별기호(Ë/Ö/Ê/Ô — normalize_sango_orthography가 Ɛ/Ɔ를 변환한
#      결과)가 소문자만 등재된 허용 문자 목록에서 다시 탈락하던 버그 수정 (실측 재현: "BƐ ƆKO"
#      -> "BË ÖKO"까지는 정상 변환됐는데 그 결과가 script-invalid로 걸려 데이터가 유실됨)
#  11. audit freshness checksum 추가 — stage_audit()이 audit_report.json에 당시
#      pool/candidates/train sha256을 함께 기록하고, stage_manifest()가 이걸 현재 파일
#      해시와 대조해서 audit 이후 파일이 바뀐 채로 `--stage manifest`만 단독 실행되는
#      상황을 차단
#
# v2.2 -> v2.3 수정 사항 (4차 코드 리뷰 반영):
#  12. stage 자신의 atomic성 보완 — CURRENT_STAGE_OUTPUTS + invalidate_current_stage()
#      추가. 이전에는 재실행 도중(예: candidates가 evaluation_candidates.jsonl은 새로
#      쓰고 candidate_ids.json 쓰기 전에 중단) 이전 세대의 sidecar가 그대로 남아있으면
#      outputs_complete()가 새 파일+옛 sidecar를 완료로 오판할 수 있었음. 이제 stage
#      재실행 직전에 그 stage 자신의 기존 산출물(.tmp 잔재 포함)을 먼저 지운다.
#      (실측 재현: candidate_ids.json 쓰기 직전 중단 시뮬레이션 -> 이전 버전이면 옛
#      candidate_ids.json이 그대로 남아 완료로 오판했을 상황이, 이제는 파일이 없어서
#      outputs_complete()가 정확히 False를 반환하고 재실행됨을 확인)
#  13. audit에 candidate_sidecar_matches 체크 추가 — candidate_ids.json의 pair_ids/count/
#      seed가 evaluation_candidates.jsonl 실제 내용과 일치하는지 검사, 불일치 시 hard failure
#  14. NLLB/MEYNG vocabulary repo revision 조회 실패 시 경고만 하고 계속 진행하던 것을,
#      논문용 최종 데이터 구축이라 재현성 보장을 위해 하드 중단(RuntimeError)으로 변경
#
# v2.3 -> v2.4 수정 사항 (5차 코드 리뷰 반영, 운영 안전장치):
#  15. Google Drive가 마운트되지 않은 상태에서 OUTPUT_DIR.mkdir(parents=True)를 그대로
#      실행하면, /content/drive/MyDrive/... 아래에 로컬(임시) 디렉터리가 조용히 만들어져
#      버림 -> 실행은 정상처럼 보이지만 런타임 종료 시 산출물이 전부 사라지는 조용한
#      데이터 손실 시나리오였음. import 시점에 DRIVE_ROOT.is_dir()를 먼저 확인해서
#      마운트 안 됐으면 즉시 RuntimeError로 중단.
# =====================================================================

import argparse, hashlib, json, os, re, sys, unicodedata
from pathlib import Path

# 🔒 Drive가 마운트되지 않은 상태에서 그냥 mkdir(parents=True)를 하면, 존재하지도 않는
# /content/drive/MyDrive/... 아래에 Colab 로컬(임시) 디렉터리가 그대로 만들어져 버린다.
# 이 경우 스크립트는 정상 동작하는 것처럼 보이지만, 런타임 종료 시 그 안의 모든 산출물이
# 사라진다 — Drive에 저장됐다고 믿었던 게 통째로 날아가는 조용한 데이터 손실 시나리오.
DRIVE_ROOT = Path("/content/drive/MyDrive")
if not DRIVE_ROOT.is_dir():
    raise RuntimeError(
        "Google Drive가 마운트되지 않았습니다. 먼저 다음을 실행하세요:\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')"
    )

OUTPUT_DIR = DRIVE_ROOT / "sango_project2/data_build"  # 본인 경로로 교체
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --force로 특정 stage를 재실행하면, 그 stage에 의존하는 downstream 산출물은
# 예전 clean pool/candidate 기준으로 만들어진 낡은 상태로 남을 수 있다.
# (예: clean --force 후 candidates/train이 파일 존재만 보고 skip되면, 새 pool과
#  옛 split이 뒤섞인 manifest가 만들어짐) -> 재실행 stage보다 하위인 산출물은 자동 삭제.
DOWNSTREAM_OUTPUTS = {
    "source": [
        "final_clean_pool.jsonl", "filter_statistics.json", "excluded_pairs_sample.jsonl",
        "vocabulary_provenance.json", "evaluation_candidates.jsonl", "candidate_ids.json",
        "direct9k_train.jsonl", "direct9k_train_lineage.json", "audit_report.json",
        "prevalidation_manifest.json",
    ],
    "clean": [
        "evaluation_candidates.jsonl", "candidate_ids.json",
        "direct9k_train.jsonl", "direct9k_train_lineage.json", "audit_report.json",
        "prevalidation_manifest.json",
    ],
    "candidates": [
        "direct9k_train.jsonl", "direct9k_train_lineage.json", "audit_report.json",
        "prevalidation_manifest.json",
    ],
    "train": [
        "audit_report.json", "prevalidation_manifest.json",
    ],
}


def invalidate_downstream(stage: str):
    for filename in DOWNSTREAM_OUTPUTS.get(stage, []):
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()
            print(f"[invalidate] {stage} 재실행으로 낡은 downstream 산출물 삭제: {path}")


# 위 downstream 무효화는 "이 stage보다 하위" 산출물만 지운다. 그런데 stage 자신이
# 여러 파일을 쓰는 도중 끊기면(예: candidates가 evaluation_candidates.jsonl은 새로 썼는데
# candidate_ids.json 쓰기 전에 중단) 재실행 시 새 파일과 "이전 세대의" 옛 sidecar가
# 뒤섞인 채로 outputs_complete()가 둘 다 존재/size>0라서 완료로 오판할 수 있다.
# -> stage를 실제로 재실행하기 "직전"에 그 stage 자신의 기존 산출물도 먼저 지운다.
CURRENT_STAGE_OUTPUTS = {
    "source": [
        "after_laser.jsonl", "source_provenance.json",
    ],
    "clean": [
        "final_clean_pool.jsonl", "filter_statistics.json",
        "excluded_pairs_sample.jsonl", "vocabulary_provenance.json",
    ],
    "candidates": [
        "evaluation_candidates.jsonl", "candidate_ids.json",
    ],
    "train": [
        "direct9k_train.jsonl", "direct9k_train_lineage.json",
    ],
}


def invalidate_current_stage(stage: str):
    for filename in CURRENT_STAGE_OUTPUTS.get(stage, []):
        path = OUTPUT_DIR / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        for target in (path, tmp_path):
            if target.exists():
                target.unlink()
                print(f"[invalidate] {stage} 재실행 전 이전 세대 산출물 삭제: {target}")

# ---------------------------------------------------------------------
# 고정 파라미터 — 전부 manifest에 그대로 기록됨
# ---------------------------------------------------------------------
LASER_THRESHOLD = 1.0
MIN_LEN, MAX_LEN, LEN_RATIO_MAX = 3, 200, 3.0
FRENCH_THRESHOLD = 0.3
CANDIDATE_SEED = 42
TRAIN_SEED = 42
CANDIDATE_COUNT = 350
TRAIN_COUNT = 9000
BATCH_SIZE = 10000

SRC_DATASET = "allenai/nllb"
SRC_CONFIG = "eng_Latn-sag_Latn"
VOCAB_DATASET = "MEYNG/sango-vocabulary"
VOCAB_FILENAME = "vocabulary.jsonl"


# =======================================================================
# 공용 유틸
# =======================================================================

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def iter_jsonl(path):
    """전체를 리스트로 올리지 않고 한 줄씩 읽는 제너레이터."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl_atomic(path, records_iterable):
    """중간에 끊겨도 완성되지 않은 파일이 [skip] 대상으로 오인되지 않도록
    .tmp에 쓰고 완료 후에만 원자적으로 rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records_iterable:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp_path, path)
    return n


def save_json_atomic(path, obj):
    """🔒 JSON 사이드카(provenance/stats/lineage/audit/manifest 등)도 jsonl과 동일하게
    .tmp에 쓰고 완료 후에만 rename한다. 일반 open(...,'w')+json.dump로 쓰면 중간에
    끊겼을 때 0보다 큰 partial JSON 파일이 남고, outputs_complete()의 size>0 조건이
    이걸 완료로 오판할 수 있다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def outputs_complete(paths):
    """단일 out_path 존재만으로 stage 완료를 판단하면, 중간에 끊겨서 보조 산출물
    (통계·provenance 등)이 안 만들어진 상태를 완료로 오판할 수 있다.
    이 stage의 필수 산출물 전부가 있어야 완료로 본다."""
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_en(text: str) -> str:
    """pair_id / overlap 검사용 canonical 형태. NFC + 공백정규화 + casefold."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text.strip()).casefold()


# Sango 정서법 정규화 (원본 노트북 cell 23 규칙 그대로 재사용)
# 🔒 대소문자별 규칙을 명시해서 casefold 없이도 원문 대소문자를 보존한 채 정규화한다.
_SANGO_ORTHO_RULES = [
    (r"\s+", " "),
    (r"[\u2019']", "'"),
    (r"\u025b", "\u00eb"),    # ɛ -> ë
    (r"\u0190", "\u00cb"),    # Ɛ -> Ë
    (r"\u0254", "\u00f6"),    # ɔ -> ö
    (r"\u0186", "\u00d6"),    # Ɔ -> Ö
    (r"e\u0308", "\u00eb"),   # e + combining diaeresis -> ë
    (r"E\u0308", "\u00cb"),
    (r"e\u0302", "\u00ea"),   # e + combining circumflex -> ê
    (r"E\u0302", "\u00ca"),
    (r"o\u0308", "\u00f6"),
    (r"O\u0308", "\u00d6"),
    (r"o\u0302", "\u00f4"),
    (r"O\u0302", "\u00d4"),
    (r"([iaeouIAEOU])m([bpBP])", r"\1n\2"),
]

def normalize_sango_orthography(text: str) -> str:
    """🔒 저장/최종 정제용 표기 정규화. casefold는 하지 않아 원문 대소문자를 보존한다.
    이전 버전은 정규화 없이 원문 sg를 그대로 script validity 검사·최종 저장에 썼기 때문에,
    정규화하면 ë/ö가 될 ɛ/ɔ 표기 원문이 script-invalid로 잘못 제거되고 direct9k_train.jsonl에도
    비정규화 원문이 그대로 실리는 문제가 있었다. canonical_sg()는 이 결과에 casefold만
    추가로 적용해 ID/overlap 비교용으로 쓴다."""
    text = unicodedata.normalize("NFC", str(text))
    for pattern, repl in _SANGO_ORTHO_RULES:
        text = re.sub(pattern, repl, text)
    return text.strip()


def canonical_sg(text: str) -> str:
    """pair_id / overlap 검사용. 표기 정규화 결과에 casefold를 추가로 적용한다.
    (casefold를 정규화 "다음"에 적용해야 대문자 Ɛ/Ɔ → Ë/Ö로 정상 변환된 뒤 소문자로
    내려가서, 같은 문장의 대소문자 변이가 서로 다른 canonical id를 갖지 않는다.)"""
    return normalize_sango_orthography(text).casefold()


def compute_pair_id(en: str, sg: str) -> str:
    """content-based immutable ID. canonical(NFC+casefold) 형태 기준이라 대소문자·
    조합형 악센트 차이만 있는 문장도 같은 ID로 묶인다."""
    payload = f"{canonical_en(en)}\0{canonical_sg(sg)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seeded_rank(pair_id: str, seed: int) -> str:
    payload = f"{seed}\0{pair_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# =======================================================================
# 필터 함수
# =======================================================================

def filter_pairs_stream(record_iterable, min_len=MIN_LEN, max_len=MAX_LEN,
                         len_ratio_max=LEN_RATIO_MAX):
    """레코드({'en','sg','laser_score'})를 스트리밍으로 받아 길이/중복 필터링.
    laser_score를 레코드에 그대로 실어 옮기므로 별도 lookup dict가 필요 없다
    (기존 laser_lookup의 '마지막 값으로 덮어써지는' 불일치 문제 자체를 제거)."""
    seen = set()
    reasons = {"empty": 0, "exact_duplicate": 0, "below_min_length": 0,
               "above_max_length": 0, "excessive_length_ratio": 0}
    for r in record_iterable:
        src, tgt = r["en"].strip(), r["sg"].strip()
        if not src or not tgt:
            reasons["empty"] += 1
            continue
        key = (src, tgt)
        if key in seen:
            reasons["exact_duplicate"] += 1
            continue
        src_len, tgt_len = len(src.split()), len(tgt.split())
        if src_len < min_len or tgt_len < min_len:
            reasons["below_min_length"] += 1
            continue
        if src_len > max_len or tgt_len > max_len:
            reasons["above_max_length"] += 1
            continue
        ratio = max(src_len, tgt_len) / max(min(src_len, tgt_len), 1)
        if ratio > len_ratio_max:
            reasons["excessive_length_ratio"] += 1
            continue
        seen.add(key)
        yield {"en": src, "sg": tgt, "laser_score": r.get("laser_score")}
    filter_pairs_stream.reasons = reasons
    filter_pairs_stream.n_removed = sum(reasons.values())


_SANGO_ALLOWED_CHARS = re.compile(
    r"^[a-zA-Z\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc'\u2019\-\s.,!?;:()\"0-9]+$"
)
_URL_OR_HTML_PATTERN = re.compile(r"https?://|www\.|<[^>]+>|&[a-z]+;")
_EXCESSIVE_DIGIT_PATTERN = re.compile(r"\d{5,}")

# French 토큰화 — 공백만 split하면 "garder," 같은 콤마 붙은 토큰이 vocab과 매치 안 됨
# (MEYNG의 french 필드가 "garder, préserver, maintenir"처럼 콤마구분 gloss라 실제로 발생하는 문제).
_LEXICAL_TOKEN_RE = re.compile(
    r"[a-z\u00e0\u00e2\u00e4\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00ff\u0153\u00e6]+"
    r"(?:[-'\u2019][a-z\u00e0\u00e2\u00e4\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00ff\u0153\u00e6]+)*",
    re.IGNORECASE,
)

def lexical_tokens(text: str) -> list:
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    return _LEXICAL_TOKEN_RE.findall(normalized)


def check_script_validity(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return False
    # 🔒 _SANGO_ALLOWED_CHARS는 소문자 발음구별기호(ë/ö/ê/ô 등)만 등재돼 있다.
    # normalize_sango_orthography()가 대문자 입력(Ɛ/Ɔ)을 Ë/Ö로 정상 변환해도, 그 결과인
    # 대문자 Ë/Ö가 여기서 다시 script-invalid로 탈락하던 버그가 있었다.
    # casefold() 대신 lower()를 쓰는 이유: 단순 대소문자 대응 검사가 목적이라, 일부 문자를
    # 여러 글자로 확장할 수 있는 casefold보다 1:1 대응인 lower가 이 용도에 더 적합하다.
    allowed = sum(1 for c in text if _SANGO_ALLOWED_CHARS.match(c.lower()) or c.isspace())
    return (allowed / max(len(text), 1)) >= 0.9


def check_degenerate_pattern(text: str) -> bool:
    if _URL_OR_HTML_PATTERN.search(text):
        return False
    if _EXCESSIVE_DIGIT_PATTERN.search(text):
        return False
    return True


def flag_code_switching(sag_text: str, french_vocab: set, threshold: float) -> bool:
    """🔒 fail-open 버그 수정: vocab이 비어있으면 통과가 아니라 즉시 RuntimeError."""
    if not french_vocab:
        raise RuntimeError(
            "flag_code_switching: french_vocab이 비어있음. "
            "load_sango_vocab()이 먼저 성공해야 auto_clean_pairs를 호출할 수 있음."
        )
    tokens = lexical_tokens(sag_text)
    if not tokens:
        return False
    fr_count = sum(1 for t in tokens if t in french_vocab)
    return (fr_count / len(tokens)) > threshold


def check_not_too_french(sag_text: str, french_vocab: set, threshold: float = FRENCH_THRESHOLD) -> bool:
    return not flag_code_switching(sag_text, french_vocab, threshold)


def auto_clean_pairs_stream(record_iterable, french_vocab: set,
                             french_threshold: float = FRENCH_THRESHOLD):
    """script validity -> degenerate -> French contamination 순서로 스트리밍 필터링.
    french_vocab이 비어있으면 즉시 실패.
    🔒 sg는 검사 전에 normalize_sango_orthography()로 표기 정규화한다 — 정규화 없이 원문을
    그대로 검사하면, 정규화하면 ë/ö가 될 ɛ/ɔ 표기 문장이 script-invalid로 잘못 제거된다.
    yield하는 레코드는 정규화된 sg와 원문 sg_raw를 함께 싣는다."""
    if not french_vocab:
        raise RuntimeError(
            "auto_clean_pairs_stream: french_vocab 없이는 실행할 수 없음. "
            "MEYNG vocabulary 로딩을 먼저 완료할 것."
        )
    reasons = {"script": 0, "degenerate": 0, "french_contamination": 0}
    samples = {"script": [], "degenerate": [], "french_contamination": []}
    n_pass = 0
    for r in record_iterable:
        en, sg_raw = r["en"], r["sg"]
        sg = normalize_sango_orthography(sg_raw)
        if not check_script_validity(sg):
            reasons["script"] += 1
            if len(samples["script"]) < 50:
                samples["script"].append({"en": en, "sg": sg_raw})
            continue
        if not check_degenerate_pattern(en) or not check_degenerate_pattern(sg):
            reasons["degenerate"] += 1
            if len(samples["degenerate"]) < 50:
                samples["degenerate"].append({"en": en, "sg": sg_raw})
            continue
        if not check_not_too_french(sg, french_vocab, french_threshold):
            reasons["french_contamination"] += 1
            if len(samples["french_contamination"]) < 50:
                samples["french_contamination"].append({"en": en, "sg": sg_raw})
            continue
        n_pass += 1
        yield {"en": en, "sg": sg, "sg_raw": sg_raw, "laser_score": r.get("laser_score")}
    auto_clean_pairs_stream.reasons = reasons
    auto_clean_pairs_stream.samples = samples
    auto_clean_pairs_stream.n_pass = n_pass


# =======================================================================
# Stage A: source acquisition — allenai/nllb 로드 + LASER 필터 (lazy, 배치 스트리밍)
# =======================================================================

def _check_datasets_version():
    from packaging.version import Version
    import datasets
    if Version(datasets.__version__) >= Version("4.0.0"):
        raise RuntimeError(
            f"현재 datasets=={datasets.__version__}. allenai/nllb는 legacy loading "
            f"script(trust_remote_code)를 쓰는데, datasets>=4.0.0에서 이 기능이 완전히 "
            f"제거됨(RuntimeError: 'Dataset scripts are no longer supported'). "
            f"먼저 실행할 것: pip install -q \"datasets<4.0.0\" (설치 후 런타임 재시작 필요)."
        )


def stage_source(force=False):
    out_path = OUTPUT_DIR / "after_laser.jsonl"
    prov_path = OUTPUT_DIR / "source_provenance.json"
    if outputs_complete([out_path, prov_path]) and not force:
        print(f"[skip] source stage 산출물 모두 존재 (--force로 재실행 가능)")
        return
    # 🔒 --force일 때만이 아니라, 산출물 일부가 없어 자연스럽게 재실행되는 경우에도
    # downstream을 무효화한다. (예: 이전 실행은 완전히 끝났는데 누군가 산출물 하나를
    # 수동으로 지워서 불완전 상태가 됐다면, --force 없이도 재실행되면서 낡은 downstream이
    # 그대로 남아있을 수 있음)
    invalidate_downstream("source")
    invalidate_current_stage("source")

    _check_datasets_version()

    from datasets import load_dataset
    import datasets as datasets_pkg
    from huggingface_hub import HfApi

    # repo SHA를 로드 "전에" 조회해서 revision으로 고정 — provenance 신뢰도 향상
    # 🔒 논문용 최종 데이터 구축이므로, revision을 못 고정하면 진행하지 않고 중단한다.
    # (조회 실패 상태로 계속 진행하면 재현성 보장이 깨진 채로 pool이 만들어질 수 있음)
    api = HfApi()
    try:
        repo_sha = api.dataset_info(SRC_DATASET).sha
    except Exception as e:
        raise RuntimeError(
            f"NLLB repository revision을 고정하지 못해 데이터 구축을 중단함: {e}"
        )

    print(f"[source] {SRC_DATASET} / {SRC_CONFIG} 로딩 중... (revision={repo_sha})")
    load_kwargs = {"trust_remote_code": True, "revision": repo_sha}
    # nllb.py 소스 확인 결과 SplitGenerator가 항상 datasets.Split.TRAIN을 씀 -> "train"으로
    # 명시 고정하는 게 list(ds.keys())[0]보다 결정적. 혹시 다르면 폴백으로 첫 split 사용.
    try:
        split = load_dataset(SRC_DATASET, SRC_CONFIG, split="train", **load_kwargs)
        split_name = "train"
    except (KeyError, ValueError) as e:
        print(f"[source] split='train' 직접 지정 실패({e}) — 첫 번째 split으로 폴백")
        ds = load_dataset(SRC_DATASET, SRC_CONFIG, **load_kwargs)
        split_name = list(ds.keys())[0]
        split = ds[split_name]
    loaded_rows = len(split)
    print(f"[source] 로드 완료: split='{split_name}', {loaded_rows}쌍")

    # LASER 필터를 Arrow 레벨에서 lazy하게 적용 (전체를 파이썬 리스트로 올리지 않음)
    filtered = split.filter(
        lambda row: row["laser_score"] is not None and row["laser_score"] >= LASER_THRESHOLD
    )
    n_after_laser = len(filtered)
    print(f"[source] LASER>={LASER_THRESHOLD} 필터 후: {n_after_laser}쌍 "
          f"(미달/결측 {loaded_rows - n_after_laser}쌍 제외)")

    def extract_batch(batch):
        translations = batch["translation"]
        return {
            "src": [t["eng_Latn"] for t in translations],
            "tgt": [t["sag_Latn"] for t in translations],
            "laser_score": batch["laser_score"],
        }

    mapped = filtered.map(extract_batch, batched=True, batch_size=BATCH_SIZE,
                           remove_columns=filtered.column_names)

    # 배치 단위로 스트리밍 write — 전체를 records 리스트로 한 번에 들고 있지 않음
    def gen():
        for i in range(0, len(mapped), BATCH_SIZE):
            batch = mapped[i:i + BATCH_SIZE]
            for src, tgt, score in zip(batch["src"], batch["tgt"], batch["laser_score"]):
                yield {"en": src, "sg": tgt, "laser_score": float(score)}

    n_written = save_jsonl_atomic(out_path, gen())
    print(f"[source] after_laser.jsonl 저장: {n_written}쌍")

    provenance = {
        "dataset": SRC_DATASET,
        "config": SRC_CONFIG,
        "revision_used": repo_sha,
        "note": "revision은 HF repo(로딩 스크립트)의 commit sha다. 실제 병렬 데이터는 "
                "AllenAI GCS(storage.googleapis.com/allennlp-data-bucket/nllb/)에서 별도로 "
                "받아오므로, GCS 파일 자체의 byte-level 불변성까지는 보장하지 않는다. "
                "그래서 final_clean_pool의 content-based pair_id와 SHA-256을 별도로 보존한다.",
        "datasets_library_version": datasets_pkg.__version__,
        "loaded_rows_this_run": loaded_rows,
        "laser_threshold": LASER_THRESHOLD,
        "rows_after_laser_filter": n_written,
    }
    save_json_atomic(prov_path, provenance)
    print(f"[source] provenance 저장: {prov_path}")


# =======================================================================
# Stage B: deterministic cleaning (스트리밍)
# =======================================================================

def load_sango_vocab():
    from huggingface_hub import hf_hub_download, HfApi

    # revision을 다운로드 "전에" 고정 — 순서를 반대로 하면 metadata에 적힌 SHA와
    # 실제로 받은 파일의 revision이 다를 수 있음 (경합 상태에서 repo가 그 사이 갱신될 경우).
    # 🔒 논문용 최종 데이터 구축이므로, revision을 못 고정하면 진행하지 않고 중단한다.
    try:
        vocab_sha = HfApi().dataset_info(VOCAB_DATASET).sha
    except Exception as e:
        raise RuntimeError(
            f"MEYNG vocabulary repository revision을 고정하지 못해 데이터 구축을 중단함: {e}"
        )

    vocab_path = hf_hub_download(VOCAB_DATASET, VOCAB_FILENAME,
                                  repo_type="dataset", revision=vocab_sha)
    with open(vocab_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    if not entries:
        raise RuntimeError(f"{VOCAB_DATASET}/{VOCAB_FILENAME} 로드 결과가 비어있음.")

    required_fields = {"sango", "french"}
    missing = required_fields - set(entries[0].keys())
    if missing:
        raise RuntimeError(f"vocabulary 스키마 불일치, 누락 필드: {sorted(missing)}")

    # 콤마구분 gloss("garder, préserver, maintenir")를 정규식 토큰화로 안전하게 분해
    french_vocab = set()
    for row in entries:
        fr = row.get("french")
        if fr:
            french_vocab.update(lexical_tokens(fr))

    metadata = {
        "dataset": VOCAB_DATASET,
        "filename": VOCAB_FILENAME,
        "repo_sha_at_fetch_time": vocab_sha,
        "n_entries": len(entries),
        "n_french_tokens": len(french_vocab),
        "source_file_sha256": sha256_file(vocab_path),
    }
    return french_vocab, metadata


def stage_clean(force=False):
    src_path = OUTPUT_DIR / "after_laser.jsonl"
    out_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    stats_path = OUTPUT_DIR / "filter_statistics.json"
    excluded_path = OUTPUT_DIR / "excluded_pairs_sample.jsonl"
    vocab_meta_path = OUTPUT_DIR / "vocabulary_provenance.json"

    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} 없음. --stage source 먼저 실행할 것.")
    # out_path 존재만으로 완료 판단하면, final_clean_pool.jsonl까지는 썼지만
    # filter_statistics.json/vocabulary_provenance.json이 안 만들어진 중단 상태를
    # 완료로 오판할 수 있다 (manifest stage가 이후 filter_statistics.json을 필수로 요구함).
    # excluded_pairs_sample.jsonl은 제외 사례가 실제로 0건이면 정당하게 빈 파일일 수
    # 있으므로 outputs_complete()의 size>0 조건 대신 존재 여부만 확인한다.
    required_complete = outputs_complete([out_path, stats_path, vocab_meta_path])
    if required_complete and excluded_path.exists() and not force:
        print(f"[skip] clean stage 필수 산출물 모두 존재 (--force로 재실행 가능)")
        return
    invalidate_downstream("clean")
    invalidate_current_stage("clean")

    with open(src_path, encoding="utf-8") as f:
        n_after_laser = sum(1 for _ in f)
    print(f"[clean] after_laser.jsonl: {n_after_laser}쌍 (스트리밍 처리)")

    print("[clean] MEYNG vocabulary 로딩 중 (French contamination 검사 필수 의존성)...")
    french_vocab, vocab_meta = load_sango_vocab()
    save_json_atomic(vocab_meta_path, vocab_meta)
    print(f"[clean] vocab 로드 완료: {vocab_meta['n_entries']}항목, "
          f"French 토큰 {vocab_meta['n_french_tokens']}개 (정규식 토큰화 적용)")

    # 스트리밍 파이프라인: length/dup 필터 -> auto_clean -> pair_id 부여 -> write
    length_filtered_stream = filter_pairs_stream(iter_jsonl(src_path))
    cleaned_stream = auto_clean_pairs_stream(length_filtered_stream, french_vocab)

    seen_ids = set()
    n_id_duplicate = 0

    def gen():
        nonlocal n_id_duplicate
        for r in cleaned_stream:
            pid = compute_pair_id(r["en"], r["sg"])
            if pid in seen_ids:
                n_id_duplicate += 1
                continue
            seen_ids.add(pid)
            yield {
                "pair_id": pid,
                "en": r["en"],
                "sg": r["sg"],
                "sg_raw": r["sg_raw"],
                "laser_score": r["laser_score"],
                "en_word_len": len(r["en"].split()),
                "sg_word_len": len(r["sg"].split()),
            }

    n_final = save_jsonl_atomic(out_path, gen())
    pool_sha = sha256_file(out_path)

    length_reasons = filter_pairs_stream.reasons
    clean_reasons = auto_clean_pairs_stream.reasons
    excluded_samples = auto_clean_pairs_stream.samples

    print(f"[clean] final_clean_pool.jsonl 저장: {n_final}쌍 "
          f"(pair_id 중복 {n_id_duplicate}개 추가 제거), SHA-256={pool_sha[:16]}...")

    stats = {
        "input_after_laser": n_after_laser,
        "removed_empty": length_reasons["empty"],
        "removed_exact_duplicate": length_reasons["exact_duplicate"],
        "removed_below_min_length": length_reasons["below_min_length"],
        "removed_above_max_length": length_reasons["above_max_length"],
        "removed_excessive_length_ratio": length_reasons["excessive_length_ratio"],
        "removed_script_invalid": clean_reasons["script"],
        "removed_degenerate": clean_reasons["degenerate"],
        "removed_french_contamination": clean_reasons["french_contamination"],
        "removed_pair_id_duplicate": n_id_duplicate,
        "final_clean_pool_size": n_final,
        "final_clean_pool_sha256": pool_sha,
    }
    save_json_atomic(stats_path, stats)

    all_excluded = []
    for reason, samples in excluded_samples.items():
        for s in samples:
            all_excluded.append({"reason": reason, **s})
    save_jsonl_atomic(excluded_path, all_excluded)

    print(f"[clean] 통계 저장: {stats_path}")
    print(f"[clean] 제외 샘플 저장: {excluded_path} ({len(all_excluded)}개)")


# =======================================================================
# Stage C: candidate isolation — largest-remainder allocation으로 정확히 CANDIDATE_COUNT
# =======================================================================

def _en_length_bucket(n_words):
    if n_words <= 8:
        return "short"
    elif n_words <= 20:
        return "medium"
    return "long"


def _laser_bucket(score):
    if score is None:
        return "unknown"
    if score < 1.1:
        return "low"
    elif score < 1.3:
        return "mid"
    return "high"


def stage_candidates(force=False):
    pool_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    out_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    ids_path = OUTPUT_DIR / "candidate_ids.json"

    if not pool_path.exists():
        raise FileNotFoundError(f"{pool_path} 없음. --stage clean 먼저 실행할 것.")
    # out_path(evaluation_candidates.jsonl)만 보고 완료 판단하면, candidate_ids.json
    # (lineage checksum 포함 — audit/train이 필수로 요구) 쓰기 전에 끊긴 상태를
    # 완료로 오판할 수 있다. 두 산출물 모두 있어야 완료로 본다.
    if outputs_complete([out_path, ids_path]) and not force:
        print(f"[skip] candidates stage 필수 산출물 모두 존재 (--force로 재실행 가능)")
        return
    invalidate_downstream("candidates")
    invalidate_current_stage("candidates")

    pool_sha_parent = sha256_file(pool_path)  # lineage 기록용 — audit에서 재검증

    # ⚠️ pool 크기가 수십만~수백만이면 여기서 전체를 메모리에 올리는 것이 부담될 수 있음.
    # --stage clean 로그의 "final_clean_pool_size"를 먼저 확인할 것. 규모가 크면
    # (heap 기반 streaming top-k 선택으로 교체 필요 — 현재는 300~350 후보 규모라
    # 수만~수십만 단위 pool까지는 이 방식으로도 충분히 감당 가능할 것으로 예상됨)
    pool = load_jsonl(pool_path)  # 최종 pool은 300~350 후보 선정을 위해 메모리에 올려야 함
    print(f"[candidates] pool 로드: {len(pool)}쌍")

    strata = {}
    for r in pool:
        key = (_en_length_bucket(r["en_word_len"]), _laser_bucket(r["laser_score"]))
        strata.setdefault(key, []).append(r)

    total = len(pool)

    # largest-remainder allocation — quota 합이 정확히 CANDIDATE_COUNT가 되도록 보장
    raw_quotas = {key: CANDIDATE_COUNT * len(group) / total for key, group in strata.items()}
    quotas = {key: int(v) for key, v in raw_quotas.items()}
    remaining = CANDIDATE_COUNT - sum(quotas.values())
    priority = sorted(strata.keys(), key=lambda k: (-(raw_quotas[k] - quotas[k]), str(k)))
    for key in priority[:remaining]:
        quotas[key] += 1

    assert sum(quotas.values()) == CANDIDATE_COUNT, \
        f"quota 합계 불일치: {sum(quotas.values())} != {CANDIDATE_COUNT}"

    # canonical EN/SG 전역 고유성을 선택 시점에 강제 — 같은 source가 evaluation set
    # 안에서 반복되면 문장 단위 bootstrap이 사실상 같은 입력을 중복 표집하게 되고,
    # 원어민 검증 부담도 낭비되므로 사후 감지가 아니라 선택 단계에서 원천 차단.
    used_en, used_sg = set(), set()
    selected = []
    for key in sorted(strata.keys(), key=str):  # dict 순회 순서 비의존적으로 정렬 후 처리
        group = strata[key]
        ranked = sorted(group, key=lambda r: seeded_rank(r["pair_id"], CANDIDATE_SEED))
        chosen = []
        for r in ranked:
            en_key, sg_key = canonical_en(r["en"]), canonical_sg(r["sg"])
            if en_key in used_en or sg_key in used_sg:
                continue
            chosen.append(r)
            used_en.add(en_key)
            used_sg.add(sg_key)
            if len(chosen) == quotas[key]:
                break
        selected.extend(chosen)
        print(f"[candidates] stratum {key}: pool {len(group)}개 중 {len(chosen)}개 선택 "
              f"(quota={quotas[key]}, EN/SG 고유성 적용)")

    # quota 미달분(중복 제거로 일부 stratum이 못 채워진 경우) 전체 미선택 pool에서
    # 동일한 고유성 조건으로 보충
    if len(selected) < CANDIDATE_COUNT:
        selected_ids = {r["pair_id"] for r in selected}
        remainder = [r for r in pool if r["pair_id"] not in selected_ids]
        remainder_ranked = sorted(remainder, key=lambda r: seeded_rank(r["pair_id"], CANDIDATE_SEED))
        n_needed = CANDIDATE_COUNT - len(selected)
        backfilled = 0
        for r in remainder_ranked:
            if backfilled >= n_needed:
                break
            en_key, sg_key = canonical_en(r["en"]), canonical_sg(r["sg"])
            if en_key in used_en or sg_key in used_sg:
                continue
            selected.append(r)
            used_en.add(en_key)
            used_sg.add(sg_key)
            backfilled += 1
        print(f"[candidates] 고유성 제약으로 인한 quota 미달분 보충: {backfilled}개")

    # 최종 선택을 seeded_rank로 재정렬 -> strata dict 삽입 순서(파이썬 버전/실행마다
    # 달라질 수 있음)에 output 순서가 의존하지 않게 함
    selected = sorted(selected, key=lambda r: seeded_rank(r["pair_id"], CANDIDATE_SEED))

    assert len(selected) == CANDIDATE_COUNT, \
        f"최종 후보 수 불일치: {len(selected)} != {CANDIDATE_COUNT} " \
        f"(pool 안에 고유 EN/SG 조합이 부족할 수 있음)"

    save_jsonl_atomic(out_path, selected)

    candidate_ids = sorted(r["pair_id"] for r in selected)
    save_json_atomic(ids_path, {
        "seed": CANDIDATE_SEED, "count": len(candidate_ids),
        "pair_ids": candidate_ids,
        "parent_final_clean_pool_sha256": pool_sha_parent,
    })

    print(f"[candidates] evaluation_candidates.jsonl 저장: {len(selected)}개 (quota 정확히 일치)")


# =======================================================================
# Stage D: Direct-9K 신규 추출 — pair_id뿐 아니라 canonical EN/SG까지 제외
# =======================================================================

def stage_train(force=False):
    pool_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    cand_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    out_path = OUTPUT_DIR / "direct9k_train.jsonl"

    if not pool_path.exists():
        raise FileNotFoundError(f"{pool_path} 없음. --stage clean 먼저 실행할 것.")
    if not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} 없음. --stage candidates 먼저 실행할 것.")
    lineage_path = OUTPUT_DIR / "direct9k_train_lineage.json"
    # out_path(direct9k_train.jsonl)만 보고 완료 판단하면, direct9k_train_lineage.json
    # (audit의 lineage 재검증이 필수로 요구) 쓰기 전에 끊긴 상태를 완료로 오판할 수 있다.
    if outputs_complete([out_path, lineage_path]) and not force:
        print(f"[skip] train stage 필수 산출물 모두 존재 (--force로 재실행 가능)")
        return
    invalidate_downstream("train")
    invalidate_current_stage("train")

    pool_sha_parent = sha256_file(pool_path)
    cand_sha_parent = sha256_file(cand_path)

    pool = load_jsonl(pool_path)
    candidates = load_jsonl(cand_path)

    candidate_ids = {r["pair_id"] for r in candidates}
    # pair_id뿐 아니라 canonical EN/SG도 제외 -> 같은 문장이 다른 상대와 짝지어져
    # 다른 pair_id를 갖는 경우까지 train에서 선제 차단 (audit에서 뒤늦게 걸리는 것 방지)
    candidate_en = {canonical_en(r["en"]) for r in candidates}
    candidate_sg = {canonical_sg(r["sg"]) for r in candidates}

    eligible = [
        r for r in pool
        if r["pair_id"] not in candidate_ids
        and canonical_en(r["en"]) not in candidate_en
        and canonical_sg(r["sg"]) not in candidate_sg
    ]
    print(f"[train] pool {len(pool)}개 중 candidate(pair_id/EN/SG 기준) 제외 -> "
          f"eligible {len(eligible)}개")

    if len(eligible) < TRAIN_COUNT:
        raise RuntimeError(
            f"eligible pool({len(eligible)})이 TRAIN_COUNT({TRAIN_COUNT})보다 작음. "
            f"LASER threshold를 낮추거나 필터 기준을 재검토할 것."
        )

    ranked = sorted(eligible, key=lambda r: seeded_rank(r["pair_id"], TRAIN_SEED))
    selected = ranked[:TRAIN_COUNT]

    records = []
    for i, r in enumerate(selected, start=1):
        records.append({
            "sentence_id": f"direct9k_{i:05d}",
            "pair_id": r["pair_id"],
            "en": r["en"],
            "sg": r["sg"],
            "sg_raw": r["sg_raw"],
        })
    save_jsonl_atomic(out_path, records)
    save_json_atomic(lineage_path, {
        "parent_final_clean_pool_sha256": pool_sha_parent,
        "parent_evaluation_candidates_sha256": cand_sha_parent,
    })
    print(f"[train] direct9k_train.jsonl 저장: {len(records)}쌍, seed={TRAIN_SEED}")
    print(f"[train] lineage 기록: {lineage_path}")


# =======================================================================
# Stage E: audit — overlap·count·무결성 검사, 전부 hard failure
# =======================================================================

def stage_audit():
    pool_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    cand_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    ids_path = OUTPUT_DIR / "candidate_ids.json"
    train_path = OUTPUT_DIR / "direct9k_train.jsonl"
    lineage_path = OUTPUT_DIR / "direct9k_train_lineage.json"
    report_path = OUTPUT_DIR / "audit_report.json"

    for p in [pool_path, cand_path, ids_path, train_path, lineage_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음. 이전 stage를 먼저 실행할 것.")

    candidates = load_jsonl(cand_path)
    train = load_jsonl(train_path)

    cand_ids = {r["pair_id"] for r in candidates}
    train_ids = {r["pair_id"] for r in train}

    cand_en_norm = {canonical_en(r["en"]) for r in candidates}
    train_en_norm = {canonical_en(r["en"]) for r in train}
    cand_sg_norm = {canonical_sg(r["sg"]) for r in candidates}
    train_sg_norm = {canonical_sg(r["sg"]) for r in train}

    # lineage 재검증 — --force로 stage 일부만 재실행돼서 서로 다른 pool/candidate
    # 세대가 섞인 manifest가 만들어지는 상황을 여기서 잡아냄.
    current_pool_sha = sha256_file(pool_path)
    current_cand_sha = sha256_file(cand_path)
    with open(ids_path, encoding="utf-8") as f:
        cand_lineage = json.load(f)
    with open(lineage_path, encoding="utf-8") as f:
        train_lineage = json.load(f)

    lineage_ok = (
        cand_lineage.get("parent_final_clean_pool_sha256") == current_pool_sha
        and train_lineage.get("parent_final_clean_pool_sha256") == current_pool_sha
        and train_lineage.get("parent_evaluation_candidates_sha256") == current_cand_sha
    )

    # candidate_ids.json은 evaluation_candidates.jsonl과 별도로 쓰이는 sidecar라, 재실행
    # 도중 한쪽만 새로 쓰이고 한쪽은 이전 세대로 남는 상황(invalidate_current_stage가
    # 있어도 --force 없이 수동으로 파일을 건드린 경우 등)을 여기서 한 번 더 잡는다.
    sidecar_candidate_ids = set(cand_lineage.get("pair_ids", []))
    candidate_sidecar_ok = (
        sidecar_candidate_ids == cand_ids
        and cand_lineage.get("count") == len(candidates)
        and cand_lineage.get("seed") == CANDIDATE_SEED
    )

    checks = {
        "pair_id_disjoint": cand_ids.isdisjoint(train_ids),
        "candidate_count_matches": len(candidates) == len(cand_ids) == CANDIDATE_COUNT,
        "train_count_matches": len(train) == len(train_ids) == TRAIN_COUNT,
        "train_ids_unique": len(train_ids) == len(train),
        "candidate_ids_unique": len(cand_ids) == len(candidates),
        "candidate_en_unique": len(candidates) == len(cand_en_norm),
        "candidate_sg_unique": len(candidates) == len(cand_sg_norm),
        "normalized_en_overlap": len(cand_en_norm & train_en_norm),
        "normalized_sg_overlap": len(cand_sg_norm & train_sg_norm),
        "lineage_consistent": lineage_ok,
        "candidate_sidecar_matches": candidate_sidecar_ok,
    }

    print("[audit] 검사 결과:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    hard_failures = []
    if not checks["pair_id_disjoint"]:
        hard_failures.append("pair_id_disjoint 위반 — evaluation candidate와 train이 겹침")
    if not checks["candidate_count_matches"]:
        hard_failures.append(f"candidate count mismatch (expected {CANDIDATE_COUNT})")
    if not checks["train_count_matches"]:
        hard_failures.append(f"train count mismatch (expected {TRAIN_COUNT})")
    if checks["normalized_en_overlap"] > 0:
        hard_failures.append(f"normalized EN overlap {checks['normalized_en_overlap']}건")
    if checks["normalized_sg_overlap"] > 0:
        hard_failures.append(f"normalized SG overlap {checks['normalized_sg_overlap']}건")
    if not checks["candidate_sidecar_matches"]:
        hard_failures.append(
            "candidate_ids.json이 evaluation_candidates.jsonl과 일치하지 않음 "
            "— 서로 다른 세대의 파일이 섞였을 가능성"
        )
    if not checks["train_ids_unique"] or not checks["candidate_ids_unique"]:
        hard_failures.append("train 또는 candidate 내부에 중복 pair_id 존재")
    if not checks["candidate_en_unique"]:
        hard_failures.append("evaluation candidate 내부에 동일 EN이 서로 다른 SG와 중복 등장")
    if not checks["candidate_sg_unique"]:
        hard_failures.append("evaluation candidate 내부에 동일 SG가 서로 다른 EN과 중복 등장")
    if not checks["lineage_consistent"]:
        hard_failures.append(
            "lineage checksum 불일치 — --force로 일부 stage만 재실행돼서 "
            "서로 다른 세대의 pool/candidate/train이 섞였을 가능성. 전체를 --force로 "
            "재실행하거나 candidates/train부터 다시 만들 것."
        )

    save_json_atomic(report_path, {
        "checks": checks,
        "hard_failures": hard_failures,
        # 🔒 audit이 통과했다는 사실만으로는, standalone `--stage manifest` 실행 시점에
        # 파일들이 audit 당시와 "같은" 파일인지 보장 못 한다 (audit 이후 누군가 train
        # 파일을 수정하고 manifest만 다시 돌리는 경우). 당시 해시를 남겨서 manifest가
        # 재검증할 수 있게 한다.
        "artifact_hashes": {
            "final_clean_pool_sha256": current_pool_sha,
            "evaluation_candidates_sha256": current_cand_sha,
            "direct9k_train_sha256": sha256_file(train_path),
        },
    })

    if hard_failures:
        print("\n[audit] 🚨 FAILED:")
        for msg in hard_failures:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print(f"\n[audit] ✅ 모든 검사 통과. 리포트: {report_path}")


# =======================================================================
# manifest 취합
# =======================================================================

def stage_manifest():
    prov_path = OUTPUT_DIR / "source_provenance.json"
    vocab_meta_path = OUTPUT_DIR / "vocabulary_provenance.json"
    stats_path = OUTPUT_DIR / "filter_statistics.json"
    pool_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    cand_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    train_path = OUTPUT_DIR / "direct9k_train.jsonl"
    audit_path = OUTPUT_DIR / "audit_report.json"
    out_path = OUTPUT_DIR / "prevalidation_manifest.json"

    required = [prov_path, vocab_meta_path, stats_path, pool_path, cand_path, train_path, audit_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"다음 파일들이 없음 — 이전 stage를 먼저 완료할 것: {missing}")

    with open(prov_path, encoding="utf-8") as f:
        provenance = json.load(f)
    with open(vocab_meta_path, encoding="utf-8") as f:
        vocab_meta = json.load(f)
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)

    # 🔒 audit_report.json이 "존재"하는 것과 audit이 "통과"한 것은 다르다.
    # 이전 버전은 파일 존재만 확인해서, audit이 hard failure로 sys.exit(1)한 뒤에도
    # (예: 사람이 그 사실을 못 보고 그냥 다음 명령을 실행하면) manifest가 그대로
    # 만들어질 수 있었다. audit 실패 상태로 만든 manifest는 논문에 쓸 수 없으므로 차단한다.
    if audit.get("hard_failures"):
        raise RuntimeError(
            "Audit가 실패했으므로 manifest를 생성할 수 없음: "
            + "; ".join(audit["hard_failures"])
        )
    if not audit.get("checks", {}).get("lineage_consistent", False):
        raise RuntimeError(
            "Lineage 검사가 통과하지 않음 — manifest를 생성할 수 없음. "
            "audit_report.json의 lineage_consistent를 확인할 것."
        )

    # 🔒 audit 통과 여부는 확인했지만, 그 audit이 "현재" 파일들을 대상으로 한 것인지는
    # 아직 모른다. `--stage audit` 이후 누군가 direct9k_train.jsonl 등을 수정하고
    # `--stage manifest`만 단독 실행하면, 낡은 audit 결과가 현재 파일과 함께 manifest에
    # 실릴 수 있다. audit 당시 해시와 지금 해시를 비교해서 이걸 막는다.
    current_hashes = {
        "final_clean_pool_sha256": sha256_file(pool_path),
        "evaluation_candidates_sha256": sha256_file(cand_path),
        "direct9k_train_sha256": sha256_file(train_path),
    }
    audit_hashes = audit.get("artifact_hashes")
    if audit_hashes is None:
        raise RuntimeError(
            "audit_report.json에 artifact_hashes가 없음 (구버전 audit 결과로 보임). "
            "--stage audit을 다시 실행할 것."
        )
    if audit_hashes != current_hashes:
        raise RuntimeError(
            "Audit 이후 데이터 파일이 변경됨 — audit 결과가 현재 파일과 더 이상 일치하지 "
            "않으므로 manifest를 생성할 수 없음. --stage audit을 다시 실행할 것."
        )

    manifest = {
        "source": provenance,
        "vocabulary": vocab_meta,
        "parameters": {
            "laser_threshold": LASER_THRESHOLD,
            "min_len": MIN_LEN, "max_len": MAX_LEN, "len_ratio_max": LEN_RATIO_MAX,
            "french_threshold": FRENCH_THRESHOLD,
            "candidate_seed": CANDIDATE_SEED,
            "train_seed": TRAIN_SEED,
            "candidate_count": CANDIDATE_COUNT,
            "train_count": TRAIN_COUNT,
        },
        "filter_statistics": stats,
        "artifacts": current_hashes,
        "audit": audit,
    }
    save_json_atomic(out_path, manifest)
    print(f"[manifest] 저장 완료: {out_path}")


# =======================================================================
# CLI
# =======================================================================

STAGE_FUNCS = {
    "source": stage_source,
    "clean": stage_clean,
    "candidates": stage_candidates,
    "train": stage_train,
    "audit": stage_audit,
    "manifest": stage_manifest,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=list(STAGE_FUNCS.keys()) + ["all"])
    parser.add_argument("--force", action="store_true",
                         help="이미 존재하는 stage 출력을 덮어쓰고 재실행")
    args = parser.parse_args()

    if args.stage == "all":
        stage_source(force=args.force)
        stage_clean(force=args.force)
        stage_candidates(force=args.force)
        stage_train(force=args.force)
        stage_audit()
        stage_manifest()
    elif args.stage in ("audit", "manifest"):
        STAGE_FUNCS[args.stage]()
    else:
        STAGE_FUNCS[args.stage](force=args.force)
