# =====================================================================
# export_deo_evaluation_review.py — 350개 evaluation candidate 검증용 Deo 배포 파일
#
# 🔒 이건 French contamination audit(audit_dedup_and_laser.py)과 완전히 다른 작업이다.
#   - French audit 대상: 자동 필터에서 "탈락한" 200,958건 중 표본 -> 필터 타당성 점검용
#   - 이 스크립트 대상: 모든 필터를 "통과한" 최종 clean pool에서 뽑은 350개 evaluation
#     candidate -> 논문의 실제 adaptation-dev/adaptation-test gold reference가 될 문장들
#   350개는 Direct-9K 학습 데이터와 이미 pair_id/EN/SG 전역 기준으로 격리돼 있음.
#
# 판정 체계 (B는 "제한적 수정"만 — 전체 재번역이 필요하면 C):
#   A. Usable as-is / B. Usable after limited revision / C. Not usable / D. Uncertain
#
# 🔒 3차 코드 리뷰 반영 (v2 -> v3):
#   6. reserve 자동 계산이 "아직 export도 안 된 나머지 배치"를 반영하지 않던 문제 수정 —
#      전체 candidate pool 크기에서 지금까지 실제로 merge된 행 수를 뺀 것도 "미판정
#      대기분"으로 잡아서 같은 통과율로 투영한다.
#   7. Issue type 옵션에 쉼표가 포함돼 있어서(예: "Incorrect name, number, or citation")
#      inline 문자열 데이터 검증(formula1='"A,B,C"')을 쓰면 Excel이 쉼표를 항목 구분자로
#      해석해 옵션이 쪼개질 수 있었음 -> 시트의 숨김 열(J/K)에 옵션을 쓰고 그 범위를
#      참조하는 방식으로 변경. merge에서도 issue_type이 정확한 옵션인지 검증.
#   8. reserve를 두 번 다른 크기로 만들면 reserve_candidates.jsonl이 덮어써져서 예전
#      reserve batch가 source index를 못 찾고 merge가 깨지던 문제 -> reserve를
#      round(reserve_round_01.jsonl, reserve_round_02.jsonl...)로 분리, 새 round는 이전
#      모든 round + 기존 candidate/train을 다 제외하고 뽑는다. source index도 모든
#      round 파일을 다 읽는다.
#   9. 판정된(Decision이 채워진) 행의 불완전/잘못된 입력(B인데 수정문 없음, B/C인데
#      issue type 없음, 잘못된 Decision/issue type 문자열)을 경고만 하고 넘어가던 것을
#      hard failure로 변경 — 미판정(빈 Decision) 행만 정상으로 허용한다.
#   추가: _existing_file_has_answers()가 Decision 열만 보던 것을 D~G 전체로 확장,
#   예시 행이 검증 안 된 Sango 문법 주장을 담고 있던 것을 형식 설명만 남기도록 수정.
#
# 사용법 (build_final_splits.py와 같은 폴더에서):
#   python3 export_deo_evaluation_review.py --export              # 350개 -> 7개 xlsx 배치
#   python3 export_deo_evaluation_review.py --export-reserve [N]  # 예비 후보 새 round 생성
#   python3 export_deo_evaluation_review.py --merge                # manifest에 기록된 배치 병합
#   python3 export_deo_evaluation_review.py --export --force       # 이미 채워진 배치도 덮어쓰기
# =====================================================================

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_splits as bfs  # noqa: E402  (같은 폴더에 build_final_splits.py 필요)

OUTPUT_DIR = bfs.OUTPUT_DIR
REVIEW_DIR = OUTPUT_DIR / "deo_candidate_review"
MANIFEST_PATH = REVIEW_DIR / "review_export_manifest.json"
BATCH_SIZE = 50
RESERVE_SEED = 43  # CANDIDATE_SEED/TRAIN_SEED(42)와 겹치지 않는 별도 seed

FINAL_DEV_COUNT = 50
FINAL_TEST_COUNT = 150  # 또는 200
FINAL_TARGET_COUNT = FINAL_DEV_COUNT + FINAL_TEST_COUNT
RESERVE_BUFFER = 0.25
MIN_DECIDED_FOR_AUTO_RESERVE = 50  # 최소 이만큼 판정돼야 자동 계산을 신뢰할 만함

DECISION_OPTIONS = [
    "A - Usable as-is",
    "B - Usable after limited revision",
    "C - Not usable",
    "D - Uncertain",
]
ISSUE_TYPE_OPTIONS = [
    "Semantic mismatch",
    "Non-Sango / other language",
    "Mixed-language or excessive code-switching",
    "Incorrect name, number, or citation",
    "Grammar or naturalness",
    "Orthography",
    "Incomplete or malformed source/target",
    "Other",
]

HEADERS = [
    "sentence_id", "English source", "Original Sango", "Decision",
    "Issue type", "Corrected Sango", "Optional comment", "pair_id",
]


# =======================================================================
# manifest
# =======================================================================

def _read_manifest():
    if not MANIFEST_PATH.exists():
        return {"batches": []}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _append_manifest(entries):
    manifest = _read_manifest()
    by_path = {e["path"]: e for e in manifest["batches"]}
    for e in entries:
        by_path[e["path"]] = e
    manifest["batches"] = list(by_path.values())
    bfs.save_json_atomic(MANIFEST_PATH, manifest)


def _existing_file_has_answers(path):
    """이미 있는 배치 파일에 Deo가 뭔가 채워 넣었는지 확인 — Decision뿐 아니라
    Issue type/Corrected Sango/comment 중 하나라도 있으면 "답변 있음"으로 본다.
    🔒 파일이 손상돼서 열리지 않으면(예: 과거 크래시로 반쯤 써진 파일) 예외로 죽지
    않고 "답변 없음"으로 안전하게 처리한다 — 어차피 재실행이 새로 덮어쓸 것이므로
    손상된 파일 때문에 재시작 자체가 막히면 안 된다."""
    if not path.exists():
        return False
    import openpyxl
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as e:
        print(f"[warn] {path}를 열 수 없음({e}) — 손상된 파일로 보고 새로 덮어씀")
        return False
    ws = wb.active
    header_row = None
    for r in range(1, ws.max_row + 1):
        if ws.cell(row=r, column=1).value == "sentence_id":
            header_row = r
            break
    if header_row is None:
        return False
    for r in range(header_row + 2, ws.max_row + 1):  # +2: 예시 행 건너뜀
        if any(ws.cell(row=r, column=c).value for c in (4, 5, 6, 7)):
            return True
    return False


def _export_batches(candidates, sentence_id_prefix, file_prefix, source_label, force=False):
    """공통 배치 생성 로직."""
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    n_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_paths = []
    manifest_entries = []

    for b in range(n_batches):
        batch = candidates[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        if not batch:
            continue

        out_path = REVIEW_DIR / f"{file_prefix}{b + 1}_of_{n_batches}.xlsx"
        if not force and _existing_file_has_answers(out_path):
            raise RuntimeError(
                f"{out_path}에 이미 Deo가 입력한 내용이 있음 — 그냥 재실행하면 덮어써서 "
                f"날아간다. 의도한 것이면 force=True(--force)로 다시 실행할 것."
            )

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"Batch {b + 1}"

        # 🔒 옵션 목록을 inline 콤마-구분 문자열(formula1='"A,B,C"')로 넣으면, 옵션
        # 자체에 쉼표가 있는 경우(예: "Incorrect name, number, or citation") Excel이
        # 그 쉼표를 항목 구분자로 오인해서 드롭다운이 세 개로 쪼개질 수 있다. 그래서
        # 숨김 열(J: Decision, K: Issue type)에 옵션을 실제로 써넣고 그 범위를
        # 참조하는 방식으로 만든다 — openpyxl에서 파일이 만들어지는 것과 Excel/Sheets에서
        # 드롭다운이 의도대로 뜨는 것은 별개라 이 방식이 안전하다.
        for i, opt in enumerate(DECISION_OPTIONS, start=1):
            ws.cell(row=i, column=10, value=opt)
        for i, opt in enumerate(ISSUE_TYPE_OPTIONS, start=1):
            ws.cell(row=i, column=11, value=opt)
        ws.column_dimensions["J"].hidden = True
        ws.column_dimensions["K"].hidden = True

        # 안내문 (legend)
        ws.merge_cells("A1:H1")
        ws["A1"] = (
            "For each row, compare the Sango translation with the English source and "
            "select one Decision. Choose B only when the existing translation is "
            "substantially usable and can be corrected through a limited revision. "
            "If the target is unrelated, severely corrupted, not in Sango, or would "
            "require a substantially new translation, choose C. For Decision B, provide "
            "the corrected Sango translation. For Decisions B and C, select the primary "
            "issue type. See the example row below for the expected format (the example "
            "sentence is illustrative only, not a verified translation). Please only edit "
            "the Decision, Issue type, Corrected Sango, and Optional comment columns."
        )
        ws["A1"].font = Font(name="Arial", bold=True, size=11)
        ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[1].height = 85

        header_row = 3
        for col, h in enumerate(HEADERS, start=1):
            c = ws.cell(row=header_row, column=col, value=h)
            c.font = Font(name="Arial", bold=True)
            c.fill = PatternFill("solid", fgColor="D9E1F2")
            c.alignment = Alignment(wrap_text=True, vertical="center")

        # 예시 행 — 검증 안 된 문법 주장을 담지 않도록 형식 설명으로만 구성
        example_row = header_row + 1
        example_values = [
            "example", "(illustrative English source)", "(illustrative Sango translation)",
            "B - Usable after limited revision", "Grammar or naturalness",
            "(illustrative corrected Sango translation)",
            "Example row only — shows where each answer goes, not a verified translation",
            "(example-only, not a real pair_id)",
        ]
        for col, v in enumerate(example_values, start=1):
            c = ws.cell(row=example_row, column=col, value=v)
            c.font = Font(name="Arial", italic=True, color="808080")
            c.alignment = Alignment(wrap_text=True, vertical="top")

        start_row = example_row + 1
        for i, r in enumerate(batch):
            row = start_row + i
            sentence_id = f"{sentence_id_prefix}_{b * BATCH_SIZE + i + 1:04d}"
            ws.cell(row=row, column=1, value=sentence_id).font = Font(name="Arial")
            c_en = ws.cell(row=row, column=2, value=r["en"])
            c_en.alignment = Alignment(wrap_text=True, vertical="top")
            c_en.font = Font(name="Arial")
            c_sg = ws.cell(row=row, column=3, value=r["sg"])
            c_sg.alignment = Alignment(wrap_text=True, vertical="top")
            c_sg.font = Font(name="Arial")
            for col in (4, 5, 6, 7):
                ws.cell(row=row, column=col).font = Font(name="Arial")
            c_pid = ws.cell(row=row, column=8, value=r["pair_id"])
            c_pid.font = Font(name="Arial", size=8, color="A6A6A6")

        end_row = start_row + len(batch) - 1

        dv_decision = DataValidation(type="list",
                                      formula1=f"$J$1:$J${len(DECISION_OPTIONS)}",
                                      allow_blank=True)
        ws.add_data_validation(dv_decision)
        dv_decision.add(f"D{start_row}:D{end_row}")

        dv_issue = DataValidation(type="list",
                                   formula1=f"$K$1:$K${len(ISSUE_TYPE_OPTIONS)}",
                                   allow_blank=True)
        ws.add_data_validation(dv_issue)
        dv_issue.add(f"E{start_row}:E{end_row}")

        widths = {"A": 12, "B": 45, "C": 45, "D": 24, "E": 22, "F": 45, "G": 30, "H": 14}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        ws.freeze_panes = f"A{start_row}"

        # 🔒 실수로 English source/Original Sango/pair_id를 건드리는 것 자체를 줄이기
        # 위해 D~G(Decision/Issue type/Corrected Sango/comment)만 편집 가능하게 하고
        # 나머지는 잠근다. 비밀번호는 안 걸어서 "보안"이 아니라 "실수 방지" 목적임을
        # 분명히 한다 — Google Sheets로 옮기면 이 보호가 그대로 안 지켜질 수 있어서,
        # merge 시 셀 무결성 검증(source_index 대조)이 실제 방어선으로 여전히 필요하다.
        for row in ws.iter_rows():
            for cell in row:
                cell.protection = Protection(locked=True)
        for r in range(start_row, end_row + 1):
            for col in (4, 5, 6, 7):  # D, E, F, G
                ws.cell(row=r, column=col).protection = Protection(locked=False)
        ws.protection.sheet = True
        ws.column_dimensions["H"].hidden = True  # pair_id는 내부 추적용, 안 보여도 됨

        # 🔒 다른 모든 산출물(JSONL/JSON)은 .tmp -> os.replace로 atomic write하는데
        # xlsx만 예외였다 — Colab이 wb.save() 도중 죽으면 반쯤 쓰인 손상된 파일이
        # out_path에 그대로 남고, 다음 재실행 시 그 파일을 열려는
        # _existing_file_has_answers()가 예외로 죽을 수 있었다. 같은 패턴으로 통일.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        wb.save(tmp_path)
        os.replace(tmp_path, out_path)
        batch_paths.append(out_path)
        manifest_entries.append({
            "path": str(out_path), "source": source_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[export] batch {b + 1}/{n_batches}: {len(batch)}개 -> {out_path}")

    _append_manifest(manifest_entries)
    print(f"[export] 총 {len(candidates)}개를 {len(batch_paths)}개 배치로 저장 완료 "
          f"(폴더: {REVIEW_DIR}, manifest: {MANIFEST_PATH})")
    print("[export] 각 xlsx를 Deo에게 전달하고, 다 채운 뒤 같은 파일명으로 같은 폴더에 "
          "저장하도록 안내할 것. 일부 배치만 먼저 돌아와도 --merge로 그때그때 반영 가능.")
    return batch_paths


def export_batches(force=False):
    """evaluation_candidates.jsonl(350개)을 BATCH_SIZE(50)씩 나눠 xlsx 배치로 내보낸다."""
    cand_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    if not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} 없음. --stage candidates까지 먼저 완료할 것.")

    candidates = bfs.load_jsonl(cand_path)
    if len(candidates) != bfs.CANDIDATE_COUNT:
        print(f"[warn] evaluation_candidates.jsonl 개수({len(candidates)})가 "
              f"CANDIDATE_COUNT({bfs.CANDIDATE_COUNT})와 다름 — 그대로 진행은 함")

    return _export_batches(candidates, sentence_id_prefix="eval",
                            file_prefix="deo_review_batch", source_label="eval", force=force)


def _compute_reserve_target(explicit_target=None):
    """🔒 아직 export/merge도 안 된 나머지 배치(예: 배치1만 merge하고 나머지 300개는
    아직 안 건드린 상태)까지 "미판정 대기분"으로 잡아서 같은 통과율로 투영한다.
    summary의 unreviewed_count는 "이미 merge에 올라온 배치 안에서 빈칸인 것"만
    반영하므로, 그것과는 별개로 (전체 candidate pool - 지금까지 merge에 실제로
    포함된 행 수)도 더해야 한다. total_pool은 evaluation_candidates.jsonl뿐 아니라
    기존에 이미 만들어진 reserve_round_*.jsonl까지 전부 합산한다."""
    if explicit_target is not None:
        if explicit_target <= 0:
            raise ValueError("명시적 reserve 개수 N은 1 이상의 정수여야 함.")
        return explicit_target

    summary_path = OUTPUT_DIR / "evaluation_candidates_review_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            "예비 후보 개수를 자동 계산하려면 먼저 --merge를 한 번 이상 실행해서 "
            "통과율을 확인해야 함. 지금 바로 뽑고 싶으면 --export-reserve <N>으로 "
            "개수를 직접 지정할 것."
        )
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    decided = summary["total_decided"]
    if decided < MIN_DECIDED_FOR_AUTO_RESERVE:
        raise RuntimeError(
            f"자동 reserve 계산에는 최소 {MIN_DECIDED_FOR_AUTO_RESERVE}개의 판정이 필요함. "
            f"현재 {decided}개. 그전까지는 --export-reserve <N>으로 개수를 직접 지정할 것."
        )
    passed = summary["decision_counts"].get("A", 0) + summary["decision_counts"].get("B", 0)
    pass_rate = passed / decided
    if pass_rate == 0:
        raise RuntimeError(
            "현재까지 A+B 통과 문장이 0개라 reserve 필요량을 자동 추정할 수 없음. "
            "후보 추출 또는 정제 기준을 재검토하거나 --export-reserve <N>으로 개수를 "
            "직접 지정할 것."
        )

    finalized_so_far = summary["finalized_count"]
    if finalized_so_far >= FINAL_TARGET_COUNT:
        print(f"[reserve] 현재 최종 채택 {finalized_so_far}개 >= 목표 {FINAL_TARGET_COUNT}개 — "
              f"reserve 없이도 목표 달성. 그래도 진행하려면 --export-reserve <N>으로 직접 지정할 것.")
        return 0

    cand_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    total_pool = len(bfs.load_jsonl(cand_path)) if cand_path.exists() else 0
    for reserve_path in sorted(OUTPUT_DIR.glob("reserve_round_*.jsonl")):
        total_pool += len(bfs.load_jsonl(reserve_path))
    not_yet_loaded = max(0, total_pool - summary["total_rows_loaded"])
    remaining_pending = not_yet_loaded + summary["unreviewed_count"]

    expected_from_pending = remaining_pending * pass_rate
    expected_finalized_total = finalized_so_far + expected_from_pending
    expected_shortfall = max(0, FINAL_TARGET_COUNT - expected_finalized_total)

    if expected_shortfall == 0:
        print(f"[reserve] 현재 채택 {finalized_so_far}개 + 남은 {remaining_pending}개(아직 미로드 "
              f"{not_yet_loaded}개 + 로드됐지만 미판정 {summary['unreviewed_count']}개) × 통과율 "
              f"{pass_rate*100:.1f}% 투영 시 목표 {FINAL_TARGET_COUNT}개 달성 예상 — reserve 불필요. "
              f"그래도 만들고 싶으면 --export-reserve <N>으로 직접 지정할 것.")
        return 0

    target = math.ceil(expected_shortfall / pass_rate * (1 + RESERVE_BUFFER))
    print(f"[reserve] 판정 {decided}개 중 통과 {passed}개(통과율 {pass_rate*100:.1f}%) | "
          f"현재 채택 {finalized_so_far}개 + 남은 대기분 {remaining_pending}개(미로드 "
          f"{not_yet_loaded} + 로드됐지만 미판정 {summary['unreviewed_count']}) 투영 시 예상 채택 "
          f"{expected_finalized_total:.1f}개 -> 예상 부족 {expected_shortfall:.1f}개 -> "
          f"reserve 약 {target}개 필요 (버퍼 {RESERVE_BUFFER*100:.0f}% 포함)")
    return target


def _next_reserve_round():
    """🔒 reserve를 여러 번 만들 때마다 round를 분리한다 — 같은 파일을 다른 크기로
    덮어쓰면 이전 round의 batch가 가리키던 reserve_candidates.jsonl이 사라져서 merge가
    깨지는 문제를 방지. 매 round는 이전 모든 round + 기존 candidate/train을 다 제외."""
    existing = sorted(OUTPUT_DIR.glob("reserve_round_*.jsonl"))
    if not existing:
        return 1
    nums = [int(p.stem.split("_")[-1]) for p in existing]
    return max(nums) + 1


def export_reserve_candidates(target_count=None):
    """🔒 350개 candidate + Direct-9K + 이전 모든 reserve round와 pair_id/canonical
    EN/SG 기준으로 전혀 겹치지 않는 새 reserve round를 final_clean_pool에서 뽑는다."""
    target_count = _compute_reserve_target(target_count)
    if target_count == 0:
        return [], None

    round_num = _next_reserve_round()

    pool_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    cand_path = OUTPUT_DIR / "evaluation_candidates.jsonl"
    train_path = OUTPUT_DIR / "direct9k_train.jsonl"
    for p in (pool_path, cand_path, train_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음. Phase 1 데이터 빌드를 먼저 완료할 것.")

    pool = bfs.load_jsonl(pool_path)
    candidates = bfs.load_jsonl(cand_path)
    train = bfs.load_jsonl(train_path)
    prior_rounds = []
    for p in sorted(OUTPUT_DIR.glob("reserve_round_*.jsonl")):
        prior_rounds.extend(bfs.load_jsonl(p))

    excluded_sets = [candidates, train, prior_rounds]
    excluded_ids = set()
    excluded_en = set()
    excluded_sg = set()
    for group in excluded_sets:
        excluded_ids |= {r["pair_id"] for r in group}
        excluded_en |= {bfs.canonical_en(r["en"]) for r in group}
        excluded_sg |= {bfs.canonical_sg(r["sg"]) for r in group}

    eligible = [
        r for r in pool
        if r["pair_id"] not in excluded_ids
        and bfs.canonical_en(r["en"]) not in excluded_en
        and bfs.canonical_sg(r["sg"]) not in excluded_sg
    ]
    print(f"[reserve] round {round_num}: pool {len(pool)}개 중 candidate({len(candidates)})/"
          f"train({len(train)})/이전 round({len(prior_rounds)}) 전역 제외 -> "
          f"eligible {len(eligible)}개")
    if len(eligible) < target_count:
        raise RuntimeError(f"eligible({len(eligible)})이 목표 개수({target_count})보다 적음.")

    strata = {}
    for r in eligible:
        key = (bfs._en_length_bucket(r["en_word_len"]), bfs._laser_bucket(r["laser_score"]))
        strata.setdefault(key, []).append(r)
    total = len(eligible)
    raw_quotas = {k: target_count * len(g) / total for k, g in strata.items()}
    quotas = {k: int(v) for k, v in raw_quotas.items()}
    remaining = target_count - sum(quotas.values())
    priority = sorted(strata.keys(), key=lambda k: (-(raw_quotas[k] - quotas[k]), str(k)))
    for k in priority[:remaining]:
        quotas[k] += 1

    used_en, used_sg = set(), set()
    selected = []
    for key in sorted(strata.keys(), key=str):
        group = strata[key]
        ranked = sorted(group, key=lambda r: bfs.seeded_rank(r["pair_id"], RESERVE_SEED))
        chosen = []
        for r in ranked:
            en_k, sg_k = bfs.canonical_en(r["en"]), bfs.canonical_sg(r["sg"])
            if en_k in used_en or sg_k in used_sg:
                continue
            chosen.append(r)
            used_en.add(en_k)
            used_sg.add(sg_k)
            if len(chosen) == quotas[key]:
                break
        selected.extend(chosen)
        print(f"[reserve] stratum {key}: {len(group)}개 중 {len(chosen)}개 선택 (quota={quotas[key]})")

    if len(selected) < target_count:
        selected_ids = {r["pair_id"] for r in selected}
        remainder = [r for r in eligible if r["pair_id"] not in selected_ids]
        remainder_ranked = sorted(remainder, key=lambda r: bfs.seeded_rank(r["pair_id"], RESERVE_SEED))
        needed = target_count - len(selected)
        backfilled = 0
        for r in remainder_ranked:
            if backfilled >= needed:
                break
            en_k, sg_k = bfs.canonical_en(r["en"]), bfs.canonical_sg(r["sg"])
            if en_k in used_en or sg_k in used_sg:
                continue
            selected.append(r)
            used_en.add(en_k)
            used_sg.add(sg_k)
            backfilled += 1
        print(f"[reserve] 고유성 제약으로 인한 quota 미달분 보충: {backfilled}개")

    selected = sorted(selected, key=lambda r: bfs.seeded_rank(r["pair_id"], RESERVE_SEED))[:target_count]

    # 🔒 최종 방어선: 저장 전에 개수·내부 중복·기존 자료와의 overlap을 한 번 더 검증한다.
    # 위의 quota/backfill 로직에 버그가 있거나 canonical EN/SG 고유성 제약 때문에 목표보다
    # 적게 뽑혔는데 그냥 조용히 저장되는 상황을 막는다.
    if len(selected) != target_count:
        raise RuntimeError(
            f"reserve 목표는 {target_count}개였지만 canonical EN/SG 고유성 제약을 만족하며 "
            f"선택된 문장은 {len(selected)}개뿐임."
        )

    selected_ids = [r["pair_id"] for r in selected]
    selected_en = [bfs.canonical_en(r["en"]) for r in selected]
    selected_sg = [bfs.canonical_sg(r["sg"]) for r in selected]

    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("reserve 내부 pair_id 중복 발견 — 저장하지 않음.")
    if len(selected_en) != len(set(selected_en)):
        raise RuntimeError("reserve 내부 canonical EN 중복 발견 — 저장하지 않음.")
    if len(selected_sg) != len(set(selected_sg)):
        raise RuntimeError("reserve 내부 canonical SG 중복 발견 — 저장하지 않음.")
    if set(selected_ids) & excluded_ids:
        raise RuntimeError("reserve와 기존 candidate/train/이전 round 간 pair_id overlap — 저장하지 않음.")
    if set(selected_en) & excluded_en:
        raise RuntimeError("reserve와 기존 candidate/train/이전 round 간 canonical EN overlap — 저장하지 않음.")
    if set(selected_sg) & excluded_sg:
        raise RuntimeError("reserve와 기존 candidate/train/이전 round 간 canonical SG overlap — 저장하지 않음.")

    out_path = OUTPUT_DIR / f"reserve_round_{round_num:02d}.jsonl"
    bfs.save_jsonl_atomic(out_path, selected)
    ids_path = OUTPUT_DIR / f"reserve_round_{round_num:02d}_ids.json"
    bfs.save_json_atomic(ids_path, {
        "round": round_num, "seed": RESERVE_SEED, "count": len(selected),
        "pair_ids": sorted(r["pair_id"] for r in selected),
        "parent_final_clean_pool_sha256": bfs.sha256_file(pool_path),
        "excluded_evaluation_candidates_sha256": bfs.sha256_file(cand_path),
        "excluded_direct9k_train_sha256": bfs.sha256_file(train_path),
        "excluded_prior_reserve_rounds": round_num - 1,
    })
    print(f"[reserve] round {round_num}: {len(selected)}개 -> {out_path} (개수 일치, 내부 "
          f"중복 없음, 기존 candidate/Direct-9K/이전 round와 pair_id/canonical EN/SG 전역 "
          f"기준으로 overlap 없음 — 저장 전 재검증 통과)")
    return selected, round_num


def export_reserve_batches(target_count=None, force=False):
    """새 reserve round를 뽑아서 바로 xlsx 배치로도 만든다."""
    selected, round_num = export_reserve_candidates(target_count)
    if not selected:
        return []
    return _export_batches(
        selected, sentence_id_prefix=f"reserve_r{round_num:02d}",
        file_prefix=f"deo_review_reserve_r{round_num:02d}_batch",
        source_label=f"reserve_r{round_num:02d}", force=force,
    )


# =======================================================================
# merge
# =======================================================================

def _load_source_index():
    """pair_id -> {en, sg} 원본 색인. evaluation_candidates.jsonl + 모든
    reserve_round_*.jsonl을 합쳐서 만든다. 🔒 같은 pair_id가 서로 다른 내용으로
    두 파일에 존재하면(정상적으론 있을 수 없는 상태 — 데이터 오염 신호) 조용히
    덮어쓰지 않고 즉시 중단한다."""
    index = {}
    paths = [OUTPUT_DIR / "evaluation_candidates.jsonl"]
    paths.extend(sorted(OUTPUT_DIR.glob("reserve_round_*.jsonl")))
    found_any = False
    for path in paths:
        if path.exists():
            found_any = True
            for r in bfs.load_jsonl(path):
                pid = r["pair_id"]
                content = {"en": r["en"], "sg": r["sg"]}
                if pid in index and index[pid] != content:
                    raise RuntimeError(
                        f"동일 pair_id가 서로 다른 내용으로 존재함: {pid} (파일: {path}) — "
                        f"candidate/reserve 파일들이 오염됐을 가능성."
                    )
                index[pid] = content
    if not found_any:
        raise FileNotFoundError("evaluation_candidates.jsonl도, reserve_round_*.jsonl도 없음.")
    return index


def merge_batches():
    """review_export_manifest.json에 기록된 배치 파일만 읽어서 병합하고, Decision별로
    최종 evaluation set을 구성한다."""
    import openpyxl

    manifest = _read_manifest()
    batch_entries = manifest.get("batches", [])
    if not batch_entries:
        raise FileNotFoundError(f"{MANIFEST_PATH}에 기록된 배치가 없음. --export를 먼저 실행할 것.")

    source_index = _load_source_index()

    all_rows = []
    for entry in batch_entries:
        path = Path(entry["path"])
        if not path.exists():
            print(f"[merge] 경고: manifest에 있는 {path}가 실제로는 없음 — 건너뜀")
            continue
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active

        header_row = None
        for r in range(1, ws.max_row + 1):
            if ws.cell(row=r, column=1).value == "sentence_id":
                header_row = r
                break
        if header_row is None:
            print(f"[merge] 경고: {path}에서 헤더 행을 못 찾음 — 건너뜀")
            continue

        for r in range(header_row + 1, ws.max_row + 1):
            sid = ws.cell(row=r, column=1).value
            if sid is None or str(sid) == "example":
                continue
            row = {
                "sentence_id": str(sid),
                "en": ws.cell(row=r, column=2).value or "",
                "sg_original": ws.cell(row=r, column=3).value or "",
                "decision_raw": (ws.cell(row=r, column=4).value or "").strip(),
                "issue_type": (ws.cell(row=r, column=5).value or "").strip(),
                "corrected_sg": (ws.cell(row=r, column=6).value or "").strip(),
                "comment": (ws.cell(row=r, column=7).value or "").strip(),
                "pair_id": ws.cell(row=r, column=8).value or "",
                "source_batch": path.name,
            }
            all_rows.append(row)

    print(f"[merge] manifest 등록 배치 {len(batch_entries)}개에서 총 {len(all_rows)}개 행 읽음")

    # --- hard failure: sentence_id/pair_id 전역 중복 ---
    sentence_ids = [r["sentence_id"] for r in all_rows]
    pair_ids = [r["pair_id"] for r in all_rows]
    dup_sids = {x for x in sentence_ids if sentence_ids.count(x) > 1}
    dup_pids = {x for x in pair_ids if pair_ids.count(x) > 1}
    if dup_sids:
        raise RuntimeError(
            f"중복 sentence_id가 있는 review 파일이 함께 병합됨: {sorted(dup_sids)[:10]} "
            f"(총 {len(dup_sids)}개). manifest({MANIFEST_PATH})나 배치 파일을 확인할 것."
        )
    if dup_pids:
        raise RuntimeError(
            f"중복 pair_id가 있는 review 파일이 함께 병합됨: {sorted(dup_pids)[:10]} "
            f"(총 {len(dup_pids)}개). manifest({MANIFEST_PATH})나 배치 파일을 확인할 것."
        )

    # --- hard failure: 시트의 English source/Original Sango가 원본과 다름 ---
    integrity_errors = []
    for row in all_rows:
        src = source_index.get(row["pair_id"])
        if src is None:
            integrity_errors.append(f"{row['sentence_id']}: 알 수 없는 pair_id "
                                     f"'{row['pair_id']}' (원본 파일에 없음)")
            continue
        if row["en"] != src["en"] or row["sg_original"] != src["sg"]:
            integrity_errors.append(f"{row['sentence_id']}: 원본 English source 또는 "
                                     f"Original Sango 셀이 변경됨")
    if integrity_errors:
        raise RuntimeError(
            "시트 무결성 검사 실패 — English source/Original Sango/pair_id 열은 Deo가 "
            "수정하면 안 되는 열인데 변경된 행이 있음:\n  " + "\n  ".join(integrity_errors[:20])
            + (f"\n  ... 외 {len(integrity_errors) - 20}건" if len(integrity_errors) > 20 else "")
        )

    decision_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "(미기입)": 0, "(잘못된 값)": 0}
    finalized = []
    excluded = []
    uncertain = []
    b_missing_correction = []
    missing_issue_type = []
    invalid_issue_type_rows = []
    invalid_rows = []

    decision_by_full_text = {opt: opt[0] for opt in DECISION_OPTIONS}

    for row in all_rows:
        decision_raw = row["decision_raw"]
        if not decision_raw:
            decision_counts["(미기입)"] += 1
            continue  # 아직 안 채운 건 정상 — pilot 단계에서 당연히 있을 수 있음
        if decision_raw not in decision_by_full_text:
            decision_counts["(잘못된 값)"] += 1
            invalid_rows.append(row)
            continue
        decision = decision_by_full_text[decision_raw]
        decision_counts[decision] += 1

        # 🔒 판정이 "채워진" 행은 형식이 정확해야 한다 — 미판정 행과 달리 여기서부터는
        # 문제가 있으면 hard failure 대상으로 수집한다 (경고만 하고 넘어가지 않음).
        row_ok = True
        if row["issue_type"] and row["issue_type"] not in ISSUE_TYPE_OPTIONS:
            invalid_issue_type_rows.append(row)
            row_ok = False
        if decision in ("B", "C") and not row["issue_type"]:
            missing_issue_type.append(row)
            row_ok = False
        if decision == "B" and not row["corrected_sg"]:
            b_missing_correction.append(row)
            row_ok = False

        if not row_ok:
            continue  # 아래 finalized/excluded/uncertain 반영은 검증 통과한 행만

        src = source_index[row["pair_id"]]
        if decision == "A":
            finalized.append({
                "sentence_id": row["sentence_id"], "source_pair_id": row["pair_id"],
                "final_pair_id": row["pair_id"], "en": row["en"], "sg": src["sg"],
                "sg_original": src["sg"], "revised": False,
            })
        elif decision == "B":
            sg_final_raw = row["corrected_sg"]
            sg_final = bfs.normalize_sango_orthography(sg_final_raw)
            final_pid = bfs.compute_pair_id(row["en"], sg_final)
            finalized.append({
                "sentence_id": row["sentence_id"], "source_pair_id": row["pair_id"],
                "final_pair_id": final_pid, "en": row["en"], "sg": sg_final,
                "sg_original": src["sg"], "sg_corrected_raw": sg_final_raw, "revised": True,
                "issue_type": row["issue_type"], "comment": row["comment"],
            })
        elif decision == "C":
            excluded.append(row)
        elif decision == "D":
            uncertain.append(row)

    total_decided = len(all_rows) - decision_counts["(미기입)"]

    print("[merge] Decision 분포:")
    for k, v in decision_counts.items():
        pct = f"{v/len(all_rows)*100:.1f}%" if all_rows else "0.0%"
        print(f"  {k}: {v}개 ({pct})")

    # --- hard failure: 판정된 행의 불완전/잘못된 입력 ---
    validation_problems = []
    if invalid_rows:
        validation_problems.append(f"잘못된 Decision 문자열 {len(invalid_rows)}건 "
                                    f"(예: {[r['sentence_id'] for r in invalid_rows[:5]]})")
    if b_missing_correction:
        validation_problems.append(f"Decision=B인데 Corrected Sango 없음 {len(b_missing_correction)}건 "
                                    f"(예: {[r['sentence_id'] for r in b_missing_correction[:5]]})")
    if missing_issue_type:
        validation_problems.append(f"Decision=B/C인데 Issue type 없음 {len(missing_issue_type)}건 "
                                    f"(예: {[r['sentence_id'] for r in missing_issue_type[:5]]})")
    if invalid_issue_type_rows:
        validation_problems.append(f"잘못된 Issue type 문자열 {len(invalid_issue_type_rows)}건 "
                                    f"(예: {[r['sentence_id'] for r in invalid_issue_type_rows[:5]]})")
    if validation_problems:
        raise RuntimeError(
            "Review 입력 검증 실패 — 판정(Decision)이 채워진 행은 형식이 정확해야 함 "
            "(미판정 행은 문제 없음):\n  " + "\n  ".join(validation_problems)
            + "\n해당 행들을 Deo에게 다시 확인받은 뒤 재병합할 것."
        )

    # --- hard failure: 수정 후 leakage 재감사 ---
    final_pids = [r["final_pair_id"] for r in finalized]
    final_en_canon = [bfs.canonical_en(r["en"]) for r in finalized]
    final_sg_canon = [bfs.canonical_sg(r["sg"]) for r in finalized]

    dup_final_pids = {x for x in final_pids if final_pids.count(x) > 1}
    dup_final_en = {x for x in final_en_canon if final_en_canon.count(x) > 1}
    dup_final_sg = {x for x in final_sg_canon if final_sg_canon.count(x) > 1}

    train_path = OUTPUT_DIR / "direct9k_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} 없음 — Direct-9K와의 leakage 감사를 수행할 수 없으므로 "
            "최종 evaluation set을 생성하지 않음. (없는 채로 진행하면 빈 집합과 비교돼 "
            "감사가 항상 통과한 것처럼 보이는 false pass가 됨)"
        )
    train = bfs.load_jsonl(train_path)
    train_sg_canon = {bfs.canonical_sg(r["sg"]) for r in train}
    train_en_canon = {bfs.canonical_en(r["en"]) for r in train}
    sg_overlap_with_train = set(final_sg_canon) & train_sg_canon
    en_overlap_with_train = set(final_en_canon) & train_en_canon

    leakage_problems = []
    if dup_final_pids:
        leakage_problems.append(f"final_pair_id 내부 중복 {len(dup_final_pids)}건")
    if dup_final_en:
        leakage_problems.append(f"canonical EN 내부 중복 {len(dup_final_en)}건")
    if dup_final_sg:
        leakage_problems.append(f"canonical SG 내부 중복 {len(dup_final_sg)}건")
    if sg_overlap_with_train:
        leakage_problems.append(f"Direct-9K와 canonical SG overlap {len(sg_overlap_with_train)}건")
    if en_overlap_with_train:
        leakage_problems.append(f"Direct-9K와 canonical EN overlap {len(en_overlap_with_train)}건")

    if leakage_problems:
        raise RuntimeError(
            "수정 후 leakage 재감사 실패 — 최종 세트를 그대로 쓰면 안 됨:\n  "
            + "\n  ".join(leakage_problems)
            + "\n해당 문장들을 다시 확인하거나 배제한 뒤 재병합할 것."
        )
    print("[merge] ✅ 수정 후 leakage 재감사 통과")

    finalized_path = OUTPUT_DIR / "evaluation_candidates_finalized.jsonl"
    bfs.save_jsonl_atomic(finalized_path, finalized)
    print(f"[merge] 최종 채택 문장(A+B) {len(finalized)}개 -> {finalized_path}")

    excluded_path = OUTPUT_DIR / "evaluation_candidates_excluded.jsonl"
    bfs.save_jsonl_atomic(excluded_path, excluded)
    print(f"[merge] 제외(C) {len(excluded)}개 -> {excluded_path}")

    uncertain_path = OUTPUT_DIR / "evaluation_candidates_uncertain.jsonl"
    bfs.save_jsonl_atomic(uncertain_path, uncertain)
    print(f"[merge] 불확실(D) {len(uncertain)}개 -> {uncertain_path}")

    summary_path = OUTPUT_DIR / "evaluation_candidates_review_summary.json"
    bfs.save_json_atomic(summary_path, {
        "total_rows_loaded": len(all_rows),
        "total_decided": total_decided,
        "unreviewed_count": decision_counts["(미기입)"],
        "invalid_decision_count": decision_counts["(잘못된 값)"],
        "decision_counts": decision_counts,
        "finalized_count": len(finalized),
        "excluded_count": len(excluded),
        "uncertain_count": len(uncertain),
        "final_target_count": FINAL_TARGET_COUNT,
    })
    print(f"[merge] 요약 저장: {summary_path}")

    if len(finalized) < FINAL_TARGET_COUNT:
        shortfall = FINAL_TARGET_COUNT - len(finalized)
        print(f"[merge] 참고: 최종 채택 문장이 목표({FINAL_TARGET_COUNT}=dev {FINAL_DEV_COUNT}+"
              f"test {FINAL_TEST_COUNT})보다 {shortfall}개 부족함 — 아직 판정 안 된 배치가 "
              f"남아있다면 그걸 먼저 받고, 그래도 부족하면 --export-reserve로 예비 후보를 "
              f"추가할 것 (인자 없이 실행하면 지금 통과율로 자동 계산됨).")

    return finalized, excluded, uncertain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="350개를 50개씩 xlsx 배치로 내보내기")
    parser.add_argument("--export-reserve", type=int, nargs="?", const=-1, metavar="N",
                         help="새 reserve round를 xlsx 배치로 내보내기. N 생략 시 "
                              "지금까지의 merge 통과율로 자동 계산")
    parser.add_argument("--merge", action="store_true", help="manifest에 기록된 배치들을 병합")
    parser.add_argument("--force", action="store_true",
                         help="이미 Deo가 채운 배치 파일도 덮어쓰기 허용")
    args = parser.parse_args()

    if not args.export and args.export_reserve is None and not args.merge:
        parser.print_help()
        sys.exit(1)
    if args.export:
        export_batches(force=args.force)
    if args.export_reserve is not None:
        n = None if args.export_reserve == -1 else args.export_reserve
        export_reserve_batches(n, force=args.force)
    if args.merge:
        merge_batches()
