# =====================================================================
# flores_prep.py — FLORES-200 English–Sango 준비 (Deo 검증 불필요)
#
# 🔒 이건 350개 evaluation candidate/Direct-9K와 완전히 다른 자료다.
#   - evaluation candidate/Direct-9K: 우리가 자동 채굴한 NLLB 코퍼스에서 뽑은 거라
#     사람(Deo) 검증이 필요함
#   - FLORES-200: Meta/OLD(openlanguagedata)가 전문 번역가를 통해 만들고 품질 관리까지
#     마친 공개 벤치마크 -> 그대로 재사용하는 게 설계 의도. 다시 원어민 검증할 필요 없음
#   OOD(out-of-domain) 성능을 재는 독립적인 잣대로 쓰는 거라, 우리 파이프라인의 정제
#   과정과 아예 분리돼 있어야 의미가 있다 (candidate/Direct-9K와 유일하게 겹쳐도 되는
#   부분이라는 게 애초 연구 설계였음).
#
# 🔒 스키마 불확실성에 대한 안내: openlanguagedata/flores_plus와 facebook/flores의
# 정확한 컬럼명/config 구조는 이 스크립트를 작성한 시점 기준 문서로 확정하지 못했다
# (네트워크 제약으로 실제 조회 불가). 그래서 컬럼을 하드코딩하지 않고 실행 시점에
# 자동 탐지하도록 만들었다 — 탐지 실패 시 사용 가능한 config/컬럼 목록을 그대로
# 출력하니, 그 로그를 보고 필요하면 바로 고칠 수 있다.
#
# 우선순위: openlanguagedata/flores_plus(gated, HF 로그인 필요) 시도 -> 실패하면
# facebook/flores(공개)로 폴백.
#
# 사용법: python3 flores_prep.py [--splits dev,devtest]
# =====================================================================

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_splits as bfs  # noqa: E402  (같은 폴더에 build_final_splits.py 필요.
                                    # 정규화·script validity 검사 함수를 재사용하기 위함)

OUTPUT_DIR = bfs.OUTPUT_DIR.parent / "flores"

PRIMARY_DATASET = "openlanguagedata/flores_plus"
FALLBACK_DATASET = "facebook/flores"
EN_LANG_CANDIDATES = ["eng_Latn"]
SG_LANG_CANDIDATES = ["sag_Latn"]
DEFAULT_SPLITS = ["dev", "devtest"]

# 실제 텍스트가 들어있을 법한 컬럼명 후보 (데이터셋 버전에 따라 다를 수 있어서
# 여러 개를 순서대로 시도한다)
TEXT_COLUMN_CANDIDATES = ["text", "sentence", "sentence_text"]
ID_COLUMN_CANDIDATES = ["id", "sentence_id", "URL", "domain"]


def _pin_revision(dataset_name):
    from huggingface_hub import HfApi
    try:
        info = HfApi().dataset_info(dataset_name)
        return info.sha
    except Exception as e:
        raise RuntimeError(f"{dataset_name} revision을 고정하지 못함: {e}")


def _detect_text_column(ds, lang_config_name):
    """컬럼명이 버전마다 다를 수 있어서, 흔히 쓰이는 이름들과 lang config 이름이
    들어간 컬럼("sentence_eng_Latn" 같은)까지 순서대로 시도한다. 못 찾으면 사용 가능한
    컬럼 목록을 그대로 보여주고 중단 — 추측으로 잘못된 컬럼을 쓰는 것보다 안전하다."""
    cols = ds.column_names
    candidates = list(TEXT_COLUMN_CANDIDATES) + [f"sentence_{lang_config_name}"]
    for c in candidates:
        if c in cols:
            return c
    raise RuntimeError(
        f"텍스트 컬럼을 못 찾음 (시도한 이름: {candidates}). "
        f"실제 컬럼 목록: {cols} — 이 목록 보고 TEXT_COLUMN_CANDIDATES를 고쳐야 함."
    )


def _detect_id_column(ds):
    cols = ds.column_names
    for c in ID_COLUMN_CANDIDATES:
        if c in cols:
            return c
    return None  # id 컬럼이 없으면 위치(인덱스)로 정렬 — FLORES는 관례상 언어 간 줄 순서가 정렬돼 있음


def _load_lang_split(dataset_name, lang_config, split, revision):
    from datasets import load_dataset
    try:
        return load_dataset(dataset_name, lang_config, split=split, revision=revision)
    except Exception as e:
        raise RuntimeError(f"{dataset_name}/{lang_config}/{split} 로드 실패: {e}")


def _try_source(dataset_name, splits):
    print(f"[flores] {dataset_name} 시도 중...")
    revision = _pin_revision(dataset_name)
    print(f"[flores] revision 고정: {revision}")

    en_config = next((c for c in EN_LANG_CANDIDATES), None)
    sg_config = next((c for c in SG_LANG_CANDIDATES), None)

    result = {}
    for split in splits:
        en_ds = _load_lang_split(dataset_name, en_config, split, revision)
        sg_ds = _load_lang_split(dataset_name, sg_config, split, revision)

        if len(en_ds) != len(sg_ds):
            raise RuntimeError(
                f"{split}: en({len(en_ds)}) != sg({len(sg_ds)}) 행 수 불일치 — "
                f"인덱스로 그냥 짝지으면 안 됨. id 컬럼 기준 매칭이 필요할 수 있음."
            )

        en_text_col = _detect_text_column(en_ds, en_config)
        sg_text_col = _detect_text_column(sg_ds, sg_config)
        id_col = _detect_id_column(en_ds)

        print(f"[flores] {split}: en 컬럼='{en_text_col}', sg 컬럼='{sg_text_col}', "
              f"id 컬럼={'있음(' + id_col + ')' if id_col else '없음(위치로 정렬)'}, "
              f"{len(en_ds)}쌍")

        pairs = []
        for i in range(len(en_ds)):
            sentence_id = en_ds[i][id_col] if id_col else i
            pairs.append({
                "sentence_id": f"flores_{split}_{sentence_id}",
                "en": en_ds[i][en_text_col],
                "sg": sg_ds[i][sg_text_col],
            })
        result[split] = pairs

    return dataset_name, revision, en_config, sg_config, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = None
    errors = []
    for dataset_name in (PRIMARY_DATASET, FALLBACK_DATASET):
        try:
            result = _try_source(dataset_name, splits)
            break
        except Exception as e:
            print(f"[flores] {dataset_name} 실패: {e}")
            errors.append(f"{dataset_name}: {e}")

    if result is None:
        raise RuntimeError(
            "openlanguagedata/flores_plus와 facebook/flores 둘 다 실패:\n  " + "\n  ".join(errors)
        )

    dataset_name, revision, en_config, sg_config, splits_data = result

    # 🔒 검증을 전부 끝내고 문제 없을 때만 저장한다 — split별로 검증+저장을 같이 하면
    # 나중 split에서 문제가 걸려도 앞선 split들은 이미 저장된 뒤라 "검증 실패했는데
    # 파일은 이미 생김" 상태가 될 수 있다. 이 프로젝트의 다른 스크립트들과 동일하게
    # "검증 통과한 뒤에만 저장" 원칙을 지킨다.
    problems = []
    for split, pairs in splits_data.items():
        # Deo 검증은 불필요하지만, 최소한 기계적으로 확인 가능한 것만:
        # script validity(정상적인 Sango 문자셋인지), 빈 문장, sentence_id 중복
        sids = [p["sentence_id"] for p in pairs]
        if len(sids) != len(set(sids)):
            problems.append(f"{split}: sentence_id 중복 있음")

        n_empty = sum(1 for p in pairs if not p["en"].strip() or not p["sg"].strip())
        if n_empty:
            problems.append(f"{split}: 빈 문장 {n_empty}개")

        n_script_invalid = sum(
            1 for p in pairs
            if not bfs.check_script_validity(bfs.normalize_sango_orthography(p["sg"]))
        )
        if n_script_invalid:
            print(f"[flores] ⚠️  {split}: script validity 낮은 문장 {n_script_invalid}개 "
                  f"(FLORES는 이미 검증된 벤치마크라 정상 범위일 수 있음 — 참고용 경고일 뿐, "
                  f"자동 제거는 안 함)")

    if problems:
        raise RuntimeError("FLORES 데이터 sanity check 실패 (파일은 저장하지 않음):\n  "
                            + "\n  ".join(problems))

    all_written = {}
    for split, pairs in splits_data.items():
        out_path = OUTPUT_DIR / f"flores_{split}_eng_sag.jsonl"
        bfs.save_jsonl_atomic(out_path, pairs)
        all_written[split] = {"path": str(out_path), "count": len(pairs),
                               "sha256": bfs.sha256_file(out_path)}
        print(f"[flores] {split}: {len(pairs)}쌍 -> {out_path}")

    provenance = {
        "dataset": dataset_name,
        "revision": revision,
        "en_config": en_config,
        "sg_config": sg_config,
        "splits": all_written,
        "note": "FLORES-200은 공개 검증된 벤치마크라 원어민(Deo) 재검증 없이 그대로 사용. "
                "evaluation candidate(350개)/Direct-9K와 유일하게 겹쳐도 되는 자료.",
    }
    prov_path = OUTPUT_DIR / "flores_provenance.json"
    bfs.save_json_atomic(prov_path, provenance)
    print(f"[flores] provenance 저장: {prov_path}")
    print("\n[PASS] FLORES-200 준비 완료")


if __name__ == "__main__":
    main()
