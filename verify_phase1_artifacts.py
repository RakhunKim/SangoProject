# # =====================================================================
# verify_phase1_artifacts.py — Phase 2 시작 전 Phase 1 산출물 무결성 재검증
#
# Phase 1(build_final_splits.py) 산출물을 "존재하는지"만 보는 게 아니라, 실제로
# 논문에 쓸 수 있는 상태 그대로인지(checksum 일치, audit 통과, 개수 일치) 다시
# 확인한다. Phase 2 노트북(sango_project2_experiments.ipynb)이 매번 이 스크립트를
# 호출해서, [PASS]가 안 뜨면 뒤 단계로 못 넘어가게 막는 게 목적이다.
#
# 검사 항목:
#   1. 필수 파일 5개 존재
#   2. prevalidation_manifest.json에 기록된 artifact checksum과 지금 파일의 실제
#      sha256이 일치 (파일이 실수로 수정/교체되지 않았는지)
#   3. audit_report.json의 hard_failures == [], 필수 Boolean checks 전부 True,
#      overlap count(normalized_en_overlap/normalized_sg_overlap) 전부 0
#      (누락된 key도 실패로 처리 — "checks가 비어있으면 통과"가 되지 않게)
#   4. audit_report.json 자체의 artifact_hashes가 지금 이 순간의 실제 파일과 일치
#      (audit freshness 재확인) + manifest에 저장된 audit 스냅샷과 현재
#      audit_report.json이 같은지 확인
#   5. evaluation_candidates.jsonl 개수 == CANDIDATE_COUNT(350),
#      direct9k_train.jsonl 개수 == TRAIN_COUNT(9000)
#
# 사용법: python3 verify_phase1_artifacts.py
# 통과하면 "[PASS] Phase 1 artifacts verified"만 찍고 exit 0.
# 하나라도 실패하면 문제 목록을 찍고 exit 1 (Colab에서 subprocess.run(check=True)로
# 호출하면 이 실패가 노트북 실행도 즉시 멈춘다).
# =====================================================================

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_splits as bfs  # noqa: E402  (같은 폴더에 build_final_splits.py 필요)

OUTPUT_DIR = bfs.OUTPUT_DIR

REQUIRED_FILES = [
    "final_clean_pool.jsonl",
    "evaluation_candidates.jsonl",
    "direct9k_train.jsonl",
    "audit_report.json",
    "prevalidation_manifest.json",
]

# 🔒 audit_report.json의 checks에는 Boolean만 있는 게 아니다 — overlap 관련 두 항목은
# 정수 카운트(0이어야 통과)다. "v is False"로만 훑으면 이 두 항목이 0이 아니어도
# 안 걸리고, checks 자체가 비어있거나 key가 누락돼도 "전부 통과"로 잘못 보고된다.
# 그래서 필요한 key를 명시적으로 나열해서 각각의 정확한 통과 조건으로 검사한다.
REQUIRED_TRUE_CHECKS = [
    "pair_id_disjoint",
    "candidate_count_matches",
    "train_count_matches",
    "train_ids_unique",
    "candidate_ids_unique",
    "candidate_en_unique",
    "candidate_sg_unique",
    "candidate_sidecar_matches",
    "lineage_consistent",
]
REQUIRED_ZERO_CHECKS = [
    "normalized_en_overlap",
    "normalized_sg_overlap",
]


def main():
    problems = []

    # --- 1. 필수 파일 존재 ---
    missing = [f for f in REQUIRED_FILES if not (OUTPUT_DIR / f).exists()]
    if missing:
        problems.append(f"필수 파일 누락: {missing}")
        # 파일이 없으면 아래 검사들은 의미가 없으니 여기서 바로 보고
        _report_and_exit(problems)

    for f in REQUIRED_FILES:
        print(f"[OK] {f} 존재")

    # --- 2. manifest에 기록된 checksum과 현재 파일의 실제 checksum 대조 ---
    with open(OUTPUT_DIR / "prevalidation_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    artifact_checks = {
        "final_clean_pool.jsonl": manifest.get("artifacts", {}).get("final_clean_pool_sha256"),
        "evaluation_candidates.jsonl": manifest.get("artifacts", {}).get("evaluation_candidates_sha256"),
        "direct9k_train.jsonl": manifest.get("artifacts", {}).get("direct9k_train_sha256"),
    }
    for filename, expected_sha in artifact_checks.items():
        if not expected_sha:
            problems.append(f"manifest에 {filename}의 checksum 기록이 없음")
            continue
        actual_sha = bfs.sha256_file(OUTPUT_DIR / filename)
        if actual_sha != expected_sha:
            problems.append(
                f"{filename} checksum 불일치 — manifest: {expected_sha[:16]}..., "
                f"실제: {actual_sha[:16]}... (파일이 수정/교체됐을 가능성)"
            )
        else:
            print(f"[OK] {filename} checksum 일치 ({actual_sha[:16]}...)")

    # --- 3. audit_report.json 재확인 ---
    with open(OUTPUT_DIR / "audit_report.json", encoding="utf-8") as f:
        audit = json.load(f)

    hard_failures = audit.get("hard_failures", [])
    if hard_failures:
        problems.append(f"audit_report.json에 hard_failures 있음: {hard_failures}")
    else:
        print("[OK] audit hard_failures 없음")

    checks = audit.get("checks", {})
    for key in REQUIRED_TRUE_CHECKS:
        if checks.get(key) is not True:
            problems.append(f"audit check 실패 또는 누락: {key}={checks.get(key)!r}")
    for key in REQUIRED_ZERO_CHECKS:
        if checks.get(key) != 0:
            problems.append(f"audit overlap 실패 또는 누락: {key}={checks.get(key)!r}")
    if not any(p.startswith("audit check") or p.startswith("audit overlap") for p in problems):
        print(f"[OK] audit checks 전부 통과 ({len(REQUIRED_TRUE_CHECKS) + len(REQUIRED_ZERO_CHECKS)}개 항목 확인)")

    # --- audit freshness 재확인: audit_report.json 자체에 기록된 artifact_hashes가
    # 지금 이 순간의 실제 파일과 일치하는지 다시 검증한다. 위 2번은 "manifest가 audit
    # 당시와 일치하는지"만 봤지, "audit_report.json이 스스로 기록한 해시가 지금도
    # 맞는지"는 아직 안 봤음 — manifest 생성 이후 누군가 파일을 또 바꿨을 가능성까지
    # 잡으려면 이것도 필요하다. ---
    current_hashes = {
        "final_clean_pool_sha256": bfs.sha256_file(OUTPUT_DIR / "final_clean_pool.jsonl"),
        "evaluation_candidates_sha256": bfs.sha256_file(OUTPUT_DIR / "evaluation_candidates.jsonl"),
        "direct9k_train_sha256": bfs.sha256_file(OUTPUT_DIR / "direct9k_train.jsonl"),
    }
    audit_hashes = audit.get("artifact_hashes", {})
    if not audit_hashes:
        problems.append("audit_report.json에 artifact_hashes가 없음 (구버전 audit로 보임)")
    else:
        for key, actual_sha in current_hashes.items():
            if audit_hashes.get(key) != actual_sha:
                problems.append(f"audit freshness checksum 불일치 또는 누락: {key}")
        if all(audit_hashes.get(k) == v for k, v in current_hashes.items()):
            print("[OK] audit freshness checksum 일치 (audit 당시 파일 == 지금 파일)")

    # manifest에 저장된 audit 스냅샷 자체가 지금의 audit_report.json과 같은지도 확인 —
    # manifest 생성 이후 audit_report.json만 따로 재실행/수정됐다면 둘이 어긋난다.
    if manifest.get("audit") != audit:
        problems.append("현재 audit_report.json이 manifest 생성 당시의 audit 내용과 다름")
    else:
        print("[OK] manifest에 기록된 audit 스냅샷과 현재 audit_report.json 일치")

    # --- 4. 개수 확인 ---
    n_candidates = sum(1 for _ in bfs.iter_jsonl(OUTPUT_DIR / "evaluation_candidates.jsonl"))
    if n_candidates != bfs.CANDIDATE_COUNT:
        problems.append(f"evaluation_candidates.jsonl 개수 {n_candidates} != "
                         f"CANDIDATE_COUNT({bfs.CANDIDATE_COUNT})")
    else:
        print(f"[OK] evaluation_candidates.jsonl 개수 {n_candidates}개 일치")

    n_train = sum(1 for _ in bfs.iter_jsonl(OUTPUT_DIR / "direct9k_train.jsonl"))
    if n_train != bfs.TRAIN_COUNT:
        problems.append(f"direct9k_train.jsonl 개수 {n_train} != TRAIN_COUNT({bfs.TRAIN_COUNT})")
    else:
        print(f"[OK] direct9k_train.jsonl 개수 {n_train}개 일치")

    _report_and_exit(problems)


def _report_and_exit(problems):
    if problems:
        print("\n[FAIL] Phase 1 산출물 검증 실패:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\n[PASS] Phase 1 artifacts verified")
    sys.exit(0)


if __name__ == "__main__":
    main()
