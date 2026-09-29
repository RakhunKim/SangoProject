# =====================================================================
# build_adaptation_split.py — evaluation_candidates_finalized.jsonl(Deo 검증
# 통과분)에서 adaptation-dev/test를 확정하고 checksum/ID manifest를 남긴다.
#
# 🔒 이 manifest가 있어야 train_run.py/evaluate_checkpoint.py가 "받은 dev/test
# 파일이 진짜 그 role의 확정된 split인지"를 checksum + sentence_id 완전일치로
# 검증할 수 있다 (단순히 finalized의 부분집합인지만 보면 dev/test를 서로 바꿔
# 넣어도 안 걸리는 문제가 있었음 — 이 스크립트가 그 근본 원인을 없앤다).
#
# 사용법:
#   python3 build_adaptation_split.py \
#     --finalized data_build/evaluation_candidates_finalized.jsonl \
#     --output-dir evaluation --dev-count 50 --test-count 150 --seed 42
#
# 🔒 2026-08-16 수정: 이전 버전은 test 크기를 명시적으로 받지 않고 dev 이후
# "남은 전부"를 test로 넣었음 — accepted 수가 dev+test 요구치보다 적으면
# (예: 153개인데 dev50+test150=200 필요) 에러 없이 조용히 test=103으로
# 통과해버려 원고에 고정한 test=150과 어긋난 상태가 될 위험이 있었음.
# 반대로 accepted가 넘치는 경우에도 test가 150을 초과해 들어갈 수 있었음.
# 이제 test-count를 명시적으로 받고, dev+test에 필요한 수를 밑돌면 하드
# 실패하며, 그 이상 남는 pair는 test에 섞이지 않고 별도 unused pool로 분리함.
# =====================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_splits as bfs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--finalized", required=True,
                    help="evaluation_candidates_finalized.jsonl 경로 (export_deo_evaluation_review.py --merge 산출물)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dev-count", type=int, default=50)
    p.add_argument("--test-count", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    finalized_path = Path(args.finalized)
    if not finalized_path.exists():
        raise FileNotFoundError(
            f"{finalized_path} 없음 — export_deo_evaluation_review.py --merge를 먼저 완료할 것."
        )
    finalized = bfs.load_jsonl(finalized_path)
    if not finalized:
        raise RuntimeError(f"{finalized_path}가 비어있음.")

    for r in finalized:
        if "sentence_id" not in r:
            raise RuntimeError(f"finalized 레코드에 sentence_id가 없음: {r}")
    sids = [r["sentence_id"] for r in finalized]
    if len(sids) != len(set(sids)):
        raise RuntimeError("finalized 내부에 중복 sentence_id가 있음.")

    rank_key = lambda r: bfs.seeded_rank(r.get("final_pair_id") or r["sentence_id"], args.seed)
    ranked = sorted(finalized, key=rank_key)

    required = args.dev_count + args.test_count
    if len(ranked) < required:
        raise ValueError(
            f"Insufficient eligible pairs: {len(ranked)} available, "
            f"{required} required ({args.dev_count} dev + {args.test_count} test). "
            f"Run export_deo_evaluation_review.py --export-reserve for additional "
            f"candidates before retrying."
        )

    dev = ranked[:args.dev_count]
    test = ranked[args.dev_count:required]
    unused = ranked[required:]

    dev_ids = [r["sentence_id"] for r in dev]
    test_ids = [r["sentence_id"] for r in test]
    unused_ids = [r["sentence_id"] for r in unused]
    assert not (set(dev_ids) & set(test_ids)), "내부 로직 오류: dev/test가 겹침 (있으면 안 되는 상태)"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = output_dir / "adaptation_dev.jsonl"
    test_path = output_dir / "adaptation_test.jsonl"
    bfs.save_jsonl_atomic(dev_path, dev)
    bfs.save_jsonl_atomic(test_path, test)

    manifest = {
        "finalized_path": str(finalized_path.resolve()),
        "finalized_parent_sha256": bfs.sha256_file(finalized_path),
        "split_seed": args.seed,
        "dev_count": len(dev),
        "test_count": len(test),
        "unused_count": len(unused),
        "adaptation_dev_sha256": bfs.sha256_file(dev_path),
        "adaptation_test_sha256": bfs.sha256_file(test_path),
        "dev_sentence_ids": dev_ids,
        "test_sentence_ids": test_ids,
        "unused_sentence_ids": unused_ids,
    }
    bfs.save_json_atomic(output_dir / "adaptation_split_manifest.json", manifest)

    print(f"[split] dev={len(dev)}개 -> {dev_path}")
    print(f"[split] test={len(test)}개 -> {test_path}")
    print(f"[split] unused(accepted이지만 미사용)={len(unused)}개")
    print(f"[split] manifest -> {output_dir / 'adaptation_split_manifest.json'}")
    print(f"[split] dev/test sentence_id overlap: {len(set(dev_ids) & set(test_ids))} (0이어야 정상)")


if __name__ == "__main__":
    main()
