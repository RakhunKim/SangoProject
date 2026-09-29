# =====================================================================
# build_experiment_lock.py — NLLB backbone revision 검증 유틸리티
#
# 🔒 2026-08-16 개정: 이전 버전은 이 스크립트가 experiment_lock.json을 만들고
# train_run.py/evaluate_checkpoint.py가 그걸 읽는 구조를 의도했었다(주석에는
# 그렇게 적혀 있었음). 하지만 실제로는 두 스크립트 모두 이 lock 파일을 전혀
# 읽지 않았고 — train_run.py는 매 run마다 HfApi를 직접 재조회했고,
# evaluate_checkpoint.py는 revision 자체를 아예 지정하지 않고 있었다. 즉
# "lock 파일 하나만 읽는다"는 설계 의도가 실제로는 지켜지지 않은 채 세 갈래
# (이 파일 / train_run.py의 재조회 / 아무것도 안 함)로 흩어져 있었다.
#
# 이제는 정반대 방향으로 통일한다 — lock 파일을 만들어서 다른 스크립트가
# 읽게 하는 대신, 2026-08-16 Colab 세션에서 이미 조회·확정한 값을
# train_run.py/evaluate_checkpoint.py 양쪽에 동일한 리터럴 상수
# MODEL_REVISION으로 직접 박아 넣었다. 이 스크립트는 이제 lock을 "생성"하지
# 않고, 그 상수값이 Hub에 여전히 유효한지만 확인하는 용도로 남는다.
#
# 사용법:
#   python3 build_experiment_lock.py
# =====================================================================

from huggingface_hub import HfApi

MODEL_ID = "facebook/nllb-200-distilled-600M"

# train_run.py / evaluate_checkpoint.py에 박아둔 값과 정확히 동일해야 함.
LOCKED_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def main():
    print(f"[check] {MODEL_ID} @ {LOCKED_REVISION} 조회 중...")
    try:
        resolved = HfApi().model_info(MODEL_ID, revision=LOCKED_REVISION).sha
    except Exception as e:
        raise RuntimeError(
            f"{MODEL_ID}@{LOCKED_REVISION} 조회 실패 — Hub에서 이 revision을 "
            f"더 이상 찾을 수 없거나 네트워크 문제일 수 있음: {e}"
        )

    if resolved != LOCKED_REVISION:
        raise RuntimeError(
            f"revision 불일치 — locked={LOCKED_REVISION}, resolved={resolved}. "
            f"train_run.py/evaluate_checkpoint.py의 MODEL_REVISION 상수와 이 "
            f"파일의 LOCKED_REVISION이 서로 다른 값을 가리키고 있을 수 있음."
        )
    print(f"[ok] {MODEL_ID}@{LOCKED_REVISION} 확인됨 — "
          f"train_run.py / evaluate_checkpoint.py의 상수와 일치")


if __name__ == "__main__":
    main()
