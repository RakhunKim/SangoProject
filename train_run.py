# =====================================================================
# train_run.py — 본실험 학습 스크립트 (LoRA / DoRA / OFT 공통, 재개 가능) — CLI v1
#
# 스모크테스트로 검증된 train_run_dora.py(단일 파일, 모듈 상수로 METHOD/SEED/
# PROJECT_ROOT를 손으로 바꿔가며 재실행하던 버전)를 CLI 인자 기반으로 재구성.
# FinalAdapterCallback/PermanentAdapterCallback/checkpoint_metadata.json/
# 누적 wall-clock 기록 등 검증된 로직은 그대로 유지.
#
# 🔒 이번에 새로 추가된 것:
#   1. --method/--seed/--train/--dev/--output/--resume CLI 인자
#   2. Direct-9K 입력 checksum 검증 (--train-sha256 명시 또는 prevalidation_manifest.json
#      대조) — 명시값도 manifest도 없으면 하드 실패, 조용히 신뢰하지 않음
#   3. adaptation-dev 검증: dev 파일이 없거나, evaluation_candidates_finalized.jsonl
#      (Deo 검증 병합 결과)이 없거나, dev의 문장이 그 안에 없으면 하드 실패 —
#      "임시 dev로 시작했다가 나중에 dev가 바뀌면 9-run을 전부 다시 돌려야 하는"
#      상황을 원천 차단
#   4. run별 독립 출력 폴더(--output) + 완료 여부(COMPLETED.json) 자동 skip
#   5. checkpoint 있는데 --resume 없이 실행하면 하드 실패(덮어쓰기 방지) —
#      checkpoint 없으면 --resume 여부와 무관하게 처음부터 시작
#   6. config.json/environment.json/data_provenance.json/training_history.json 저장
#   7. 전부 통과한 뒤에만 COMPLETED.json 생성
#
# 사용 예:
#   python3 train_run.py --method lora --seed 42 \
#     --train data_build/direct9k_train.jsonl \
#     --dev evaluation/adaptation_dev.jsonl \
#     --output experiments/lora_seed42 \
#     --resume
#
# smoke test 예 (실제 학습 대신 아주 작은 step으로 파이프라인만 검증):
#   python3 train_run.py --method dora --seed 42 \
#     --train smoke/train30.jsonl --dev smoke/dev10.jsonl \
#     --output smoke_runs/dora_seed42 \
#     --max-steps 12 --eval-save-interval 5 --warmup-steps 2 \
#     --history-interval 5 --logging-steps 1
# =====================================================================

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_splits as bfs  # noqa: E402  (atomic write/checksum 헬퍼 재사용)

METHOD_CHOICES = ["lora", "dora", "oft"]
SEED_CHOICES = [42, 43, 44]

MODEL_ID = "facebook/nllb-200-distilled-600M"
# 🔒 2026-08-16 Colab 세션에서 HfApi로 조회 및 assert 완료. 매 run마다 재조회하지
# 않고 이 값을 리터럴로 고정한다 — HF 쪽 모델 카드가 그 사이 바뀌어도 9-run 전체가
# 동일 snapshot을 쓰도록 보장하기 위함 (build_experiment_lock.py의 설계 의도와
# 동일하되, lock 파일을 별도로 읽는 대신 상수로 직접 고정).
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SRC_LANG = "eng_Latn"
TGT_LANG = "sag_Latn"
MAX_LEN = 128
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# 두 스크립트에서 반드시 동일해야 하는 decoding 설정.
# evaluate_checkpoint.py의 NUM_BEAMS와 값이 일치하는지 항상 확인할 것.
NUM_BEAMS = 1  # KCI와 동일 조건(그리디) 유지 + 9-run 반복 비용 절감

# 본실험 기본 하이퍼파라미터 (스모크테스트 때는 CLI로 작은 값 override)
DEFAULT_MAX_STEPS = 3375
DEFAULT_EVAL_SAVE_INTERVAL = 250
DEFAULT_HISTORY_INTERVAL = 500
DEFAULT_LOGGING_STEPS = 50

# 🔒 warmup은 고정 step 수가 아니라 "linear scheduler + warmup ratio 0.1"로 잠겨
# 있음(원고 §3.3.2, config lock v2 §7). --warmup-steps를 명시하지 않으면
# ceil(max_steps * WARMUP_RATIO)로 매번 계산한다 — max_steps가 3375가 아닌
# smoke test에서도 같은 비율이 자동으로 유지되도록.
WARMUP_RATIO = 0.1


# =======================================================================
# 관리 로직 (torch/transformers 등 무거운 라이브러리 없이도 동작 — 그래서
# 이 부분만 따로 단위 테스트 가능)
# =======================================================================

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_direct9k_checksum(train_path, manifest_path=None, expected_sha256=None):
    """🔒 Direct-9K 입력이 Phase 1에서 frozen된 그 파일과 정확히 같은 바이트인지 확인.
    --train-sha256이 명시되면 그걸 최우선으로 쓰고, 없으면 manifest_path(기본: train
    파일과 같은 폴더의 prevalidation_manifest.json)에서 direct9k_train_sha256을 찾아
    대조한다. 둘 다 없으면 검증 불가로 하드 실패 — "일단 믿고 진행"하지 않는다."""
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} 없음.")
    actual = bfs.sha256_file(train_path)

    if expected_sha256:
        if actual != expected_sha256:
            raise RuntimeError(
                f"Direct-9K checksum 불일치 — 지정된 값: {expected_sha256[:16]}..., "
                f"실제: {actual[:16]}..."
            )
        print(f"[verify] Direct-9K checksum 일치 (명시된 값과 대조, {actual[:16]}...)")
        return actual

    manifest_path = Path(manifest_path) if manifest_path else train_path.parent / "prevalidation_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Direct-9K checksum을 검증할 방법이 없음 — --train-sha256도 안 주고 "
            f"manifest({manifest_path})도 없음. 학습을 시작하지 않음."
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    expected = manifest.get("artifacts", {}).get("direct9k_train_sha256")
    if not expected:
        raise RuntimeError(f"{manifest_path}에 direct9k_train_sha256 기록이 없음.")
    if actual != expected:
        raise RuntimeError(
            f"Direct-9K checksum 불일치 — manifest: {expected[:16]}..., "
            f"실제: {actual[:16]}... (파일이 수정/교체됐을 가능성)"
        )
    print(f"[verify] Direct-9K checksum 일치 (manifest 대조, {actual[:16]}...)")
    return actual


def verify_adaptation_split(split_path, split_role, manifest_path):
    """🔒 adaptation-dev/test가 정확히 그 role로 확정된 split인지 확인한다.
    이전 버전은 "finalized의 부분집합인지"만 봐서, dev 자리에 test 파일을 잘못
    넣어도(또는 그 반대) 걸리지 않았다 — 둘 다 finalized의 부분집합이기 때문. 이제:
      (1) 파일 checksum이 manifest의 adaptation_<role>_sha256과 정확히 일치
          (dev/test를 바꿔 넣거나 내용이 바뀐 파일을 잡아냄 — ID 대조보다 훨씬 강함)
      (2) sentence_id 집합이 manifest의 <role>_sentence_ids와 완전히 일치
          (부분집합이 아니라 정확히 일치 — 누락/초과 문장이 있으면 걸림)
      (3) manifest 자체의 dev/test sentence_id가 서로 안 겹치는지 재확인
    manifest는 build_adaptation_split.py가 만든다."""
    split_path = Path(split_path)
    manifest_path = Path(manifest_path)
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} 없음 — adaptation-{split_role}이 아직 준비되지 않음.")
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} 없음 — adaptation dev/test가 아직 확정되지 않음. "
            f"build_adaptation_split.py를 먼저 실행할 것. "
            f"(임시 split으로 본학습을 시작했다가 나중에 바뀌면 9-run을 전부 다시 돌려야 함)"
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    actual_sha = bfs.sha256_file(split_path)
    expected_sha = manifest.get(f"adaptation_{split_role}_sha256")
    if not expected_sha:
        raise RuntimeError(f"{manifest_path}에 adaptation_{split_role}_sha256 기록이 없음.")
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{split_path}의 checksum이 manifest의 adaptation_{split_role}_sha256과 다름 — "
            f"{split_role}가 아닌 다른 role의 파일(dev/test 오표기)이거나 내용이 바뀐 파일일 수 있음."
        )

    data = load_jsonl(split_path)
    if not data:
        raise RuntimeError(f"{split_path}가 비어있음.")
    for i, r in enumerate(data):
        if "en" not in r or "sg" not in r:
            raise RuntimeError(f"{split_path}의 {i}번째 레코드에 en/sg 필드가 없음: {r}")

    actual_ids = {r["sentence_id"] for r in data}
    expected_ids = set(manifest.get(f"{split_role}_sentence_ids", []))
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise RuntimeError(
            f"{split_path}의 sentence_id 집합이 manifest의 {split_role}_sentence_ids와 "
            f"정확히 일치하지 않음 (누락 {len(missing)}개, 초과 {len(extra)}개)."
        )

    dev_ids = set(manifest.get("dev_sentence_ids", []))
    test_ids = set(manifest.get("test_sentence_ids", []))
    if dev_ids & test_ids:
        raise RuntimeError("manifest 자체에서 dev/test sentence_id가 겹침 — split 생성이 잘못됨.")

    print(f"[verify] {split_path}가 adaptation_split_manifest.json의 '{split_role}' 정의와 "
          f"정확히 일치함 (checksum + sentence_id 집합 완전 일치, dev/test overlap 0)")
    return data


def check_already_completed(output_dir):
    return (Path(output_dir) / "COMPLETED.json").exists()


def determine_resume(output_dir, resume_flag):
    """checkpoint가 없으면 --resume 여부와 무관하게 그냥 처음부터 시작(안전).
    checkpoint가 있는데 --resume 없이 실행되면, 기존 진행상황을 덮어쓸 위험이 있으니
    하드 실패 — 이어서 하려면 --resume, 처음부터 하려면 output_dir를 비우고 실행."""
    trainer_dir = Path(output_dir) / "trainer_checkpoints"
    existing = sorted(trainer_dir.glob("checkpoint-*")) if trainer_dir.exists() else []
    if not existing:
        return False
    if resume_flag:
        print(f"[resume] {trainer_dir}에 checkpoint {len(existing)}개 발견 -> 이어서 학습")
        return True
    raise RuntimeError(
        f"{trainer_dir}에 이미 checkpoint가 {len(existing)}개 있는데 --resume 없이 "
        f"실행됨 — 그대로 진행하면 기존 진행상황을 덮어쓸 위험이 있음. 이어서 하려면 "
        f"--resume을 주고, 처음부터 다시 하려면 {output_dir}를 비우고 실행할 것."
    )


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHOD_CHOICES)
    p.add_argument("--seed", required=True, type=int, choices=SEED_CHOICES)
    p.add_argument("--train", required=True, help="Direct-9K train jsonl 경로")
    p.add_argument("--dev", required=True, help="adaptation-dev jsonl 경로")
    p.add_argument("--output", required=True, help="이 run 전용 출력 폴더")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--train-manifest", default=None,
                    help="prevalidation_manifest.json 경로 (기본: --train과 같은 폴더)")
    p.add_argument("--train-sha256", default=None,
                    help="Direct-9K 기대 checksum 직접 지정 (manifest 대신 사용)")
    p.add_argument("--split-manifest", default=None,
                    help="adaptation_split_manifest.json 경로 (build_adaptation_split.py 산출물, "
                         "기본: --dev와 같은 폴더)")
    p.add_argument("--registry", default=None,
                    help="run 결과를 누적 기록할 csv (기본: --output의 부모 폴더)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--eval-save-interval", type=int, default=DEFAULT_EVAL_SAVE_INTERVAL)
    p.add_argument("--warmup-steps", type=int, default=None,
                    help=f"명시하지 않으면 ceil(max_steps * {WARMUP_RATIO})로 자동 계산 "
                         f"(본실험 3375 step 기준 338). smoke test처럼 --max-steps를 "
                         f"줄일 때도 이 비율이 그대로 따라감.")
    p.add_argument("--history-interval", type=int, default=DEFAULT_HISTORY_INTERVAL)
    p.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    return p


# =======================================================================
# 학습 실행 (여기서부터 torch/transformers/peft/datasets/sacrebleu 필요 —
# 위 관리 로직과 분리해서, 저 함수들만은 무거운 라이브러리 없이 단위 테스트 가능)
# =======================================================================

def _run_training(args, output_dir, resume, dev_data, config):
    # 🔒 2026-08-16 이후: 환경은 이 스크립트가 아니라 Colab 노트북 Step 0
    # (phase2_env.py env-manifest --strict)에서 requirements_phase2_frozen.txt
    # 기준으로 고정한다. 예전에 여기서 하던 `pip install -U ... torchao`는
    # 제거함 — 9개 run 사이에 조용히 버전이 바뀌는 걸 막기 위한 오늘의 환경
    # freeze 작업 전체를 무력화시키는 코드였고, 우리가 의도적으로 제거한
    # torchao를 매 run마다 다시 깔아 넣는 부작용도 있었음. 이 스크립트는
    # 이제 설치된 버전을 검증만 하고, 절대 바꾸지 않는다.
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, TrainerCallback,
        set_seed,
    )
    from peft import LoraConfig, OFTConfig, get_peft_model, TaskType
    import peft as peft_pkg
    import transformers as transformers_pkg
    from datasets import Dataset

    try:
        import sacrebleu
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "sacrebleu"], check=True)
        import sacrebleu

    method, seed = args.method, args.seed
    run_name = f"{method}_seed{seed}"
    trainer_dir = output_dir / "trainer_checkpoints"
    history_dir = output_dir / "history_adapters"
    best_dir = output_dir / "best_adapter"
    final_dir = output_dir / "final_adapter"
    for d in (trainer_dir, history_dir, best_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.registry) if args.registry else output_dir.parent / "experiment_registry.csv"
    timing_path = output_dir / "timing.json"

    print(f"[run] {run_name}  (max_steps={args.max_steps}, num_beams={NUM_BEAMS})")

    set_seed(seed)
    random.seed(seed)

    train_data = load_jsonl(args.train)
    print(f"[data] train={len(train_data)}  dev={len(dev_data)}")

    # 🔒 매 run마다 HfApi를 재조회하지 않는다 — 모듈 상수 MODEL_REVISION을 그대로
    # 쓴다. run마다 따로 조회하면 그 사이 HF 모델 카드가 갱신될 경우 "방법 차이"가
    # 아니라 "base model snapshot 차이"가 9-run 비교에 섞여 들어갈 수 있음
    # (build_experiment_lock.py가 원래 막으려던 문제와 동일 — 여기서는 lock 파일을
    # 별도로 읽는 대신 리터럴 상수로 고정해 그 목적을 달성함).
    model_revision = MODEL_REVISION
    print(f"[model] {MODEL_ID} revision 고정(상수): {model_revision}")
    config["model_revision"] = model_revision
    bfs.save_json_atomic(output_dir / "config.json", config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
    tokenizer.src_lang = SRC_LANG

    def preprocess(batch):
        model_inputs = tokenizer(batch["en"], truncation=True, max_length=MAX_LEN)
        tokenizer.tgt_lang = TGT_LANG
        labels = tokenizer(text_target=batch["sg"], truncation=True, max_length=MAX_LEN)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_list(train_data).map(
        preprocess, batched=True, remove_columns=list(train_data[0].keys()))
    dev_ds = Dataset.from_list(dev_data).map(
        preprocess, batched=True, remove_columns=list(dev_data[0].keys()))

    # 🔒 2026-08-16 precision refreeze: OFT가 FP16 autocast에서 NaN을 내는 것을 확인해 세 방법 모두 FP32로 통일
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=model_revision)
    assert next(model.parameters()).dtype == torch.float32, "backbone이 float32로 로드되지 않음"

    if method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
            lora_dropout=0.05, use_dora=False, target_modules=TARGET_MODULES,
        )
    elif method == "dora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
            lora_dropout=0.05, use_dora=True, target_modules=TARGET_MODULES,
        )
    else:  # oft
        peft_config = OFTConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, oft_block_size=32,
            use_cayley_neumann=True, target_modules=TARGET_MODULES,
            module_dropout=0.0, bias="none",
        )

    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float32, f"{name}이 float32가 아님: {p.dtype}"

    # NLLB는 forced_bos_token_id를 타깃 언어로 명시하지 않으면 생성 언어가 어긋날 수
    # 있음. evaluate_checkpoint.py에서도 동일하게 지정해야 함.
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model.generation_config.num_beams = NUM_BEAMS

    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    # 🔒 01_manifests/metric_lock.json(2026-08-16)과 정확히 동일한 값을 명시.
    # 여기서는 checkpoint 선택용 진단 지표(diagnostic)로만 쓰이지만, 최종 결과에
    # 쓰는 evaluate_checkpoint.py와 설정이 어긋나면 안 되므로 값을 그대로 맞춤.
    CHRF_METRIC = sacrebleu.CHRF(char_order=6, word_order=0, beta=2,
                                  lowercase=False, whitespace=False)
    BLEU_METRIC = sacrebleu.BLEU(tokenize="13a", smooth_method="exp",
                                  lowercase=False, effective_order=False)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = [[t for t in seq if t != -100] for seq in preds]
        labels = [[t for t in seq if t != -100] for seq in labels]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        chrf = CHRF_METRIC.corpus_score(decoded_preds, [decoded_labels]).score
        bleu = BLEU_METRIC.corpus_score(decoded_preds, [decoded_labels]).score
        return {"chrf": chrf, "bleu": bleu}

    class PermanentAdapterCallback(TrainerCallback):
        """Trainer의 save_total_limit rotation과 무관하게 history_interval마다
        adapter를 history_dir에 별도 보존 (trajectory 분석용)."""
        def __init__(self, history_dir, snapshot_interval):
            self.history_dir = Path(history_dir)
            self.snapshot_interval = snapshot_interval

        def on_save(self, args, state, control, model=None, **kwargs):
            if not state.is_world_process_zero:
                return control
            step = int(state.global_step)
            if step % self.snapshot_interval != 0:
                return control
            target = self.history_dir / f"step_{step:05d}"
            if not target.exists():
                model.save_pretrained(str(target))
                with open(target / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
                    json.dump({"checkpoint_role": "history", "global_step": step}, f, indent=2)
                print(f"[history] permanent snapshot saved: step {step} -> {target}")
            return control

    class FinalAdapterCallback(TrainerCallback):
        """Trainer의 자동 checkpoint 저장(save_steps 배수)과 무관하게, 마지막 optimizer
        step에서 확실히 한 번 저장한다. max_steps가 save_steps의 배수가 아니면 "가장
        최근 checkpoint-*"가 실제 마지막 step이 아닐 수 있으므로 이 경로에 의존 안 함."""
        def __init__(self, final_dir):
            self.final_dir = Path(final_dir)
            self.saved = False

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if (state.is_world_process_zero and not self.saved
                    and state.global_step >= state.max_steps):
                if self.final_dir.exists():
                    shutil.rmtree(self.final_dir)
                model.save_pretrained(str(self.final_dir))
                with open(self.final_dir / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
                    json.dump({"checkpoint_role": "final", "global_step": int(state.global_step)},
                               f, indent=2)
                self.saved = True
                print(f"[save] true final adapter -> {self.final_dir} (step {state.global_step})")
            return control

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(trainer_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=False,  # 🔒 precision refreeze
        bf16=False,
        optim="adafactor",  # 🔒 transformers 5.15.0 _get_adafactor()가 scale_parameter=False,
                             # relative_step=False를 자동 세팅함(2026-08-16 소스 확인) — 외부
                             # learning_rate/scheduler가 실제로 LR 궤적을 결정하게 하려면 필요
        learning_rate=1e-4,
        lr_scheduler_type="linear",  # 🔒 원고 §3.3.2/lock v2 §7 요구사항. 이전엔 default(linear)에
                                      # 의존했음 — mutable default에 맡기지 않고 명시
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        label_smoothing_factor=0.0,  # 🔒 원고 §3.3.2/lock v2 §6·§8 요구사항. 모델 자체 loss는
                                      # label_smoothing_factor를 안 읽지만(CrossEntropyLoss()
                                      # 기본값이 이미 0.0), 이걸 명시 안 하면 Trainer의
                                      # LabelSmoother가 그 내부 loss를 덮어써 버릴 수 있음
        save_strategy="steps",
        save_steps=args.eval_save_interval,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_save_interval,
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        generation_num_beams=NUM_BEAMS,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    # 🔒 label_pad_token_id=-100을 명시. 이전엔 default(-100)에 의존했는데, 원고
    # §3.3.2가 "명시적으로 설정했다"고 서술하는 부분이라 실제 코드도 그래야 함.
    assert training_args.fp16 is False and training_args.bf16 is False, "fp16/bf16이 켜져 있음"
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            PermanentAdapterCallback(history_dir, snapshot_interval=args.history_interval),
            FinalAdapterCallback(final_dir),
        ],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    print(f"[done] {run_name} 완료. this-session wall-clock: {elapsed/3600:.2f}h")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    previous_seconds = 0.0
    if timing_path.exists():
        with open(timing_path, encoding="utf-8") as f:
            previous_seconds = json.load(f).get("cumulative_seconds", 0.0)
    cumulative_seconds = previous_seconds + elapsed
    bfs.save_json_atomic(timing_path, {
        "cumulative_seconds": cumulative_seconds, "last_session_seconds": elapsed,
    })
    print(f"[timing] cumulative wall-clock so far: {cumulative_seconds/3600:.2f}h")

    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    best_ckpt = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    best_step = int(best_ckpt.rsplit("-", 1)[-1]) if best_ckpt else None

    bfs.save_json_atomic(best_dir / "checkpoint_metadata.json", {
        "checkpoint_role": "best", "global_step": best_step,
        "adaptation_dev_chrf": best_metric, "source_checkpoint": best_ckpt,
    })
    print(f"[save] best adapter -> {best_dir} (step {best_step}, chrF={best_metric})")

    # --- 사후 검증 — 조용한 오저장을 막기 위한 assertion ---
    final_meta_path = final_dir / "checkpoint_metadata.json"
    if not final_meta_path.exists():
        raise RuntimeError("final adapter metadata가 생성되지 않았습니다. FinalAdapterCallback을 확인할 것.")
    with open(final_meta_path, encoding="utf-8") as f:
        final_meta = json.load(f)
    if int(final_meta["global_step"]) != args.max_steps:
        raise RuntimeError(
            f"final step 불일치: expected={args.max_steps}, saved={final_meta['global_step']}."
        )
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("best checkpoint가 선택되지 않았습니다 (load_best_model_at_end 설정 확인).")
    print(f"[verify] final step={final_meta['global_step']} == max_steps={args.max_steps} OK")
    print(f"[verify] best_model_checkpoint={trainer.state.best_model_checkpoint} OK")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    peak_mem = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None

    environment = {
        "python": sys.version,
        "transformers_version": transformers_pkg.__version__,
        "peft_version": peft_pkg.__version__,
        "torch_version": torch.__version__,
        "sacrebleu_version": sacrebleu.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_model": gpu_name,
        "training_precision": "fp32", "fp16": False, "bf16": False,
    }
    bfs.save_json_atomic(output_dir / "environment.json", environment)

    summary = {
        "run_name": run_name, "method": method, "seed": seed,
        "max_steps": args.max_steps, "num_beams": NUM_BEAMS,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "session_wallclock_h": round(elapsed / 3600, 2),
        "cumulative_wallclock_h": round(cumulative_seconds / 3600, 2),
        "trainable_params": trainable_params, "all_params": all_params,
        "trainable_pct": round(100 * trainable_params / all_params, 4),
        "best_adaptation_dev_chrf": best_metric, "best_step": best_step,
        "peak_gpu_memory_gb": peak_mem, "gpu_model": gpu_name,
        **{k: v for k, v in environment.items() if k not in ("python", "gpu_model", "cuda_available")},
    }

    header_needed = not registry_path.exists()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(",".join(summary.keys()) + "\n")
        f.write(",".join(str(v) for v in summary.values()) + "\n")
    print(f"[registry] {registry_path} 기록 완료")

    # --- 전부 통과한 뒤에만 COMPLETED.json 생성 -> 다음 재실행 시 자동 skip ---
    bfs.save_json_atomic(output_dir / "COMPLETED.json", summary)
    print(f"[complete] {output_dir / 'COMPLETED.json'} 생성 — 이 run은 완료 처리됨")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.warmup_steps is None:
        args.warmup_steps = math.ceil(args.max_steps * WARMUP_RATIO)
        print(f"[warmup] --warmup-steps 미지정 -> ceil({args.max_steps} * {WARMUP_RATIO}) "
              f"= {args.warmup_steps}로 자동 계산")

    output_dir = Path(args.output)

    if check_already_completed(output_dir):
        print(f"[skip] {output_dir / 'COMPLETED.json'} 이미 존재 — 이 run은 완료됨. 건너뜀.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    resume = determine_resume(output_dir, args.resume)

    train_sha = verify_direct9k_checksum(args.train, args.train_manifest, args.train_sha256)
    split_manifest = args.split_manifest or str(Path(args.dev).parent / "adaptation_split_manifest.json")
    dev_data = verify_adaptation_split(args.dev, "dev", split_manifest)
    dev_sha = bfs.sha256_file(args.dev)

    config = {
        "method": args.method, "seed": args.seed,
        "train_path": str(Path(args.train).resolve()), "dev_path": str(Path(args.dev).resolve()),
        "train_sha256": train_sha, "dev_sha256": dev_sha,
        "max_steps": args.max_steps, "eval_save_interval": args.eval_save_interval,
        "warmup_steps": args.warmup_steps, "warmup_ratio_locked": WARMUP_RATIO,
        "history_interval": args.history_interval,
        "logging_steps": args.logging_steps, "num_beams": NUM_BEAMS,
        "model_id": MODEL_ID, "target_modules": TARGET_MODULES,
        "started_at": datetime.now(timezone.utc).isoformat(), "resume": resume,
    }
    bfs.save_json_atomic(output_dir / "config.json", config)
    print(f"[config] {output_dir / 'config.json'} 저장")

    _run_training(args, output_dir, resume, dev_data, config)


if __name__ == "__main__":
    main()
