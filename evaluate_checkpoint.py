# =====================================================================
# 체크포인트 평가/생성 스크립트 — v2
#
# v1 대비 수정 사항 (코드 리뷰 반영):
#   1. --method에 zero_shot 추가 (zero-shot NLLB 원자료 생성 지원)
#   2. NUM_BEAMS를 train_run.py와 반드시 동일한 값(기본 1)으로 고정
#   3. sentence_id는 파일의 명시적 "sentence_id" 필드를 사용 (row order 금지)
#      + 평가 파일 내 중복 sentence_id 검사
#   4. 재실행 시 동일 (method, seed, step, dataset) 조합 중복 append 방지
#   5. adapter_path 폴더의 checkpoint_metadata.json이 있으면 거기서 실제
#      global_step을 읽어와 --step 인자보다 우선 사용 (best/final 라벨만
#      믿지 않음)
#   6. sacrebleu signature를 corpus summary에 함께 저장 (재현성)
#
# 문장별 CSV는 fixed-seed paired sentence bootstrap의 원자료가 된다.
# 반드시 있어야 하는 컬럼:
#   method, seed, checkpoint_role, step, dataset, sentence_id,
#   src, ref, hyp, sentence_chrf, checkpoint_path
#
# 주의: sentence_chrf는 오류 사례 탐색용이다. paired bootstrap에서는
# 재표집된 문장 묶음으로 corpus chrF를 다시 계산해야 하며, 이 컬럼의
# 평균을 그대로 bootstrap 결과로 쓰면 안 된다 (hyp/ref 원문이 있으므로
# 재계산에 필요한 자료는 이미 이 CSV로 충분함).
#
# 사용 예 (Colab 셀에서):
#   !python evaluate_checkpoint.py \
#       --adapter_path /content/drive/MyDrive/sango_project2/runs/lora_seed42/best_adapter \
#       --method lora --seed 42 \
#       --dataset_name adaptation_test \
#       --eval_path /content/drive/MyDrive/sango_project2/splits/adaptation_test.jsonl \
#       --output_csv /content/drive/MyDrive/sango_project2/eval_outputs/sentence_level.csv
#
#   zero-shot 예:
#   !python evaluate_checkpoint.py \
#       --method zero_shot --seed NA \
#       --dataset_name flores200 \
#       --eval_path /content/drive/MyDrive/sango_project2/splits/flores200.jsonl \
#       --output_csv /content/drive/MyDrive/sango_project2/eval_outputs/sentence_level.csv
# =====================================================================

import os, json, csv, argparse, sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

try:
    import sacrebleu
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "sacrebleu"], check=True)
    import sacrebleu

# 🔒 train_run.py와 정확히 동일한 리터럴 값. 이전에는 revision을 전혀 지정하지
# 않아 항상 "main"의 그 순간 최신 스냅샷을 받았음 — 학습에 쓴 스냅샷과
# 평가에 쓴 스냅샷이 달라질 수 있는 상태였음(lock v2 §1 HARD CHECK 위반).
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SRC_LANG = "eng_Latn"
TGT_LANG = "sag_Latn"
MAX_LEN = 128
BATCH_SIZE = 8

# train_run.py의 generation_num_beams와 반드시 동일해야 함.
NUM_BEAMS = 1

# 🔒 01_manifests/metric_lock.json(2026-08-16)과 정확히 동일한 값을 명시.
# 이전엔 일부 파라미터만 지정하고 나머지는 라이브러리 default에 의존했음
# (TER은 파라미터가 아예 없었음) — sacrebleu 2.6.0의 default가 우연히 같은
# 값이라 지금까지는 문제없었지만, 이 파일이 논문에 실릴 최종 수치를 만드는
# 스크립트이므로 default 의존을 남겨두지 않음.
CHRF_METRIC = sacrebleu.CHRF(char_order=6, word_order=0, beta=2,
                              lowercase=False, whitespace=False)
BLEU_METRIC = sacrebleu.BLEU(tokenize="13a", smooth_method="exp",
                              lowercase=False, effective_order=False)
TER_METRIC = sacrebleu.TER(normalized=False, no_punct=False,
                            asian_support=False, case_sensitive=False)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    ids = [d.get("sentence_id") for d in data]
    if any(i is None for i in ids):
        raise ValueError(
            f"{path}: 일부 행에 sentence_id가 없음. row order를 sentence_id로 "
            f"쓰면 시스템 간 문장 대응이 어긋날 수 있어 금지함. jsonl에 "
            f"고정 sentence_id를 먼저 부여할 것."
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: 중복된 sentence_id가 있음. 데이터 정합성 확인 필요.")
    return data


def read_checkpoint_metadata(adapter_path):
    """checkpoint_metadata.json이 있으면 role/step 정보를 읽어온다."""
    meta_path = os.path.join(adapter_path, "checkpoint_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_translations(model, tokenizer, sentences, device):
    outputs = []
    model.eval()
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN,
            ).to(device)
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LEN,
                num_beams=NUM_BEAMS,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(decoded)
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  ... {i + len(batch)}/{len(sentences)} sentences generated")
    return outputs


def already_evaluated(output_csv, method, seed, checkpoint_role, step, dataset_name,
                       expected_n=None):
    """동일 (method, seed, checkpoint_role, step, dataset) 조합이 이미 '완전하게'
    평가돼 있으면 True.

    단순히 행이 하나라도 있으면 완료로 보면 안 된다 — 런타임이 도중에
    끊겨 문장 일부만 기록된 경우("partial result")를 완료로 오판해 skip하면
    나머지 문장이 영원히 채워지지 않는다. 따라서:
      1) 해당 조합의 문장별 행 수가 expected_n과 일치하는지
      2) corpus summary 쪽에도 해당 조합의 행이 있는지
    둘 다 확인한다. expected_n이 주어지지 않으면(레거시 호출) 행 존재
    여부만으로 판단하되 경고를 남긴다.
    """
    if not os.path.exists(output_csv):
        return False

    matching_rows = 0
    with open(output_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["method"] == str(method) and row["seed"] == str(seed)
                    and row["checkpoint_role"] == str(checkpoint_role)
                    and row["step"] == str(step) and row["dataset"] == dataset_name):
                matching_rows += 1

    if matching_rows == 0:
        return False

    if expected_n is None:
        print("[warn] already_evaluated: expected_n이 없어 행 개수 검증을 건너뜀 "
              "(partial result를 완료로 오판할 수 있음).")
        return True

    if matching_rows != expected_n:
        print(f"[warn] partial result 감지: {matching_rows}/{expected_n}행만 존재. "
              f"완료로 보지 않고 재실행함 (기존 부분 행은 CSV에 중복으로 남으니, "
              f"본실험에서는 재실행 전에 해당 조합의 행을 수동으로 정리할 것).")
        return False

    summary_path = output_csv.replace(".csv", "_corpus_summary.csv")
    if not os.path.exists(summary_path):
        print("[warn] 문장별 행은 완전하지만 corpus_summary가 없음. 완료로 보지 않음.")
        return False
    with open(summary_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_found = any(
            row["method"] == str(method) and row["seed"] == str(seed)
            and row["checkpoint_role"] == str(checkpoint_role)
            and row["step"] == str(step) and row["dataset"] == dataset_name
            for row in reader
        )
    if not summary_found:
        print("[warn] 문장별 행은 완전하지만 corpus_summary에 해당 조합이 없음. 완료로 보지 않음.")
        return False

    return True


def evaluate_checkpoint(method, seed, dataset_name, eval_path, output_csv,
                         adapter_path=None, step_override=None, force=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.src_lang = SRC_LANG

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(device)
    assert next(base_model.parameters()).dtype == torch.float32, "backbone이 float32로 로드되지 않음"
    base_model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    base_model.generation_config.num_beams = NUM_BEAMS

    checkpoint_role = "zero_shot"
    step = "NA"
    ckpt_label = "zero_shot"

    if method == "zero_shot":
        model = base_model
    else:
        if not adapter_path:
            raise ValueError("PEFT evaluation requires --adapter_path")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False).to(device)
        ckpt_label = adapter_path

        meta = read_checkpoint_metadata(adapter_path)
        if meta:
            checkpoint_role = meta.get("checkpoint_role", "unknown")
            step = meta.get("global_step", step_override or "unknown")
        else:
            checkpoint_role = step_override or "unknown"
            step = step_override or "unknown"
            print(f"[warn] {adapter_path}에 checkpoint_metadata.json이 없음. "
                  f"--step 인자값({step_override})을 그대로 사용함.")

    print(f"[eval] {method} seed={seed} role={checkpoint_role} step={step} "
          f"on {dataset_name} (device={device}, beams={NUM_BEAMS})")

    # dedup/partial 체크에 필요한 expected_n을 얻기 위해 데이터를 먼저 로드.
    data = load_jsonl(eval_path)
    sentence_ids = [d["sentence_id"] for d in data]
    sources = [d["en"] for d in data]
    references = [d["sg"] for d in data]

    if not force and already_evaluated(output_csv, method, seed, checkpoint_role, step,
                                        dataset_name, expected_n=len(data)):
        print(f"[skip] 이미 완전하게 평가된 조합입니다 (method={method}, seed={seed}, "
              f"role={checkpoint_role}, step={step}, dataset={dataset_name}, n={len(data)}). "
              f"--force로 강제 재실행 가능 (단, master CSV에는 중복으로 append됨 — "
              f"본실험에서는 --force 대신 잘못된 행을 먼저 지우고 재실행할 것).")
        return None

    hyps = generate_translations(model, tokenizer, sources, device)

    # ---- corpus-level metrics ----
    bleu_score = BLEU_METRIC.corpus_score(hyps, [references])
    chrf_score = CHRF_METRIC.corpus_score(hyps, [references])
    ter_score = TER_METRIC.corpus_score(hyps, [references])

    print(f"[result] {method} seed={seed} step={step} {dataset_name}: "
          f"BLEU={bleu_score.score:.2f} chrF={chrf_score.score:.2f} "
          f"TER={ter_score.score:.2f} (n={len(data)})")

    # ---- 문장별 저장 (통계분석 원자료) ----
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["method", "seed", "checkpoint_role", "step", "dataset",
                              "sentence_id", "src", "ref", "hyp", "sentence_chrf",
                              "checkpoint_path"])
        for sid, src, ref, hyp in zip(sentence_ids, sources, references, hyps):
            sent_chrf = CHRF_METRIC.sentence_score(hyp, [ref]).score
            writer.writerow([method, seed, checkpoint_role, step, dataset_name,
                              sid, src, ref, hyp, round(sent_chrf, 3), ckpt_label])

    # ---- corpus-level summary 별도 저장 (signature 포함) ----
    # 주의: score 객체의 .format(signature=True)는 에러 없이 조용히 쓰레기
    # 문자열을 만든다 (signature 파라미터는 bool이 아니라 str을 기대함).
    # 반드시 metric 객체의 get_signature()를 써야 함.
    bleu_signature = BLEU_METRIC.get_signature().format()
    chrf_signature = CHRF_METRIC.get_signature().format()
    ter_signature = TER_METRIC.get_signature().format()

    summary_path = output_csv.replace(".csv", "_corpus_summary.csv")
    summary_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(["method", "seed", "checkpoint_role", "step", "dataset",
                              "n_sentences", "bleu", "chrf", "ter",
                              "bleu_signature", "chrf_signature", "ter_signature",
                              "checkpoint_path"])
        writer.writerow([method, seed, checkpoint_role, step, dataset_name, len(data),
                          round(bleu_score.score, 3), round(chrf_score.score, 3),
                          round(ter_score.score, 3),
                          bleu_signature, chrf_signature, ter_signature,
                          ckpt_label])

    return {"bleu": bleu_score.score, "chrf": chrf_score.score, "ter": ter_score.score, "n": len(data)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None,
                         help="best_adapter / final_adapter / history_adapters/step_XXXXX 경로. "
                              "method=zero_shot이면 생략.")
    parser.add_argument("--method", required=True, choices=["zero_shot", "lora", "dora", "oft"])
    parser.add_argument("--seed", required=True, help="42 | 43 | 44 | NA(zero-shot)")
    parser.add_argument("--step", default=None,
                         help="checkpoint_metadata.json이 없을 때만 사용되는 fallback. "
                              "'best' | 'final' | 정수.")
    parser.add_argument("--dataset_name", required=True,
                         choices=["adaptation_test", "flores200"])
    parser.add_argument("--eval_path", required=True,
                         help="jsonl 파일, 각 줄 {\"sentence_id\":..., \"en\": ..., \"sg\": ...}")
    parser.add_argument("--output_csv", required=True,
                         help="문장별 결과를 append할 CSV 경로 (Drive 위치 권장)")
    parser.add_argument("--force", action="store_true",
                         help="이미 평가된 조합이어도 강제로 다시 실행하고 append")
    args = parser.parse_args()

    evaluate_checkpoint(
        method=args.method,
        seed=args.seed,
        dataset_name=args.dataset_name,
        eval_path=args.eval_path,
        output_csv=args.output_csv,
        adapter_path=args.adapter_path,
        step_override=args.step,
        force=args.force,
    )
