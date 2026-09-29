# === RECOVERY CELL (2026-08-19 v8, 학습 9/9 + RQ1 전체 완료, 다음은 history/RQ2) ===
from google.colab import drive
drive.mount('/content/drive')

import os, json, sys, gc, subprocess
ROOT = '/content/drive/MyDrive/sango_project2'
os.chdir(ROOT)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', f'{ROOT}/requirements_phase2_frozen.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)

import torch, transformers, datasets, peft, sacrebleu, csv, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, OFTConfig, TaskType, get_peft_model
from sacrebleu.significance import PairedTest

def json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

def atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=json_default)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

MODEL_ID = 'facebook/nllb-200-distilled-600M'
MODEL_REVISION = 'f8d333a098d19b4fd9a8b18f94170487ad3f821d'
SRC_LANG, TGT_LANG = 'eng_Latn', 'sag_Latn'
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

gpu_lock_path = os.path.join(ROOT, '01_manifests', 'gpu_class_lock.json')
current_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
with open(gpu_lock_path, encoding='utf-8') as f:
    gpu_lock = json.load(f)
assert current_gpu == gpu_lock['gpu_name'], f"GPU mismatch: locked={gpu_lock['gpu_name']}, current={current_gpu}"

DIRECT9K_TRAIN_PATH = os.path.join(ROOT, 'data_build', 'direct9k_train.jsonl')
ADAPTATION_DEV_PATH = os.path.join(ROOT, 'evaluation', 'adaptation_dev.jsonl')
ADAPTATION_TEST_PATH = os.path.join(ROOT, 'evaluation', 'adaptation_test.jsonl')
FLORES_DEV_PATH = os.path.join(ROOT, 'flores', 'flores_dev_eng_sag.jsonl')
FLORES_DEVTEST_PATH = os.path.join(ROOT, 'flores', 'flores_devtest_eng_sag.jsonl')

RUN_ORDER = [('LoRA', 42), ('DoRA', 42), ('OFT', 42), ('DoRA', 43), ('OFT', 43),
             ('LoRA', 43), ('OFT', 44), ('LoRA', 44), ('DoRA', 44)]
train_done = sum(1 for m, s in RUN_ORDER
                  if os.path.exists(os.path.join(ROOT, '03_runs', m, f'seed{s}', 'COMPLETED.json')))

rq1_files = {
    'zero-shot(4)': ['02_zero_shot/zero_shot_adaptation_test.csv', '02_zero_shot/zero_shot_adaptation_test_pass2.csv',
                      '02_zero_shot/zero_shot_flores_devtest.csv', '02_zero_shot/zero_shot_flores_devtest_pass2.csv'],
    'best+final 평가(2)': ['04_final_outputs/adaptation_test/adaptation_test_results.csv',
                            '04_final_outputs/flores_devtest/flores_devtest_results.csv'],
    'RQ3 master': ['03_runs/rq3_master_table.json'],
    'dev/test audit': ['evaluation/dev_test_composition_audit.json'],
    'bootstrap/Holm': ['05_statistics/bootstrap_holm_chrf_results_FINAL.csv'],
    'sensitivity 3종': ['05_statistics/best_vs_final_sensitivity.csv', '05_statistics/a_only_sensitivity.csv',
                         '05_statistics/dual_reference_sensitivity.csv'],
}

print('=' * 60)
print('[ok] recovered — torch', torch.__version__, '| GPU', current_gpu)
print(f'[학습] {train_done}/9 완료')
print('[RQ1 파이프라인]')
rq1_all_ok = True
for label, paths in rq1_files.items():
    ok = all(os.path.exists(os.path.join(ROOT, p)) for p in paths)
    print(f"  {'[ok]' if ok else '[ ]'} {label}")
    rq1_all_ok = rq1_all_ok and ok
print('=' * 60)
if train_done == 9 and rq1_all_ok:
    print('[ok] 학습 9/9 + RQ1(zero-shot/평가/audit/통계/sensitivity 3종) 전부 완료')
    print('[다음] ⑥ history checkpoint(54개) 생성 -> RQ2 trajectory')
else:
    print('[대기] 위에서 [ ] 표시된 항목부터 이어서 진행')
print('[알림] RQ1 결론 freeze됨 — F1-F4 재검증 없이 그대로 사용, 이후는 diagnostic/history 단계')
