# =====================================================================
# audit_dedup_and_laser.py — Phase 1 후속 감사 (읽기 전용)
#
# 리뷰에서 지적된 세 가지를 확인한다:
#   1. LASER score 분포 (min/max/median/mean + bucket 개수) — final_clean_pool.jsonl에
#      이미 저장된 laser_score를 그대로 쓴다. LASER를 다시 계산하지 않는다.
#   2. French contamination 제거 샘플 육안 확인 — excluded_pairs_sample.jsonl에 이미
#      reason="french_contamination"으로 저장돼 있는 걸 그대로 꺼내 보여준다.
#   3. canonical pair 중복(220,497건) 샘플 — 🔒 이건 현재 build_final_splits.py가 아예
#      저장하지 않는 정보다. stage_clean()의 gen()이 중복 pair_id를 만나면 카운터만
#      올리고 버리기 때문에, "무엇과 무엇이 합쳐졌는지" 예시가 어디에도 남지 않는다.
#      그래서 after_laser.jsonl을 프로덕션과 "동일한" 필터 함수(length/dup 필터 ->
#      auto_clean -> canonical pair_id)로 다시 스트리밍하되, final_clean_pool.jsonl은
#      새로 쓰지 않고 중복이 발생하는 순간의 (kept, dropped) 쌍만 reservoir sampling으로
#      수집한다. frozen된 final_clean_pool/candidates/train/audit/manifest는 전혀
#      건드리지 않는다 — 순수 진단용이다.
#
# 사용법 (build_final_splits.py와 같은 폴더에서):
#   python3 audit_dedup_and_laser.py --check all
#   python3 audit_dedup_and_laser.py --check laser
#   python3 audit_dedup_and_laser.py --check french
#   python3 audit_dedup_and_laser.py --check dedup
# =====================================================================

import argparse, csv, json, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_splits as bfs  # noqa: E402  (같은 폴더에 build_final_splits.py 필요)

OUTPUT_DIR = bfs.OUTPUT_DIR
DEDUP_SAMPLE_CAP = 100
DEDUP_SEED = 42
FRENCH_SAMPLE_CAP = 100
FRENCH_SAMPLE_SEED = 42


def laser_distribution():
    pool_path = OUTPUT_DIR / "final_clean_pool.jsonl"
    if not pool_path.exists():
        raise FileNotFoundError(f"{pool_path} 없음. --stage clean까지 먼저 완료할 것.")

    src_path = OUTPUT_DIR / "after_laser.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} 없음. --stage source까지 먼저 완료할 것.")

    def _dist(path, label):
        scores = [r["laser_score"] for r in bfs.iter_jsonl(path)
                  if r.get("laser_score") is not None]
        if not scores:
            raise RuntimeError(f"{path}에 laser_score가 있는 레코드가 없음.")
        scores.sort()

        # candidate 층화가 실제로 쓰는 구간(_laser_bucket: <1.1 / 1.1-1.3 / >=1.3)과
        # 일치시켜서, 이 분포가 stratification과 바로 연결되게 한다.
        buckets = {"low(<1.1)": 0, "mid(1.1-1.3)": 0, "high(>=1.3)": 0}
        for s in scores:
            buckets[{"low": "low(<1.1)", "mid": "mid(1.1-1.3)", "high": "high(>=1.3)"}[bfs._laser_bucket(s)]] += 1

        dist = {
            "n": len(scores),
            "min": scores[0],
            "max": scores[-1],
            "median": statistics.median(scores),
            "mean": statistics.fmean(scores),
            "buckets": buckets,
        }
        print(f"[laser] {label} 분포:")
        for k, v in dist.items():
            print(f"  {k}: {v}")
        return dist

    # 🔒 두 모집단을 반드시 분리해서 보고한다. "LASER>=1.0을 충족한 개수"(전체 로드
    # 자료, after_laser.jsonl)와 "최종 clean pool의 LASER 분포"(final_clean_pool.jsonl)는
    # 서로 다른 모집단이라, 하나의 분포 통계(median/range)를 다른 모집단의 개수에 붙여
    # 쓰면 아직 입증 안 된 주장이 된다.
    result = {
        "full_loaded_corpus": _dist(src_path, "전체 로드 자료(after_laser.jsonl)"),
        "final_clean_pool": _dist(pool_path, "최종 clean pool(final_clean_pool.jsonl)"),
    }

    out_path = OUTPUT_DIR / "laser_score_distribution.json"
    bfs.save_json_atomic(out_path, result)
    print(f"[laser] 저장: {out_path}")
    return result


def french_contamination_samples():
    """🔒 이 함수가 저장하는 excluded_pairs_sample.jsonl 기반 샘플은 clean 단계에서
    리즌별로 "처음 발견된 순서대로" 최대 50개까지 잘라 저장한 것이지, 무작위 표본이
    아니다. 실측 확인 결과 50개 중 서로 다른 canonical SG target은 27개뿐이고, 특정
    template 하나가 11번 반복되는 등 중복이 심하다 — "50 randomly sampled exclusions"
    라고 논문에 쓰면 안 되고, 비율 추정에도 쓰면 안 된다. 필터가 대략 어떤 종류의
    문장을 잡아내는지 육안으로 빠르게 감을 잡는 pilot 용도로만 쓸 것. 공식 감사용
    무작위 표본은 french_contamination_canonical_sample()(--check french-export)을 쓸 것."""
    excluded_path = OUTPUT_DIR / "excluded_pairs_sample.jsonl"
    if not excluded_path.exists():
        raise FileNotFoundError(f"{excluded_path} 없음. --stage clean까지 먼저 완료할 것.")

    all_excluded = bfs.load_jsonl(excluded_path)
    samples = [r for r in all_excluded if r.get("reason") == "french_contamination"]
    print(f"[french] excluded_pairs_sample.jsonl 안에 french_contamination 샘플 "
          f"{len(samples)}개 (clean 단계에서 리즌별로 처음 발견된 순서대로 최대 50개까지 "
          f"저장돼 있음 — 무작위 표본 아님, 중복도 있을 수 있음)")
    print("[french] ⚠️  이건 pilot 미리보기용이다. 논문에 넣을 공식 audit 숫자는 "
          "--check french-export(canonical pair 기준 무작위 100개)로 만들 것.")

    out_path = OUTPUT_DIR / "french_contamination_review.jsonl"
    bfs.save_jsonl_atomic(out_path, samples)
    print(f"[french] 육안 확인용 저장: {out_path}")

    for s in samples[:10]:
        print("  EN:", s["en"])
        print("  SG:", s["sg"])
        print("  ---")
    return samples


FRENCH_CATEGORIES = ("A", "C1", "C2", "U")
FRENCH_CATEGORY_LABELS = {
    "A": "명백한 French 또는 강한 French code-switching",
    "C1": "French 이외의 비-Sango 언어·파손·placeholder",
    "C2": "Sango 또는 혼합 target이지만 EN source와 의미 불일치(오정렬)",
    "U": "Sango 여부·정합성을 확정하기 어려움 (원어민 확인 없이는 판정 보류)",
}
# 🔒 "B"(유효한 Sango — 오탐 확정)는 기본 범주에 없다. 연구자가 Sango 원어민이 아닌
# 이상 "이건 정상 Sango다"라고 스스로 확정하면 안 되고, 확신이 없는 건 전부 U로 정직하게
# 남겨야 한다. B는 Deo가 실제로 "Yes"라고 확인해준 경우에만(merge_deo_answers()를 통해)
# 사후적으로 부여되는 라벨이다 — 이 audit이 필수로 요구하는 라벨은 아니다.
FRENCH_CATEGORY_LABELS_DEO_ONLY = {
    "B": "Deo가 확인한 유효 Sango (원어민 확인 거친 경우에만 부여)",
}


def french_contamination_canonical_sample(cap=FRENCH_SAMPLE_CAP, seed=FRENCH_SAMPLE_SEED):
    """🔒 공식 audit용 표본 추출. after_laser.jsonl을 프로덕션과 동일한 순서
    (length/dup 필터 -> script -> degenerate -> French)로 다시 스트리밍하되, 여기서는
    French contamination으로 걸리는 "전체" 행을 대상으로 한다 (clean 단계의 50개 cap과
    무관). 같은 canonical pair_id가 여러 번 나오면(대소문자/인용표기만 다른 반복 template
    등) 첫 등장만 모집단에 넣어 canonical-unique 집합을 만들고, 그 집합에서 seed 고정
    reservoir sampling으로 cap개를 뽑는다. french_overlap_ratio/matched_french_tokens도
    같이 계산해서 분류 근거로 남긴다. frozen된 최종 산출물은 전혀 안 건드림."""
    src_path = OUTPUT_DIR / "after_laser.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} 없음. --stage source까지 먼저 완료할 것.")

    print("[french-canonical] MEYNG vocabulary 재로딩 중 (French contamination 검사 재현에 필요)...")
    french_vocab, _ = bfs.load_sango_vocab()

    length_filtered = bfs.filter_pairs_stream(bfs.iter_jsonl(src_path))

    seen_pair_ids = set()
    reservoir = []
    rng = random.Random(seed)
    n_unique_seen = 0
    n_total_rows_matched = 0

    for r in length_filtered:
        en, sg_raw = r["en"], r["sg"]
        sg = bfs.normalize_sango_orthography(sg_raw)
        if not bfs.check_script_validity(sg):
            continue
        if not bfs.check_degenerate_pattern(en) or not bfs.check_degenerate_pattern(sg):
            continue
        if bfs.check_not_too_french(sg, french_vocab):
            continue  # French contamination 아님 -> 통과, 관심 대상 아님

        n_total_rows_matched += 1
        pid = bfs.compute_pair_id(en, sg)
        if pid in seen_pair_ids:
            continue  # canonical 중복 -> 표본 모집단에서 제외 (첫 등장만 포함)
        seen_pair_ids.add(pid)
        n_unique_seen += 1

        tokens = bfs.lexical_tokens(sg)
        matched = sorted(set(t for t in tokens if t in french_vocab))
        ratio = (sum(1 for t in tokens if t in french_vocab) / len(tokens)) if tokens else 0.0

        example = {
            "pair_id": pid, "en": en, "sg": sg,
            "french_overlap_ratio": round(ratio, 4),
            "matched_french_tokens": matched,
        }
        if len(reservoir) < cap:
            reservoir.append(example)
        else:
            j = rng.randint(0, n_unique_seen - 1)  # 표준 reservoir sampling (Algorithm R)
            if j < cap:
                reservoir[j] = example

    print(f"[french-canonical] French contamination 매치 총 {n_total_rows_matched}행 "
          f"(canonical pair_id 기준 unique: {n_unique_seen}개) -> {len(reservoir)}개 무작위 샘플")

    out_path = OUTPUT_DIR / "french_contamination_categorize.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "pair_id", "en", "sg", "french_overlap_ratio",
                          "matched_french_tokens", "category(A/C1/C2/U)", "comment"])
        for i, ex in enumerate(reservoir):
            writer.writerow([i, ex["pair_id"], ex["en"], ex["sg"],
                              ex["french_overlap_ratio"], "; ".join(ex["matched_french_tokens"]),
                              "", ""])

    print(f"[french-canonical] 저장: {out_path}")
    print("[french-canonical] category(A/C1/C2/U) 열을 채울 것 (Sango 원어민이 아니어도 "
          "전부 채울 수 있음 — Deo 확인은 필수 아님):")
    for k, v in FRENCH_CATEGORY_LABELS.items():
        print(f"  {k} = {v}")
    print("[french-canonical] \"이게 정상 Sango다\"라고 스스로 확정하지 말고, 확신 없으면 "
          "정직하게 U로 남길 것. 논문에는 \"필터가 대체로 French-dominant·다국어·파손· "
          "오정렬 target을 잡아냈고, 소수의 Sango-dominant 애매 사례는 원어민 확인 없이는 "
          "확정하지 않았다\"처럼 쓰면 된다. U를 나중에 Deo에게 확인받고 싶으면 "
          "export_deo_questions()로 그것만 골라 물어볼 수 있다(선택 사항).")
    return reservoir


def export_french_for_tagging():
    """[구버전 호환용 별칭] french_contamination_canonical_sample()을 그대로 실행한다.
    --check french-export가 이 함수를 호출한다."""
    return french_contamination_canonical_sample()


def export_deo_questions():
    """🔒 이 audit은 Deo 확인 없이 A/C1/C2/U로 끝내도 된다 (French audit ≠ 350개 evaluation
    candidate 검증 — 후자만 Deo 검증이 필수). 이 함수는 선택 사항: 정확한 false-positive
    비율까지 논문에 제시하고 싶을 때만 U로 남은 항목을 Deo에게 확인받는 용도다.
    category(A/C1/C2/U)를 채운 categorize.csv에서 U(=판정 보류)로 표시된 항목만 골라
    Deo에게 보낼 예/아니오 질문지를 Excel(.xlsx)로 만든다. Deo가 "Yes"라고 확인해주면
    그 항목만 사후적으로 B(Deo 확인된 유효 Sango)로 확정된다."""
    import csv
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    csv_path = OUTPUT_DIR / "french_contamination_categorize.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} 없음. --check french-export로 먼저 CSV를 만들고 1차 분류부터 할 것."
        )

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    needs_deo = [r for r in rows
                 if (r.get("category(A/C1/C2/U)") or "").strip().upper() in ("U", "")]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Deo Review"

    # 안내문 (legend)
    ws.merge_cells("A1:D1")
    ws["A1"] = ("Could you check whether each Sango sentence below is a normal, valid "
                "sentence? Please choose Yes or No in the 'Is this a valid Sango sentence?' column.")
    ws["A1"].font = Font(bold=True, size=12)
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 40

    headers = ["idx", "Sango sentence", "Is this a valid Sango sentence?", "Notes (optional)"]
    header_row = 3
    for col, h in enumerate(headers, start=1):
        c = ws.cell(row=header_row, column=col, value=h)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="D9E1F2")
        c.alignment = Alignment(wrap_text=True, vertical="center")

    # 예시 행 — 실제 데이터가 아님을 회색 기울임체로 표시
    example_row = header_row + 1
    example_values = ["example", "Nyen la mo sara? - Jean 18:35.", "Yes",
                       "Just a biblical citation format, otherwise a normal sentence"]
    for col, v in enumerate(example_values, start=1):
        c = ws.cell(row=example_row, column=col, value=v)
        c.font = Font(italic=True, color="808080")
        c.alignment = Alignment(wrap_text=True, vertical="top")

    start_row = example_row + 1
    for i, r in enumerate(needs_deo):
        row = start_row + i
        ws.cell(row=row, column=1, value=r["idx"])
        c_sg = ws.cell(row=row, column=2, value=r["sg"])
        c_sg.alignment = Alignment(wrap_text=True, vertical="top")

    end_row = start_row + len(needs_deo) - 1

    if needs_deo:
        dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"C{start_row}:C{end_row}")

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 65
    ws.column_dimensions["C"].width = 24
    ws.column_dimensions["D"].width = 32
    ws.freeze_panes = f"A{start_row}"

    out_path = OUTPUT_DIR / "deo_questions.xlsx"
    wb.save(out_path)

    print(f"[deo-questions] 1차 분류에서 B/D/미분류로 남은 {len(needs_deo)}개 "
          f"(전체 {len(rows)}개 중) -> {out_path}")
    print("[deo-questions] 이 xlsx 파일을 Deo에게 전달하고 'Is this a valid Sango sentence?' "
          "열에 Yes/No만 받으면 됨 (Deo가 한국어 화자가 아닐 수 있어 시트 자체는 영어로 만듦). "
          "답변 채운 뒤 같은 경로에 저장하고 merge_deo_answers()로 병합할 것.")
    return out_path


def merge_deo_answers():
    """[선택 사항] Deo가 답변을 채운 deo_questions.xlsx를 categorize.csv에 다시 병합한다.
    "Yes"(정상 Sango) -> B로 확정. "No"는 Sango가 아니라는 것만 확인된 것이라, C1(다른
    언어)인지 C2(오정렬)인지는 Deo의 Sango 판단만으로는 못 정하므로 일단 "C"(분할 대기)로
    표시하고, 최종 집계 전에 연구자가 EN/SG를 보고 C1/C2로 나눠야 한다. 아직 답변이 없는
    idx는 U로 남긴다."""
    import csv
    import openpyxl

    categorize_path = OUTPUT_DIR / "french_contamination_categorize.csv"
    deo_path = OUTPUT_DIR / "deo_questions.xlsx"
    if not categorize_path.exists():
        raise FileNotFoundError(f"{categorize_path} 없음.")
    if not deo_path.exists():
        raise FileNotFoundError(f"{deo_path} 없음. export_deo_questions()를 먼저 실행할 것.")

    wb = openpyxl.load_workbook(deo_path, data_only=True)
    ws = wb.active

    # 헤더 행을 idx 열 값으로 찾는다 — 레이아웃(안내문/예시 행 위치)이 조금 바뀌어도
    # 안전하게 동작하도록.
    header_row = None
    for r in range(1, ws.max_row + 1):
        if ws.cell(row=r, column=1).value == "idx":
            header_row = r
            break
    if header_row is None:
        raise RuntimeError(f"{deo_path}에서 헤더 행을 못 찾음 (idx 열이 있어야 함).")

    resolved = {}
    unresolved = 0
    for r in range(header_row + 1, ws.max_row + 1):
        idx_val = ws.cell(row=r, column=1).value
        if idx_val is None or str(idx_val) == "example":
            continue  # 빈 행 또는 예시 행 건너뜀
        answer_val = ws.cell(row=r, column=3).value
        answer = str(answer_val).strip() if answer_val is not None else ""
        idx = str(idx_val)
        if answer in ("Yes", "yes", "Y", "y", "예"):  # 예/y 등은 예전 버전 호환용
            resolved[idx] = "B"
        elif answer in ("No", "no", "N", "n", "아니오"):
            resolved[idx] = "C"  # C1/C2 분할은 연구자 몫 -> 일단 미분할 표시
        else:
            unresolved += 1  # 아직 Deo 답변 안 옴 -> U로 유지

    # 🔒 컬럼을 하드코딩(idx/en/sg만)하지 않고 원본 CSV의 fieldnames를 그대로 써서
    # pair_id/french_overlap_ratio/matched_french_tokens/comment가 덮어쓰기 중 날아가지
    # 않게 한다.
    with open(categorize_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    cat_col = "category(A/C1/C2/U)"
    for r in rows:
        if r["idx"] in resolved:
            r[cat_col] = resolved[r["idx"]]
        elif (r.get(cat_col) or "").strip() == "":
            r[cat_col] = "U"  # 미분류였는데 Deo 답변도 없으면 U로 명시

    with open(categorize_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_b = sum(1 for v in resolved.values() if v == "B")
    n_c = sum(1 for v in resolved.values() if v == "C")
    print(f"[deo-merge] Deo 답변 {len(resolved)}개 반영 완료 (B로 확정: {n_b}개, "
          f"C(분할 대기)로 표시: {n_c}개)")
    if n_c:
        print(f"[deo-merge] ⚠️  {cat_col} 열에서 값이 'C'인 {n_c}개는 Deo가 \"Sango 아님\"이라고만 "
              f"확인해준 것 — C1(French 외 언어)인지 C2(EN과 의미 불일치)인지는 en/sg 보고 "
              f"직접 채워야 최종 집계에 들어간다.")
    if unresolved:
        print(f"[deo-merge] 아직 Deo 답변 없는 항목 {unresolved}개는 U로 유지됨")
    print(f"[deo-merge] {categorize_path} 갱신 완료. C를 C1/C2로 나눈 뒤 tally_french_tags()로 최종 집계할 것.")


def tally_french_tags():
    """french_contamination_canonical_sample()으로 만든 CSV에 사람이 category를 채워 넣은
    뒤 집계한다."""
    import csv
    csv_path = OUTPUT_DIR / "french_contamination_categorize.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} 없음. --check french-export로 먼저 CSV를 만들고 분류부터 할 것."
        )

    counts = {k: 0 for k in FRENCH_CATEGORIES}
    counts["B"] = 0  # Deo가 실제로 "Yes"라고 확인해준 경우에만 생김 (선택 사항 워크플로)
    counts["C(미분할)"] = 0  # Deo가 "No"만 확인해주고 C1/C2로 아직 안 나뉜 것
    counts["(미분류)"] = 0
    total = 0
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            tag = (row.get("category(A/C1/C2/U)") or "").strip().upper()
            if tag in counts:
                counts[tag] += 1
            elif tag == "C":
                counts["C(미분할)"] += 1
            else:
                counts["(미분류)"] += 1

    print(f"[french-tally] 총 {total}개 중:")
    for k, v in counts.items():
        label = f" ({FRENCH_CATEGORY_LABELS_DEO_ONLY[k]})" if k in FRENCH_CATEGORY_LABELS_DEO_ONLY \
            else f" ({FRENCH_CATEGORY_LABELS[k]})" if k in FRENCH_CATEGORY_LABELS else ""
        pct = f"{v/total*100:.1f}%" if total else "0.0%"
        print(f"  {k}{label}: {v}개 ({pct})")

    if counts["C(미분할)"] > 0:
        print(f"[french-tally] 경고: {counts['C(미분할)']}개가 C1/C2로 아직 안 나뉨 — "
              f"categorize.csv를 열어 en/sg 보고 C1(French 외 언어) 또는 C2(EN과 의미 "
              f"불일치)로 채운 뒤 다시 집계할 것.")
    if counts["(미분류)"] > 0:
        print(f"[french-tally] 경고: {counts['(미분류)']}개가 아직 분류 안 됨"
              f"(빈 칸이거나 A/C1/C2/U 외 값)")

    out_path = OUTPUT_DIR / "french_contamination_tally.json"
    bfs.save_json_atomic(out_path, {"total": total, "counts": counts})
    print(f"[french-tally] 저장: {out_path}")
    return counts


def canonical_duplicate_samples(cap=DEDUP_SAMPLE_CAP, seed=DEDUP_SEED):
    src_path = OUTPUT_DIR / "after_laser.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} 없음. --stage source까지 먼저 완료할 것.")

    print("[dedup] MEYNG vocabulary 재로딩 중 (French contamination 검사 재현에 필요)...")
    french_vocab, _ = bfs.load_sango_vocab()

    # 🔒 stage_clean()과 정확히 같은 순서(length/dup 필터 -> auto_clean -> canonical
    # pair_id)로 다시 스트리밍한다. 여기서 나온 최종 살아남은 집합은 final_clean_pool과
    # 반드시 같아야 하며(같은 결정적 함수들이므로), 그 과정에서 canonical pair_id가
    # 겹치는 순간의 (먼저 살아남은 것, 방금 버려진 것) 쌍만 별도로 기록한다.
    length_filtered = bfs.filter_pairs_stream(bfs.iter_jsonl(src_path))
    cleaned = bfs.auto_clean_pairs_stream(length_filtered, french_vocab)

    first_seen = {}       # pair_id -> 먼저 살아남은 레코드
    dup_examples = []     # reservoir
    rng = random.Random(seed)
    n_dupes_seen = 0

    for r in cleaned:
        pid = bfs.compute_pair_id(r["en"], r["sg"])
        if pid in first_seen:
            n_dupes_seen += 1
            example = {
                "pair_id": pid,
                "kept": first_seen[pid],
                "dropped": {"en": r["en"], "sg": r["sg"], "sg_raw": r["sg_raw"]},
            }
            if len(dup_examples) < cap:
                dup_examples.append(example)
            else:
                j = rng.randint(0, n_dupes_seen - 1)  # classic reservoir sampling (Algorithm R)
                if j < cap:
                    dup_examples[j] = example
        else:
            first_seen[pid] = {"en": r["en"], "sg": r["sg"], "sg_raw": r["sg_raw"]}

    print(f"[dedup] 재현된 canonical 중복 총 {n_dupes_seen}건 "
          f"(filter_statistics.json의 removed_pair_id_duplicate와 일치해야 함) "
          f"-> {len(dup_examples)}개 무작위 샘플 저장")

    out_path = OUTPUT_DIR / "canonical_duplicate_review.jsonl"
    bfs.save_jsonl_atomic(out_path, dup_examples)
    print(f"[dedup] 육안 확인용 저장: {out_path}")

    for ex in dup_examples[:10]:
        print("  KEPT    EN:", ex["kept"]["en"], "| SG:", ex["kept"]["sg"],
              "| SG_RAW:", ex["kept"]["sg_raw"])
        print("  DROPPED EN:", ex["dropped"]["en"], "| SG:", ex["dropped"]["sg"],
              "| SG_RAW:", ex["dropped"]["sg_raw"])
        print("  ---")
    return dup_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        choices=["laser", "french", "dedup", "french-export",
                 "deo-questions", "deo-merge", "french-tally", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.check in ("laser", "all"):
        laser_distribution()
    if args.check in ("french", "all"):
        french_contamination_samples()
    if args.check in ("dedup", "all"):
        canonical_duplicate_samples()
    # french-export/deo-questions/deo-merge/french-tally는 "all"에 안 넣는다 — 사람의
    # 1차 분류, Deo의 답변이라는 두 번의 수동 개입이 중간에 끼는 다단계 워크플로다.
    if args.check == "french-export":
        export_french_for_tagging()
    if args.check == "deo-questions":
        export_deo_questions()
    if args.check == "deo-merge":
        merge_deo_answers()
    if args.check == "french-tally":
        tally_french_tags()
