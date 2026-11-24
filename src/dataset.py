"""08 정제 데이터셋 (판정기준서 v1.1 9장, D64).

- 엄격판: 증권 CoT(P1)에서 제외 처리된 CoT를 뺀 것. 교정은 적용하지 않는다.
- 교정판: 엄격판에 교정을 적용한 것.
  · P7(D46): cot2 허용 범위 표기 교정 — 기존 교정본(outputs/corrected/cot2교정_*.jsonl)의 cot2를 그대로 쓴다.
  · D22: 메타 연령 '50대 이상'·'60대 이상' → question 본문 연령대.
  · ④ 등급 문장 (제안 A): answer에 등급이 없는 추천 상품마다 answer 끝에 `"<상품명>"의 위험등급은 N등급(명칭)입니다.`를 붙인다.
    등급은 G1 최신 등급이 '확인'(같은 클래스)된 상품이면 그것, 아니면 데이터 등록 등급(risk_grade). 등급이 없으면 붙이지 않고 태그 유지.
  · D16(보험 메타 성별)은 보험 CoT 대상이라 증권 데이터셋(P1)에는 해당이 없다.
- 태그: 두 데이터셋 모두 CoT별 '태그' 컬럼 (judgments의 태그·표시 항목).
- 결과에는 원문이 들어 있으므로 outputs/corrected/ (git 제외)에 저장한다. corrections_log에는 원문을 넣지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.extract import extract_age_decade
from src.latest_grade import LABEL

ROOT = Path(__file__).resolve().parents[1]
D22_IDS = (1077, 4477, 5000)


def tags_of(j: pd.DataFrame, g4_ids: set) -> pd.Series:
    """CoT별 태그 문자열 (' | '로 잇는다). 판정기준서의 태그·표시 항목만 모은다.
    애매한 경우 처리: '결함 아님(고객 진술 성향)'도 판정기준서상 태그라 넣는다."""
    out = {}
    for _, r in j.iterrows():
        t = []
        if r["기준② 판정"] in ("②-a 근거 부족 확정", "②-b 성향 미확정 추천"):
            t.append(r["기준② 판정"])
        if r["기준② 판정"] == "결함 아님(고객 진술 성향)":
            t.append("고객 진술 성향")
        if r["D30 태그"]:
            t.append("공개 기준 대비 완화된 허용 범위")
        if r["기준④ 판정"] in ("일부 기재", "기재 없음"):
            t.append(f"④ {r['기준④ 판정']}")
        if r["이관 판정"] == "결함":
            t.append("이관 없음(성향 밖 등급 직접 추천)")
        if str(r["원천 확인 필요"]) == "True":
            t.append("원천 확인 필요")
        if r["기록 태그"]:
            t += [x.strip() for x in str(r["기록 태그"]).split(",")]
        if r.get("사유 코드", "") == "원문 깨짐":   # D67: 사람 확인 후 원문 깨짐 태그 (194)
            t.append("원문 깨짐")
        if r["cot_id"] in g4_ids:
            t.append("최신 기준 부적합")
        out[r["cot_id"]] = " | ".join(t)
    return pd.Series(out)


def grade_sentence(name: str, grade: float) -> str:
    g = int(grade)
    return f"\"{name}\"의 위험등급은 {g}등급({LABEL[g]})입니다."


def build(cot: pd.DataFrame, j: pd.DataFrame, d4: pd.DataFrame, latest: pd.DataFrame, g4_ids: set, recs: pd.DataFrame):
    """반환: (엄격판, 교정판, 새 교정 기록 행 DataFrame).
    d4: outputs/d4_cause_by_product.csv (추천 상품별 answer 등급 유무), latest: outputs/latest_grade_check.csv
    애매한 경우 처리:
    - 같은 CoT에 같은 상품명이 두 번 추천되면 문장은 한 번만 붙인다.
    - '클래스 다름'·'사람 확인'·'최신 확인 불가' 상품은 최신 등급을 쓰지 않고 등록 등급을 쓴다 (G1: 같은 상품으로 확인된 경우만 최신).
    - answer가 '상품명 없음'(answer에 상품명이 안 나옴)인 추천도 미기재로 보고 문장을 붙인다 (D19와 같음).
    """
    sc = cot[cot["category"] == "증권"].set_index("cot_id")
    keep_ids = j.loc[j["처리 전체"].str.split("|").map(lambda xs: "제외" not in xs), "cot_id"]
    tags = tags_of(j, g4_ids)
    strict = sc.loc[keep_ids].copy()
    strict["태그"] = tags.reindex(strict.index).fillna("")
    corr = strict.copy()
    logs = []

    # P7 (D46): 기존 cot2 교정본
    fixed = {}
    for sp in ("Training", "Validation"):
        p = ROOT / "outputs" / "corrected" / f"cot2교정_{sp}.jsonl"
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line); fixed[int(d["cot_id"])] = d["cot2"]
    ids = [i for i in fixed if i in corr.index]
    corr.loc[ids, "cot2"] = [fixed[i] for i in ids]

    # D22: 메타 연령
    for cid in D22_IDS:
        if cid in corr.index:
            d = extract_age_decade(corr.at[cid, "question"])
            if d:
                old = corr.at[cid, "age"]; corr.at[cid, "age"] = f"{d}0대"
                logs.append({"cot_id": cid, "위치": "메타 연령", "원래 값": old, "교정 값": f"{d}0대",
                             "근거": "question 본문 연령대", "적용 기준": "D22", "확인자": "규칙 일괄 적용"})

    # ④ 등급 문장
    lt = latest.set_index("상품명")
    use_latest = {n: float(g) for n, g, s in zip(latest["상품명"], latest["최신 등급"], latest["상태"])
                  if s == "확인" and pd.notna(g) and str(g) != ""}
    reg = recs.drop_duplicates(["cot_id", "rec_name"]).set_index(["cot_id", "rec_name"])["json_grade"]
    miss = d4[(d4["answer"] != "있음") & d4["cot_id"].isin(corr.index)].drop_duplicates(["cot_id", "상품명"])
    add = {}
    for r in miss.itertuples(index=False):
        g = use_latest.get(r.상품명, reg.get((r.cot_id, r.상품명)))
        src = "G1 최신 등급" if r.상품명 in use_latest else "데이터 등록 등급"
        if g is None or pd.isna(g):
            continue
        add.setdefault(r.cot_id, []).append(grade_sentence(r.상품명, g))
        logs.append({"cot_id": r.cot_id, "위치": "answer 위험등급 문장", "원래 값": "(없음)", "교정 값": f"{r.상품명}: {int(g)}등급",
                     "근거": f"④ 미기재 상품 등급 문장 추가 ({src})", "적용 기준": "판정기준서 v1.1 5장 (D64), 금소법 제19조①",
                     "확인자": "규칙 일괄 적용"})
    for cid, ss in add.items():
        corr.at[cid, "answer"] = str(corr.at[cid, "answer"]).rstrip() + " " + " ".join(ss)
    return strict.reset_index(), corr.reset_index(), pd.DataFrame(logs)


COLS = ["cot_id", "split", "category", "gender", "age", "query_type", "question", "cot1", "cot2", "cot3", "answer", "product_names", "태그"]
NEW_LOG = ["메타 연령", "answer 위험등급 문장"]


def save(strict: pd.DataFrame, corr: pd.DataFrame, newlog: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """엄격판·교정판을 outputs/corrected/(git 제외)에 분할별 jsonl로 저장하고, corrections_log에서 D22·④ 기록을 지우고 새로 쓴다.
    반환: 새 corrections_log."""
    od = root / "outputs" / "corrected"; od.mkdir(parents=True, exist_ok=True)
    for name, df in [("엄격판", strict), ("교정판", corr)]:
        for sp, g in df.groupby("split"):
            with open(od / f"{name}_{sp}.jsonl", "w", encoding="utf-8") as f:
                for rec in g[COLS].to_dict("records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lp = root / "outputs" / "corrections_log.csv"
    old = pd.read_csv(lp, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    log = pd.concat([old[~old["위치"].isin(NEW_LOG)], newlog.astype(str)], ignore_index=True)
    log.to_csv(lp, index=False, encoding="utf-8-sig")
    return log


def dataset_rows(strict, corr, newlog, d4, as_of: str, src: str = "08_정제데이터셋") -> list:
    """scorecard '정제 데이터셋 (08)' 구역 행."""
    R = lambda k, t, v, a, u: {"지표": k, "Training": t, "Validation": v, "전체": a, "단위": u, "출처": src,
                               "근거 결정": "판정기준서 9장, D64", "기준일": as_of}
    cnt = lambda df, sp: int((df["split"] == sp).sum())
    a = newlog[newlog["위치"] == "answer 위험등급 문장"] if len(newlog) else pd.DataFrame(columns=["cot_id", "위치"])
    cc = a.merge(corr[["cot_id", "split"]], on="cot_id")
    miss = d4[(d4["answer"] != "있음") & d4["cot_id"].isin(corr["cot_id"])].drop_duplicates(["cot_id", "상품명"])
    s, c = strict.set_index("cot_id"), corr.set_index("cot_id")
    nt = strict[strict["태그"] == ""]
    return [R("엄격판 CoT", cnt(strict, "Training"), cnt(strict, "Validation"), len(strict), "CoT"),
            R("교정판 CoT", cnt(corr, "Training"), cnt(corr, "Validation"), len(corr), "CoT"),
            R("교정: cot2 허용 범위 (P7)", "", "", int((s["cot2"].astype(str) != c["cot2"].astype(str)).sum()), "CoT"),
            R("교정: 메타 연령 (D22)", "", "", int((s["age"].astype(str) != c["age"].astype(str)).sum()), "CoT"),
            R("교정: ④ 등급 문장 추가 상품", int((cc["split"] == "Training").sum()), int((cc["split"] == "Validation").sum()), len(cc), "추천"),
            R("교정: ④ 등급 문장 추가 CoT", cc.loc[cc["split"] == "Training", "cot_id"].nunique(),
              cc.loc[cc["split"] == "Validation", "cot_id"].nunique(), cc["cot_id"].nunique(), "CoT"),
            R("④ 미기재인데 등급이 없어 문장 미추가 상품", "", "", len(miss) - len(a), "추천"),
            R("태그 없는 CoT (엄격판)", cnt(nt, "Training"), cnt(nt, "Validation"), len(nt), "CoT")]
