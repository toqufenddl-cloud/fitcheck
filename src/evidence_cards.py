"""근거 카드 (판정기준서 v1.3, D67): 처리(제외·교정·태그)가 있거나 사람 확인인 CoT마다 적용 규칙과 근거 조항을 한 표로.

- 한 행 = (CoT, 적용 규칙, 근거 조항). 근거 조항은 src/rag/gold_articles.csv에서 규칙의 기준 유형으로 연결한다.
- 문서 시행일은 docs/regulations/regulations_index.csv.
- 원문 문장은 넣지 않는다. 범위는 숫자, 상품은 상품명과 등급만 쓴다 (P10).
애매한 경우 처리:
- 기준 ②는 규칙별로 연결한다: ②-a(C2-A) → 금소법 제17조②·③, ②-b(C2-B) → 제17조② (gold_articles 유형 '②-a'·'②-b').
- 데이터 기록 성격의 처리(기록 태그 TAG-FILE, 고객 진술 성향 태그 C2-EX, 메타 연령 교정)는 근거 조항이 없으므로
  doc_id를 비우고 '근거 연결'에 '조항 없음(데이터 기록)'으로 둔다.
- 그 밖에 유형을 못 정한 규칙이 생기면 'gold 미등록'으로 남겨 드러나게 한다 (새 연결을 만들지 않는다).
- 교정(FIX-LOG)은 corrections_log의 위치로 유형을 나눈다: cot2 허용 범위 → 범위, answer 위험등급 문장 → 설명, 메타 연령 → 조항 없음(데이터 기록).
- 원천 유형은 해당 상품(NH-Amundi 베트남 레버리지 → PRD-NHVN, 교보악사 공모주하이일드플러스 → PRD-KBAXA)을 추천한 CoT만 문서를 붙인다.
- 조건만 맞고 최종 판정을 정하지 않은 사람 확인 규칙(예: 사람 판정이 먼저 정한 CoT의 TR-HUMAN)은 카드에 넣지 않는다.
- 최신 등급은 G 조회 상태가 '확인'·'클래스 다름'인 상품만 쓴다 (클래스 다름은 표시).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CRIT_TYPE = {"③": "범위", "③태그": "범위", "이관": "이관", "④": "설명", "원천": "원천", "G": "범위"}
SRC_DOC = {"NH-Amundi 베트남 레버리지": "PRD-NHVN", "교보악사 공모주하이일드플러스": "PRD-KBAXA"}
FIX_TYPE = {"cot2 허용 범위": "범위", "answer 위험등급 문장": "설명", "메타 연령": "기록"}
RULE_TYPE = {"C2-A": "②-a", "C2-B": "②-b", "TAG-FILE": "기록", "C2-EX": "기록"}   # 기준보다 규칙으로 정하는 것
NO_CLAUSE = "조항 없음(데이터 기록)"


def _rules():
    return pd.read_csv(ROOT / "src" / "rules" / "rules.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False).set_index("rule_id")


def build_cards(j: pd.DataFrame, ctx: dict, latest: pd.DataFrame | None = None, g4: pd.DataFrame | None = None) -> pd.DataFrame:
    """j: judge() 결과, ctx: judge()의 ctx, latest: outputs/latest_grade_check.csv, g4: outputs/latest_grade_rejudge.csv."""
    rules = _rules()
    gold = pd.read_csv(ROOT / "src" / "rag" / "gold_articles.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    idx = pd.read_csv(ROOT / "docs" / "regulations" / "regulations_index.csv", encoding="utf-8-sig", dtype=str,
                      keep_default_na=False).set_index("doc_id")["시행일"]
    c, rec, rng = ctx["cots"], ctx["recs"], ctx["range"]
    corr = ctx["corrections"]
    fix_pos = corr.groupby(corr["cot_id"].astype(int))["위치"].agg(lambda s: sorted(set(s)))

    lt = {}
    if latest is not None:
        for n, g, s in zip(latest["상품명"], latest["최신 등급"], latest["상태"]):
            if s in ("확인", "클래스 다름") and pd.notna(g) and str(g) != "":
                lt[n] = f"{int(float(g))}" + ("(클래스 다름)" if s == "클래스 다름" else "")
    prod_str = {}
    for cid, g in rec.groupby("cot_id"):
        prod_str[cid] = " | ".join(
            f"{n}[데이터 {int(d) if pd.notna(d) else '없음'}" + (f"/최신 {lt[n]}" if n in lt else "") + "]"
            for n, d in zip(g["rec_name"], g["json_grade"]))
    g4_ids = set(g4.loc[g4["G4 태그"] != "", "cot_id"].astype(int)) if g4 is not None else set()

    def docs_for(typ, cid):
        if typ == "원천":
            names = " ".join(rec.loc[rec["cot_id"] == cid, "rec_name"])
            ds = [d for k, d in SRC_DOC.items() if k in names]
            return [(r.doc_id, r.조항) for r in gold.itertuples() if r.유형 == "원천" and r.doc_id in ds]
        return [(r.doc_id, r.조항) for r in gold.itertuples() if r.유형 == typ]

    js = j.set_index("cot_id")
    rows = []
    for cid, r in js.iterrows():
        human = "사람 확인" in (str(r["이관 판정"]) + str(r["기준③ 판정"]) + str(r.get("사유 코드", ""))) or \
            r["기준③ 판정"] == "판정 불가(범위 표기 손상)"
        acts = set(filter(None, str(r["처리 전체"]).split("|")))
        if not (acts & {"제외", "교정", "태그"} or human or cid in g4_ids):
            continue
        p = c.at[cid, "성향_D21"]
        base = {"cot_id": cid, "분할": r["분할"], "처리": r["처리 전체"], "성향(P5)": c.at[cid, "성향_P5"],
                "성향(판정 사용)": p, "cot2 허용 범위": c.at[cid, "허용등급"],
                "공개 기준 범위": "-".join(map(str, sorted(rng[p]))) if p in rng else "",
                "추천 상품(등급)": prod_str.get(cid, "")}
        applied = [x for x in str(r["적용 규칙"]).split("|") if x in rules.index]
        items = []
        for rid in applied:
            ru = rules.loc[rid]
            if ru["처리"] not in ("제외", "교정", "태그"):
                # 사람 확인 규칙은 그 기준의 최종 판정이 사람 확인일 때만 (먼저 정해진 규칙이 있으면 조건만 맞은 것)
                final = r["이관 판정"] if ru["기준"] == "이관" else r["기준③ 판정"] if ru["기준"] == "③" else ""
                if not ("사람 확인" in (ru["판정"] + ru["사유 코드"]) and final == ru["판정"]):
                    continue
            if rid == "FIX-LOG":
                for pos in fix_pos.get(cid, []):
                    items.append((rid, ru, FIX_TYPE.get(pos), f"교정: {pos}"))
            else:
                items.append((rid, ru, RULE_TYPE.get(rid, CRIT_TYPE.get(ru["기준"])), ""))
                if rid == "R3-D52":   # 원천 등급 반영 제외는 상품 공식 정보도 근거
                    items.append((rid, ru, "원천", "원천 등급 근거"))
        if cid in g4_ids:
            items.append(("G4", None, "범위", "최신 기준 부적합"))
        for rid, ru, typ, note in items:
            d = {**base, "적용 규칙": rid, "기준": ru["기준"] if ru is not None else "G",
                 "규칙 판정": ru["판정"] if ru is not None else "최신 기준 부적합",
                 "규칙 처리": ru["처리"] if ru is not None else "태그",
                 "근거 결정": ru["근거 결정"] if ru is not None else "G4, D64", "비고": note}
            docs = docs_for(typ, cid) if typ and typ != "기록" else []
            if not docs:
                link = NO_CLAUSE if typ == "기록" else "gold 미등록" if not typ else "해당 문서 없음"
                rows.append({**d, "doc_id": "", "조항": "", "문서 시행일": "", "근거 연결": link})
            for doc, art in docs:
                rows.append({**d, "doc_id": doc, "조항": art, "문서 시행일": idx.get(doc, ""), "근거 연결": f"gold_articles:{typ}"})
    cols = ["cot_id", "분할", "처리", "적용 규칙", "기준", "규칙 판정", "규칙 처리", "근거 결정", "doc_id", "조항", "문서 시행일",
            "근거 연결", "성향(P5)", "성향(판정 사용)", "cot2 허용 범위", "공개 기준 범위", "추천 상품(등급)", "비고"]
    return pd.DataFrame(rows, columns=cols)
