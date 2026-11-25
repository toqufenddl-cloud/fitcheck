"""품질 점수 (D70, 판정기준서 v1.4 11장). 판정 결과 데이터프레임을 받아 점수를 돌려준다.

금융감독원 미스터리쇼핑 방식을 빌린다. CoT 1건 = 판매 상담 1회. 항목을 지키면 배점, 못 지키면 0점(일부 이행도 0점).
측정하는 항목만 100점으로 환산한다: 성향 파악 영역 4항목(①·②·③·이관) 각 7.5점, 투자 위험 설명(④) 10점 → 40점 → 100점.

입력 컬럼 (judge()의 결과와 같은 이름):
  기준① 법정 미확인, 기준② 판정, 기준③ 판정, 이관 판정, 기준④ 판정

애매한 경우 처리:
- 판정할 수 없는 항목(③·④ '판정 불가…', ① 값 없음)은 그 CoT에서 빼고, 나머지 배점 합으로 100점 환산한다.
  항목이 하나도 판정되지 않은 CoT는 점수 없음(NaN)으로 두고 평균에서 뺀다.
- ③은 '범위 밖'으로 시작하는 판정(사람 판정·원천 등급 반영 포함)을 미이행으로 본다. '태그'(범위 표기 손상)는 판정 불가.
- 이관은 '결함'만 미이행. '해당 없음'·'언급 없음'·'②-b로 이관'은 이행 (②-b는 ②에서 감점한다).
- ②는 ②-a·②-b만 미이행. '결함 아님(고객 진술 성향)'은 이행.
- ④는 '전부 기재'만 이행. '일부 기재'도 0점.
- 등급 경계값은 아래 GRADES (미흡 60점대는 기사에 없어 보통과 저조 사이로 정했다, D70).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (키, 이름, 배점, 근거)
ITEMS = [
    ("①", "① 필수정보 확인", 18.75, "금소법 제17조②"),
    ("②", "② 성향 판단 근거", 18.75, "금소법 제17조②·③"),
    ("③", "③ 적합한 상품 권유", 18.75, "투자권유준칙 제6조①"),
    ("이관", "이관 (부적합 안내)", 18.75, "투자권유준칙 제6조③"),
    ("④", "④ 투자 위험 설명", 25.0, "금소법 제19조①"),
]
WEIGHT = {k: w for k, _, w, _ in ITEMS}
NAME = {k: n for k, n, _, _ in ITEMS}
LAW = {k: l for k, _, _, l in ITEMS}
GRADES = [(90, "우수"), (80, "양호"), (70, "보통"), (60, "미흡"), (-1, "저조")]


def grade(score: float) -> str:
    if score is None or pd.isna(score):
        return ""
    return next(g for cut, g in GRADES if score >= cut)


def item_pass(j: pd.DataFrame) -> pd.DataFrame:
    """CoT별 항목 이행 (1 이행, 0 미이행, NaN 판정 불가). index = cot_id."""
    def s(c):
        return j[c].fillna("").astype(str)
    p1 = s("기준① 법정 미확인").eq("").astype(float)
    p2 = (~s("기준② 판정").str.startswith(("②-a", "②-b"))).astype(float)
    v3 = s("기준③ 판정")
    p3 = pd.Series(np.where(v3.str.startswith("범위 밖"), 0.0, np.where(v3.eq("범위 안"), 1.0, np.nan)), index=j.index)
    pt = (~s("이관 판정").eq("결함")).astype(float)
    v4 = s("기준④ 판정")
    p4 = pd.Series(np.where(v4.eq("전부 기재"), 1.0, np.where(v4.isin(["일부 기재", "기재 없음"]), 0.0, np.nan)), index=j.index)
    out = pd.DataFrame({"①": p1, "②": p2, "③": p3, "이관": pt, "④": p4})
    out.index = j["cot_id"].astype(int).values
    out.index.name = "cot_id"
    return out


def _with_score(p: pd.DataFrame) -> pd.DataFrame:
    w = pd.Series(WEIGHT)
    p = p[list(WEIGHT)].copy()
    got = (p * w).sum(axis=1, min_count=1)
    avail = p.notna().mul(w).sum(axis=1)
    p["점수"] = (got / avail.replace(0, np.nan) * 100).round(2)
    return p


def cot_scores(j: pd.DataFrame) -> pd.DataFrame:
    """CoT별 항목 이행과 점수(100점 환산). 원문 없음."""
    return _with_score(item_pass(j)).reset_index()


def summarize(p: pd.DataFrame) -> dict:
    """항목 이행 표(item_pass 또는 outputs/score_cot.csv의 한 판)로 점수를 낸다.
    반환: 점수, 등급, CoT 수, 점수 없는 CoT 수, 항목표(배점, 평균 득점 = 배점 × 이행률, 감점, 이행률, 해당 건수, 미이행 건수)."""
    c = _with_score(p)
    rows = []
    for k, n, w, law in ITEMS:
        x = c[k].dropna()
        rate = float(x.mean()) if len(x) else np.nan
        rows.append({"항목": n, "키": k, "근거": law, "배점": w, "평균 득점": round(w * rate, 2), "감점": round(w * (1 - rate), 2),
                     "이행률": round(rate * 100, 1), "해당 건수": int(len(x)), "미이행 건수": int((x == 0).sum())})
    sc = round(float(c["점수"].mean()), 1)
    return {"점수": sc, "등급": grade(sc), "CoT": int(c["점수"].notna().sum()), "점수 없음": int(c["점수"].isna().sum()),
            "항목": pd.DataFrame(rows)}


def item_table(j: pd.DataFrame) -> pd.DataFrame:
    return summarize(item_pass(j))["항목"]


def data_score(j: pd.DataFrame) -> dict:
    """데이터 점수 = CoT 점수 평균."""
    return summarize(item_pass(j))


def rejudge_corrected(cot: pd.DataFrame, prod: pd.DataFrame, corr: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """교정판을 교정 후 본문으로 다시 판정한다. corr: dataset.build()의 교정판 또는 outputs/corrected/교정판_*.jsonl.
    원본 cot에서 교정판 cot_id만 남기고 교정된 본문 컬럼(age, cot2, answer 등)을 덮어쓴 뒤 judge()를 돌린다."""
    from src.judge import judge
    c = cot[cot["cot_id"].isin(corr["cot_id"])].set_index("cot_id").copy()
    cr = corr.set_index("cot_id")
    for col in ["age", "question", "cot1", "cot2", "cot3", "answer"]:
        c.loc[cr.index, col] = cr[col]
    j2, _ = judge(c.reset_index(), prod, as_of=as_of)
    return j2


def load_corrected(root=None) -> pd.DataFrame:
    import json
    from pathlib import Path
    root = Path(root or Path(__file__).resolve().parents[1])
    rows = []
    for sp in ("Training", "Validation"):
        with open(root / "outputs" / "corrected" / f"교정판_{sp}.jsonl", encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    return pd.DataFrame(rows)


def live_frame(res: pd.DataFrame, j: pd.DataFrame, basis: str) -> pd.DataFrame:
    """live 결과(run_live의 result)를 점수 입력 형식으로. basis: '데이터 기준' 또는 '최신 문서 기준'.
    ①은 저장 판정(j)에서, ③은 live의 해당 기준 열에서, ②·이관·④는 live 결과(저장 판정)에서 가져온다."""
    jj = j.set_index("cot_id")
    ids = res["cot_id"].astype(int)
    return pd.DataFrame({"cot_id": ids.values,
                         "기준① 법정 미확인": jj.loc[ids, "기준① 법정 미확인"].values,
                         "기준② 판정": res["②"].values, "기준③ 판정": res[f"③ {basis}"].values,
                         "이관 판정": res["이관"].values, "기준④ 판정": res["④"].values})


def table(scores: dict) -> pd.DataFrame:
    """report_numbers용 점수 표. scores: {'원본': data_score(...), '교정판': ...}. 행 = 항목 + 합계, 열 = 판별 득점·감점·이행률·해당 건수."""
    rows = []
    for k, n, w, law in ITEMS:
        r = {"항목": n, "배점": w, "근거": law}
        for label, s in scores.items():
            t = s["항목"].set_index("키").loc[k]
            r |= {f"{label} 평균 득점": t["평균 득점"], f"{label} 감점": t["감점"], f"{label} 이행률 %": t["이행률"],
                  f"{label} 해당 건수": int(t["해당 건수"])}
        rows.append(r)
    tot = {"항목": "점수 (CoT 점수 평균)", "배점": 100.0,
           "근거": "D70, 등급 " + " / ".join(f"{k} {v['등급']}" for k, v in scores.items())}
    for label, s in scores.items():
        tot |= {f"{label} 평균 득점": s["점수"], f"{label} 감점": round(100 - s["점수"], 1), f"{label} 이행률 %": "",
                f"{label} 해당 건수": s["CoT"]}
    rows.append(tot)
    return pd.DataFrame(rows)


def save_scores(j: pd.DataFrame, j_corr: pd.DataFrame, as_of: str, live_res: pd.DataFrame | None = None) -> dict:
    """원본·교정판 점수를 계산해 outputs/score_cot.csv(cot_id·항목 이행·점수만)와 report_tables에 저장한다.
    live_res가 있으면 live '데이터 기준'·'최신 문서 기준' 점수 표도 저장한다."""
    from pathlib import Path
    from src.report import save_table
    out = Path(__file__).resolve().parents[1] / "outputs"
    sc = {"원본": data_score(j), "교정판": data_score(j_corr)}
    per = pd.concat([cot_scores(j).assign(판="원본"), cot_scores(j_corr).assign(판="교정판")], ignore_index=True)
    per.to_csv(out / "score_cot.csv", index=False, encoding="utf-8-sig")
    save_table(table(sc), "11_품질점수", f"품질 점수: 원본 {len(j):,} CoT vs 교정판 {len(j_corr):,} CoT (교정 후 다시 판정)",
               "src/score.py", "D70, 판정기준서 v1.4 11장", as_of)
    if live_res is not None:
        lv = {b: data_score(live_frame(live_res, j, b)) for b in ("데이터 기준", "최신 문서 기준")}
        save_table(table(lv), "11_품질점수_live", f"품질 점수: live 표본 {len(live_res)} CoT, 데이터 기준 vs 최신 문서 기준",
                   "src/score.py (outputs/live)", "D69, D70", as_of)
        sc |= {f"live {b}": v for b, v in lv.items()}
    return sc
