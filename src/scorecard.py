"""품질 점수표·사람 확인 수·민감도 (판정기준서 9장, 05c와 src.fitcheck가 같이 쓴다). 숫자만 만든다. 해석은 쓰지 않는다.

애매한 경우 처리:
- 비율의 분모는 기준별 대상 CoT (G는 상위 30개 상품을 추천한 CoT, 나머지는 증권 CoT 전체).
- 민감도의 완화 기준은 공개 기준을 성향마다 한 등급 느슨하게 한 것(적극 1–6, 위험중립 3–6, 안정추구 4–6, 안정 5–6). 공격은 그대로 1–6.
- 등급 없는 추천과 성향을 알 수 없는 CoT의 추천은 민감도에서 세지 않는다.
"""
from __future__ import annotations

import pandas as pd

SP = ["Training", "Validation"]
RELAX = {"공격투자형": set(range(1, 7)), "적극투자형": set(range(1, 7)), "위험중립형": set(range(3, 7)),
         "안정추구형": set(range(4, 7)), "안정형": {5, 6}}


def _item(j, crit, name, mask, base, action, basis, human=0):
    r = {"기준": crit, "지표": name, "처리": action, "근거": basis, "사람 확인": human}
    for s in SP + ["전체"]:
        b = base if s == "전체" else base & (j["분할"] == s)
        m = mask & b
        r[f"대상_{s}"], r[f"결함_{s}"] = int(b.sum()), int(m.sum())
        r[f"비율_{s}"] = round(m.sum() / b.sum() * 100, 1) if b.sum() else float("nan")
    return r


def quality_table(j: pd.DataFrame, ctx: dict, g4: pd.DataFrame, lg: pd.DataFrame) -> pd.DataFrame:
    ALL = pd.Series(True, index=j.index)
    rows = [
        _item(j, "①", "법정 3항목 중 미확인 있음", j["기준① 법정 미확인"] != "", ALL, "점수 보고", "금소법 제17조②2호"),
        _item(j, "①", "6항목 중 확인 4개 이하", j["기준① 확인 수"].astype(int) <= 4, ALL, "점수 보고", "판정기준서 1장"),
        _item(j, "②", "②-a 근거 부족 확정", j["기준② 판정"] == "②-a 근거 부족 확정", ALL, "태그", "금소법 제17조②2호·③, D36"),
        _item(j, "②", "②-b 성향 미확정 추천", j["기준② 판정"] == "②-b 성향 미확정 추천", ALL, "태그", "금소법 제17조②·③"),
        _item(j, "③", "범위 밖 (제외)", j["기준③ 처리"] == "제외", ALL, "제외", "KIS 준칙 제6조①, D55·D57·D52·D47",
              human=int((j["기준③ 판정"] == "판정 불가(범위 표기 손상)").sum())),
        _item(j, "③", "판정 불가 (성향·추천·등급 없음, 표기 손상)", j["기준③ 판정"].str.startswith("판정 불가"), ALL, "판정 불가", "D21·D34·D10·D57"),
        _item(j, "③태그", "공개 기준보다 넓은 허용 범위", j["D30 태그"] != "", ALL, "태그(교정판 교정)", "D30"),
        _item(j, "④", "일부 기재", j["기준④ 판정"] == "일부 기재", ALL, "태그 / 교정판 등급 문장", "금소법 제19조①, D19"),
        _item(j, "④", "기재 없음", j["기준④ 판정"] == "기재 없음", ALL, "태그 / 교정판 등급 문장", "금소법 제19조①, D19"),
        _item(j, "이관", "성향 밖 등급 직접 추천·이관 없음", j["이관 판정"] == "결함", ALL, "태그",
              "KIS 준칙 제6조③, 표준준칙 10.4)·회사참고사항 10-4", human=int((j["이관 판정"] == "사람 확인").sum())),
        _item(j, "원천", "원천 확인 필요 상품 추천", j["원천 확인 필요"].astype(str) == "True", ALL, "표시", "D51·D59"),
    ]
    gt = set(g4.loc[g4["G4 태그"] != "", "cot_id"].astype(int))
    gb = set(g4["cot_id"].astype(int))
    rows.append(_item(j, "G", "최신 기준 부적합 (상위 30개 상품 추천 CoT)", j["cot_id"].isin(gt), j["cot_id"].isin(gb), "태그", "G4",
                      human=int((lg["등급 변경"] == "변경").sum())))
    return pd.DataFrame(rows)


def human_counts(j: pd.DataFrame, ctx: dict, lg: pd.DataFrame) -> dict:
    return {
        "③-3 범위 표기 손상 (CoT)": int((j["기준③ 판정"] == "판정 불가(범위 표기 손상)").sum()),
        "이관 규칙으로 분류할 수 없는 문장 (CoT)": int((j["이관 판정"] == "사람 확인").sum()),
        "G4 등급이 바뀐 상품 (상품)": int((lg["등급 변경"] == "변경").sum()),
        "G 조회 명칭 다름 (상품)": int(lg["상태"].str.startswith("사람 확인").sum()),
        "원천 확인 필요 상품 중 공식 문서 미확인 (상품)": len(ctx["mismatch_keys"]) - 1,
        "C안 LLM·규칙 불일치 (B안이라 해당 없음)": 0,
    }


def sensitivity(ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (민감도 표, 추천 행 표)."""
    recs = ctx["recs"].copy()
    recs["성향"] = recs["cot_id"].map(ctx["cots"]["성향_D21"])
    recs["분할"] = recs["cot_id"].map(ctx["cots"]["split"])
    ok = recs["json_grade"].notna() & recs["성향"].isin(RELAX)

    def outside(r):
        return [(int(g) not in r[p]) if o else False for g, p, o in zip(recs["json_grade"], recs["성향"], ok)]
    recs["공개 기준 밖"] = outside(ctx["range"]); recs["완화 기준 밖"] = outside(RELAX)
    sens = pd.concat({s: pd.DataFrame({k: {"추천": int((recs[k] & (recs["분할"] == s if s != "전체" else True)).sum()),
                                           "CoT": recs.loc[recs[k] & (recs["분할"] == s if s != "전체" else True), "cot_id"].nunique()}
                                       for k in ["공개 기준 밖", "완화 기준 밖"]}) for s in SP + ["전체"]}, axis=1)
    return sens, recs


def scorecard_rows(j, ctx, g4, lg, as_of: str, src: str = "05c_품질점수표") -> list:
    sc = quality_table(j, ctx, g4, lg)
    human = human_counts(j, ctx, lg)
    sens, recs = sensitivity(ctx)
    N = j["분할"].value_counts()

    def srow(name, t, v, a, unit, basis):
        return {"지표": name, "Training": t, "Validation": v, "전체": a, "단위": unit, "출처": src, "근거 결정": basis, "기준일": as_of}
    out = []
    for _, r in sc.iterrows():
        out.append(srow(f"{r['기준']} {r['지표']} (결함 수)", r["결함_Training"], r["결함_Validation"], r["결함_전체"], "CoT", r["근거"]))
        out.append(srow(f"{r['기준']} {r['지표']} (비율 %)", r["비율_Training"], r["비율_Validation"], r["비율_전체"], "%", r["근거"]))
    for k, v in human.items():
        out.append(srow(f"사람 확인: {k}", "", "", v, "건", "판정기준서 7장"))
    for k in ["공개 기준 밖", "완화 기준 밖"]:
        for u in ["추천", "CoT"]:
            out.append(srow(f"민감도: {k} ({u})", sens.loc[u, ("Training", k)], sens.loc[u, ("Validation", k)], sens.loc[u, ("전체", k)], u, "판정기준서 9장"))
    nrec = {s: int((recs["분할"] == s).sum()) for s in SP}; nrec["전체"] = len(recs)
    oc = {s: sens.loc["추천", (s, "공개 기준 밖")] for s in SP + ["전체"]}
    out.append(srow("③ 공개 기준 밖 추천 (추천 수, D54 대표 수치 갱신, 이전 799)", oc["Training"], oc["Validation"], oc["전체"], "추천", "D43·D54·D65"))
    out.append(srow("③ 공개 기준 밖 추천 / 증권 추천 전체 (%)", *[round(oc[s] / nrec[s] * 100, 1) for s in SP + ["전체"]], "%", "D43·D54·D65"))
    out.append(srow("증권 추천 전체", nrec["Training"], nrec["Validation"], nrec["전체"], "추천", ""))
    out.append(srow("CoT 수", N["Training"], N["Validation"], len(j), "CoT", ""))
    return out
