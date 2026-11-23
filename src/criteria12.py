"""기준 ① 필수정보 충족, 기준 ② 성향 판단 근거 (판정기준서 v1.2 1·2장, D64·D65).

판정기준서의 키워드·순서를 그대로 옮겼다. 참고 구현(docs/reference/기준12_참고구현.py)과 같은 결과가 나와야 한다.
원문 문장은 결과에 넣지 않는다.
"""
from __future__ import annotations

import re

import pandas as pd

NM = r"(공격투자형|적극투자형|위험중립형|안정추구형|안전추구형|안정형)"

# 기준 ① 확인 키워드 (판정기준서 1장 표 '확인 키워드')
ITEMS = {
    "취득 목적": r"목적|목표|자산\s*증식|노후|은퇴|주택|결혼|학자금|교육비|여유\s*자금|마련|생활\s*자금",
    "재산상황": r"소득|연봉|자산|보유\s*(?:자금|금액)|투자금|[천백]?\s*만\s*원|억\s*원|부채|대출|저축",
    # v1.2 (D65, 사용자 결정 2026-11-23): 예적금 등 '거래' 표현을 취득 경험으로 본다. '선호해 왔다'만 있으면 미확인.
    "취득 경험": (r"거래해\s*(?:본|온|왔)|거래한|거래\s*(?:이력|경험)|위주로\s*만?\s*(?:거래|운용|투자|가입)"
              r"|보유(?:한|하고\s*있는|\s*중인)\s*(?:상품|펀드|주식|채권)|경험|투자해\s*(?:본|온|왔)|운용해\s*(?:본|온|왔)|운용한"
              r"|처음\s*(?:투자|접)|(?:상품|펀드|주식|채권|ETF|예금|적금)[^.?!]{0,20}(?:보유|운용|투자해)"),
    "상품이해도": r"이해|알고\s*있|잘\s*모르|처음\s*접하|구조를",
    "위험에 대한 태도": r"감수|감내|손실|원금\s*보장|안정적|변동성|위험\s*등급|위험을|위험은|위험도",
    "연령": r"\d0\s*대|\d{2}\s*세",
}
LEGAL = ["취득 목적", "재산상황", "취득 경험"]   # 금소법 제17조②2호


def _norm(s: str) -> str:
    """P5: '안전추구형'은 안정추구형으로 읽는다."""
    return "안정추구형" if s == "안전추구형" else s


def p5_propensity(cot2: str, cot1: str) -> tuple[str, str]:
    p, src, _ = p5_detail(cot2, cot1)
    return p, src


def p5_detail(cot2: str, cot1: str) -> tuple[str, str, set]:
    """P5 성향과 판단에 쓴 후보 성향 집합. 반환: (성향 또는 '복수'/'없음', 출처 'cot2 정형'/'cot2'/'cot1', 후보 집합).
    애매한 경우 처리:
    - cot2에 '성향명(…숫자)' 정형 표현이 서로 다른 성향으로 둘 이상이면 복수.
    - 정형 표현이 하나의 성향이면 그 성향 (1순위). 없으면 cot2의 성향 명칭이 하나일 때 그 성향 (2순위), 둘 이상이면 복수.
    - cot2에 성향 명칭이 없으면 cot1 (3순위). cot1도 없으면 없음.
    """
    c2, c1 = str(cot2), str(cot1)
    par = {_norm(x) for x in re.findall(NM + r"\s*[(（:]\s*(?:허용[^)）]*)?\d", c2)}
    if len(par) > 1:
        return "복수", "cot2 정형", par
    m = re.findall(NM + r"\s*[,(（]\s*허용", c2) + re.findall(r"'" + NM + r"\s*,\s*허용", c2)
    s = {_norm(x) for x in m}
    if len(s) == 1:
        return next(iter(s)), "cot2 정형", s
    s = {_norm(x) for x in re.findall(NM, c2)}
    if len(s) == 1:
        return next(iter(s)), "cot2", s
    if len(s) > 1:
        return "복수", "cot2", s
    s = {_norm(x) for x in re.findall(NM, c1)}
    if len(s) == 1:
        return next(iter(s)), "cot1", s
    return ("복수" if s else "없음"), "cot1", s


def judge12(cots: pd.DataFrame) -> pd.DataFrame:
    """증권 CoT 표(cot_id 색인, question·cot1·cot2·age·product_names)를 받아 기준 ①② 결과를 돌려준다.

    기준 ①: question + cot1에서 항목별 '확인'/'미확인' (연령은 본문에 없고 메타 age가 있으면 '메타'). 처리 없음(점수 보고).
    기준 ② (판정 순서):
      1. 성향(P5)이 없음·복수: 추천이 있으면 '②-b 성향 미확정 추천', 없으면 '판정 불가(추천 없음)'
      2. question에 고객이 말한 성향이 하나이고 P5 성향과 같음: '결함 아님(고객 진술 성향)'
      3. 법정 3항목 중 미확인이 있음: '②-a 근거 부족 확정'
      4. 그 밖: '결함 아님'
    애매한 경우 처리 (판정기준서에 적힌 그대로, 새 규칙 없음):
    - 키워드가 하나라도 나오면 '확인'으로 본다. 판정기준서 1장의 '확인으로 보지 않는 경우'(관심만 있음, 자녀·부모 나이,
      상품 설명 요청)는 참고 구현과 같이 키워드 단계에서 따로 거르지 않는다 (알려진 한계, 판정기준서 1장).
    - question에 성향 명칭이 둘 이상이면 고객 진술 성향으로 보지 않는다.
    """
    rows = []
    for cid, d in cots.iterrows():
        t = f"{d['question']} {d['cot1']}"
        it = {k: ("확인" if re.search(p, t) else "미확인") for k, p in ITEMS.items()}
        if it["연령"] == "미확인" and str(d.get("age", "") or "").strip() not in ("", "nan"):
            it["연령"] = "메타"
        p, src, cand = p5_detail(d["cot2"], d["cot1"])
        stated = {_norm(x) for x in re.findall(NM, str(d["question"]))}
        rec = bool(d["product_names"]) if isinstance(d["product_names"], list) else bool(str(d["product_names"]).strip())
        miss = [k for k in LEGAL if it[k] == "미확인"]
        if p in ("없음", "복수"):
            c2 = "②-b 성향 미확정 추천" if rec else "판정 불가(추천 없음)"
        elif len(stated) == 1 and p in stated:
            c2 = "결함 아님(고객 진술 성향)"
        elif miss:
            c2 = "②-a 근거 부족 확정"
        else:
            c2 = "결함 아님"
        rows.append({"cot_id": cid, **{f"①{k}": v for k, v in it.items()},
                     "① 확인 수": sum(v != "미확인" for v in it.values()), "① 법정 미확인": "|".join(miss),
                     "성향_P5": p, "성향 출처": src, "P5 후보": "|".join(sorted(cand)), "고객 진술 성향": "|".join(sorted(stated)), "② 분류": c2})
    return pd.DataFrame(rows).set_index("cot_id")
