"""기준 ④(위험 우선 설명) 미기재 원인 분해 (D63).

추천 상품 단위로 세 단계에 위험등급이 있는지 본다.
- 원천: 상품 설명(investment_strategy) 속 운용사 등급 body_grade (D59)
- cot3: 후보 비교 단계에서 그 상품 뒤 구간에 숫자 등급 또는 공식 명칭이 있는가
- answer: 최종 답변에서 같은 방식으로 (기준 ④ D19와 같은 함수)

cot3·answer는 기준 ④와 같은 함수(risk_mention_after_name)를 써서 기준 ④ 숫자와 맞춰 볼 수 있게 한다.
판정 결과(결함 여부)는 만들지 않고 단계별 상태만 돌려준다. 원문 문장은 결과에 넣지 않는다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.extract import RISK_LABELS, _LABEL_RE, _segments_after_name, body_grade, grade_after_name, risk_mention_after_name

HAS = {"숫자 등급", "공식 명칭", "둘 다"}
# 공식 명칭 → 숫자 (금융투자협회 6등급 체계: 1등급 매우높은위험 … 6등급 매우낮은위험)
LABEL_GRADE = {"매우높은위험": 1, "높은위험": 2, "다소높은위험": 3, "보통위험": 4, "낮은위험": 5, "매우낮은위험": 6}


def source_status(keys: list, body: dict) -> tuple[str, float]:
    """추천 상품의 원천 등급 상태.
    반환: (상태, 등급) 상태는 '있음' / '본문 등급 없음' / '본문 등급 복수' / '상품 매칭 안 됨'
    애매한 경우 처리:
    - 매칭 키가 여러 개(이름 중복)면 각 키의 body_grade를 모은다. 숫자가 하나로 같으면 '있음', 둘 이상이면 '본문 등급 복수'.
    - 키 중 일부만 숫자 등급이 있으면 숫자 등급만 모은다 (없음보다 있음을 우선).
    - 증권 상품 설명에 없는 키(보험 등)는 매칭 안 된 것으로 본다.
    """
    vals = [body[k] for k in keys if k in body]
    if not vals:
        return "상품 매칭 안 됨", np.nan
    ints = {v for v in vals if isinstance(v, int)}
    if len(ints) == 1:
        return "있음", float(ints.pop())
    if len(ints) > 1 or "본문 등급 복수" in vals:
        return "본문 등급 복수", np.nan
    return "본문 등급 없음", np.nan


def mention_status(text, name, names) -> str:
    """'있음' / '없음' / '상품명 없음'. 있음 = 숫자 등급 또는 공식 명칭 (D19와 같은 정의)."""
    r = risk_mention_after_name(text, name, names)
    return "있음" if r in HAS else r


def answer_label_grades(text, name, names) -> set:
    """answer의 상품명 뒤 구간에 나온 공식 명칭들을 숫자 등급으로 바꾼 집합.
    애매한 경우 처리: 구간에 명칭이 여러 개면 모두 모은다(다른 상품 설명일 수 있어 비교는 하나뿐일 때만 한다)."""
    segs = _segments_after_name(text, name, names) or []
    out = set()
    for seg in segs:
        for m in _LABEL_RE.finditer(seg):
            out.add(LABEL_GRADE[re.sub(r"\s+", "", m.group(0))])
    return out


def cause_v11(src: str, c3: str, an: str) -> str:
    """판정기준서 v1.1 5장 미기재 원인 이름: 등급이 마지막으로 있었던 단계 기준. answer에 등급이 있으면 ''.
    - cot3 있음·answer 없음 → '답변 정리 단계에서 빠짐' (원천 본문 유무와 관계없이)
    - 원천 본문 없음·cot3 없음 → '데이터 생성 단계'
    - 원천 본문 있음·cot3 없음 → '추론 단계부터 빠짐'
    애매한 경우 처리: cot3 '상품명 없음'은 없음으로 본다 (D63과 같음). 원천이 '본문 등급 복수'·'상품 매칭 안 됨'이고 cot3도 없으면
    판정기준서에 이름이 없으므로 새 이름을 만들지 않고 '사람 확인'으로 둔다.
    """
    if an == "있음":
        return ""
    if c3 == "있음":
        return "답변 정리 단계에서 빠짐"
    if src == "본문 등급 없음":
        return "데이터 생성 단계"
    if src == "있음":
        return "추론 단계부터 빠짐"
    return "사람 확인"


def d4_cause(ctx: dict, prod: pd.DataFrame) -> pd.DataFrame:
    """judge의 ctx(recs, cots)와 상품 표를 받아 추천 상품 단위 원인 표를 돌려준다.

    열: cot_id, 분할, 상품명, 원천, 원천 등급, 등록 등급 있음, cot3, answer, 조합, answer 숫자 등급, 숫자 등급 비교,
        answer 명칭 등급, 명칭 등급 비교
    조합 이름(사용자 정의, 2026-11-23). 모두 answer에 등급이 없는(기준 ④ 미충족) 추천 상품에만 붙인다:
    - 원천 없음(본문 등급 없음)·answer 없음 → '데이터 생성 단계'
    - 원천 있음·cot3 있음·answer 없음 → '답변 정리 단계에서 빠짐'
    - 원천 있음·cot3 없음·answer 없음 → '추론 단계부터 빠짐'
    - 그 밖의 조합은 이름 없이 값 그대로 둔다.
    애매한 경우 처리:
    - '상품명 없음'(cot3·answer에 상품명이 안 나옴)은 '없음'과 따로 센다. 기준 ④(D19)는 answer의 '상품명 없음'을 미충족으로 본다.
      조합 이름에서는 '상품명 없음'을 '없음' 쪽에 넣는다 (cot3 상품명 없음 → '추론 단계부터 빠짐'). 표에는 값 그대로 남긴다.
    - 원천 '본문 등급 복수'·'상품 매칭 안 됨'은 이름 붙인 조합에 넣지 않는다.
    - 비교: answer 숫자 등급(grade_after_name, 범위 표현 제외 첫 단일 등급)과 원천 등급이 모두 있을 때만 '같음'/'다름'.
      공식 명칭은 하나만 나왔을 때만 숫자로 바꿔 따로 비교한다.
    """
    ps = prod[prod["domain"] == "증권"]
    body = dict(zip(ps["file_name"], ps["investment_strategy"].map(body_grade)))
    sc, recs = ctx["cots"], ctx["recs"]
    rows = []
    for r in recs.itertuples(index=False):
        x = sc.loc[r.cot_id]
        names = x["product_names"]
        src, sg = source_status(r.keys, body)
        c3 = mention_status(x["cot3"], r.rec_name, names)
        an = mention_status(x["answer"], r.rec_name, names)
        st, ag = grade_after_name(x["answer"], r.rec_name, names)
        lg = answer_label_grades(x["answer"], r.rec_name, names)
        lg1 = float(next(iter(lg))) if len(lg) == 1 else np.nan
        combo = ""
        if an != "있음":
            if src == "본문 등급 없음":
                combo = "데이터 생성 단계"
            elif src == "있음":
                combo = "답변 정리 단계에서 빠짐" if c3 == "있음" else "추론 단계부터 빠짐"
        cmp_ = lambda a: ("같음" if a == sg else "다름") if pd.notna(a) and pd.notna(sg) else "비교 불가"
        rows.append({"cot_id": r.cot_id, "분할": x["split"], "상품명": r.rec_name, "원천": src, "원천 등급": sg,
                     "등록 등급 있음": pd.notna(r.json_grade), "cot3": c3, "answer": an, "조합": combo,
                     "원인(v1.1)": cause_v11(src, c3, an),
                     "answer 숫자 등급": ag, "숫자 등급 비교": cmp_(ag),
                     "answer 명칭 등급": lg1, "명칭 등급 비교": cmp_(lg1)})
    return pd.DataFrame(rows)


def cot_level_d19(t: pd.DataFrame) -> pd.Series:
    """answer 상태만으로 CoT 단위 기준 ④ 단계를 다시 만든다 (D19 대조용)."""
    return t.groupby("cot_id")["answer"].agg(
        lambda s: "전부 충족" if (s == "있음").all() else ("전무" if not (s == "있음").any() else "일부"))
