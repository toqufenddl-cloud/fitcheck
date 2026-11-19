"""LLM 진단에 넣을 사례를 만든다. judgments.csv(규칙 판정) + 필요한 문장만.

사례 유형
- 범위: 기준③ 범위 밖·판정 불가·추론 결함 (근거 질의: 성향별 권유 가능 위험등급)
- 이관: 등급 언급이 성향 밖이거나 사람 확인·이관 결함 (근거 질의: 성향 밖 요청 시 부적합 고지·확인 절차)
- 원천: 원천 확인 필요 (근거 질의: 집합투자기구 위험등급 분류)
한 CoT가 여러 유형이면 유형마다 사례 하나씩 만든다.
원문 문장은 이 컴퓨터 안의 LLM에만 넣는다 (D60). 결과 파일에는 넣지 않는다.
"""
from __future__ import annotations

import re

import pandas as pd

from src.extract import _mask_ranges, body_grade, parse_allowed_grades
from src.transfer import sentence_mentions

_SPLIT = re.compile(r"(?<=[.!?])\s+")
MAX_SENT = 300  # 문장 하나 최대 길이 (프롬프트 길이 제한)

QUERY = {
    "범위": "투자성향 {성향} 고객에게 권유할 수 있는 금융투자상품 위험등급 적합성 원칙",
    "이관": "고객이 투자성향보다 위험이 높은 상품을 원할 때 부적합 사실 고지 확인 절차 상담",
    "원천": "집합투자기구 투자위험등급 분류 기준 설명서 기재",
}


def _short(s: str) -> str:
    s = s.strip()
    return s if len(s) <= MAX_SENT else s[:MAX_SENT] + "…"


def range_sentences(cot2: str) -> list:
    """cot2에서 허용 범위 표기가 들어 있는 문장."""
    _, _, texts = parse_allowed_grades(cot2, with_text=True)
    return [_short(s) for s in _SPLIT.split(str(cot2)) if any(t in s for t in texts)][:2]


def mention_sentences(question: str) -> list:
    """question에서 위험등급(숫자·명칭)이 나온 문장."""
    out = []
    for s in _SPLIT.split(str(question)):
        if sentence_mentions(_mask_ranges(s)):
            out.append(_short(s))
    return out[:3]


def case_types(row) -> list:
    t = []
    if str(row["기준③ 판정"]).startswith(("범위 밖", "추론 결함", "판정 불가")):
        t.append("범위")
    if row["이관 판정"] in ("결함", "사람 확인") or str(row["성향 밖 언급"] or ""):
        t.append("이관")
    if bool(row["원천 확인 필요"]):
        t.append("원천")
    return t


def build_cases(j: pd.DataFrame, ctx: dict, prod: pd.DataFrame, ids=None, types=None, force_types=None) -> list:
    """반환: 사례 dict 목록. ids로 cot_id를, types로 유형을 좁힐 수 있다.
    force_types: 규칙상 해당 유형이 아니어도 이 유형으로 만든다 (검증: 사람 판정이 있는 표본)."""
    c, rec = ctx["cots"], ctx["recs"]
    ps = prod[prod["domain"] == "증권"].set_index("file_name")
    j = j if ids is None else j[j["cot_id"].isin(set(ids))]
    cases = []
    for _, r in j.iterrows():
        cid = r["cot_id"]
        x = c.loc[cid]
        rr = rec[rec["cot_id"] == cid]
        base = {"cot_id": int(cid), "분할": r["분할"], "성향_cot2": x["성향_cot2"], "성향_D21": x["성향_D21"],
                "cot2 허용 범위": x["허용등급"], "기준 범위": sorted(ctx["range"].get(x["성향_D21"], [])),
                "추천 등급": [None if pd.isna(g) else int(g) for g in rr["json_grade"]],
                "규칙": {"기준③": r["기준③ 판정"], "처리": r["기준③ 처리"], "사유 코드": r["사유 코드"],
                         "이관 분류": r["이관 분류"], "이관 판정": r["이관 판정"], "원천 확인 필요": bool(r["원천 확인 필요"])}}
        for t in (force_types or case_types(r)):
            if types and t not in types:
                continue
            case = dict(base, 유형=t, case_id=f"{cid}:{t}",
                        질의=QUERY[t].format(성향=x["성향_D21"]))
            if t == "범위":
                case["문장"] = {"cot2 허용 범위 문장": range_sentences(x["cot2"])}
            elif t == "이관":
                case["문장"] = {"질문 속 등급 언급 문장": mention_sentences(x["question"])}
                case["이관 표현"] = bool(r["이관 표현"])
            else:
                prods = []
                for _, p in rr.iterrows():
                    for k in p["keys"]:
                        if k in ctx["mismatch_keys"] and k in ps.index:
                            desc = str(ps.loc[k, "investment_strategy"])
                            m = re.search(r".{0,80}등급.{0,80}", re.sub(r"\s+", " ", desc))
                            prods.append({"상품명": p["rec_name"], "등록 등급": None if pd.isna(p["json_grade"]) else int(p["json_grade"]),
                                          "설명 속 등급": body_grade(desc), "설명 중 등급 부분": _short(m.group(0)) if m else ""})
                case["문장"] = {"상품": prods[:3]}
            cases.append(case)
    return cases


def product_cases(names: list, prod: pd.DataFrame) -> list:
    """원천 검증용: 상품 단위 사례 (원천검수표 20건). 상품 설명은 AI Hub 원천데이터이므로 로컬 LLM에만 넣는다 (D60).
    애매한 경우 처리: 같은 이름이 여러 상품 JSON에 있으면 첫 번째를 쓴다."""
    from src.extract import grade_num
    ps = prod[prod["domain"] == "증권"].drop_duplicates("product_name").set_index("product_name")
    out = []
    for i, n in enumerate(names):
        if n not in ps.index:
            continue
        desc = re.sub(r"\s+", " ", str(ps.loc[n, "investment_strategy"]))
        m = re.search(r".{0,80}등급.{0,80}", desc)
        reg = grade_num(ps.loc[n, "risk_grade"])
        out.append({"case_id": f"P{i:02d}:원천", "cot_id": -1, "분할": "-", "유형": "원천",
                    "성향_cot2": "-", "성향_D21": "-", "cot2 허용 범위": "-", "기준 범위": [], "추천 등급": [],
                    "규칙": {"원천 확인 필요(body_grade)": None},
                    "질의": QUERY["원천"], "상품명": n,
                    "문장": {"상품": [{"상품명": n, "등록 등급": None if pd.isna(reg) else int(reg),
                                      "설명 속 등급": body_grade(desc), "설명 앞부분": _short(desc[:300]),
                                      "설명 중 등급 부분": _short(m.group(0)) if m else ""}]}})
    return out
