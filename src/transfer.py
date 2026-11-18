"""고객 등급 언급 분류와 이관 판정 (D50·D56). 키워드는 src/rules/transfer_terms.csv에서 읽는다.

판정이 아니라 '규칙으로 분류할 수 있는가'만 본다. 규칙으로 정할 수 없는 문장은 '사람 확인'으로 남긴다
(이후 C안: LLM 1차 판정 대상, D56).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.extract import _LABEL_RE, _mask_ranges

TERMS_PATH = Path(__file__).resolve().parent / "rules" / "transfer_terms.csv"
LABEL_GRADE = {"매우높은위험": 1, "높은위험": 2, "다소높은위험": 3, "보통위험": 4, "낮은위험": 5, "매우낮은위험": 6}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUM_GRADE = re.compile(r"([1-6])\s*등급")


def load_terms(path: Path = TERMS_PATH) -> dict:
    t = pd.read_csv(path, encoding="utf-8")
    return dict(zip(t["구분"], t["패턴"]))


def sentence_mentions(sent: str) -> list:
    """문장 하나에서 등급 언급을 찾는다. 반환: [(등급, 형태, 시작, 끝)] 형태 = '숫자' / '명칭'.

    - 범위·목록 표현(예: 3~6등급)은 _mask_ranges로 지운 뒤 찾는다 (02b I2와 같음).
    - 명칭은 긴 이름부터 맞춘다 ('다소 높은 위험'을 '높은 위험'(2)으로 잘못 세지 않는다).
    """
    out = [(int(m.group(1)), "숫자", m.start(), m.end()) for m in _NUM_GRADE.finditer(sent)]
    out += [(LABEL_GRADE[re.sub(r"\s+", "", m.group(0))], "명칭", m.start(), m.end()) for m in _LABEL_RE.finditer(sent)]
    return out


def classify_sentence(sent: str, grade: int, mentions: list, terms: dict) -> tuple:
    """언급 등급 하나가 나온 문장을 분류한다. 반환: (분류, 해석 등급, 사람 확인 사유)

    순서 (처음 맞는 것 하나)
    1. 회피: 회피 표현이 있거나, 언급 바로 뒤에 '보다'가 붙음 ('높은 위험보다는 …')
    2. 경험: 보유·경험 표현
    3. R4 비교 표현: '예금보다 다소 높은 위험' → 해석 등급 4
    4. R1 명확한 희망: 숫자 등급 + 요청 표현
    5. R2 고려 중: 감수·수용·감내·고민·고려 표현
    6. 사람 확인: 위 어느 것에도 해당하지 않음

    애매한 경우 처리
    - 경험과 요청이 한 문장에 함께 있으면 요청을 우선한다 (D50 갱신 2026-11-18, 근거 5011):
      숫자 등급 → R1, 고려 표현 → R2. 둘 다 아니면(명칭 + 요청만) 일반 표현일 수 있어 '사람 확인'.
    - 명칭만 있고 요청 표현만 있는 문장(예: '낮은 위험 등급 상품을 찾는')은 일반 표현일 수 있어(희망검수표1 오탐 11건) R1로 보지 않는다.
    - 회피 표현은 문장 전체에서 찾으므로, 다른 대상에 대한 '부담' 등이 있어도 회피로 잡힌다.
    """
    if re.search(terms["회피"], sent):
        return "회피", grade, ""
    after = terms["회피_비교"].replace("{G}", "")  # 언급 표현 바로 뒤에 붙는 부분
    for g, _, s, e in mentions:
        if g == grade and re.match(after, sent[e:]):
            return "회피", grade, ""
    has_exp = bool(re.search(terms["경험"], sent))
    has_req = bool(re.search(terms["요청"], sent))
    numeric = any(g == grade and f == "숫자" for g, f, _, _ in mentions)
    if has_exp and has_req:
        if numeric:
            return "R1 명확한 희망", grade, ""
        if re.search(terms["고려"], sent):
            return "R2 고려 중", grade, ""
        return "사람 확인", grade, "경험·요청 동시, 명칭만"
    if has_exp:
        return "경험", grade, ""
    if grade == 3 and re.search(terms["비교"], sent):
        return "R4 비교 표현", 4, ""
    if numeric and has_req:
        return "R1 명확한 희망", grade, ""
    if re.search(terms["고려"], sent):
        return "R2 고려 중", grade, ""
    why = "숫자 등급·요청 표현 없음" if numeric else ("명칭·요청 표현만" if has_req else "명칭·관련 표현 없음")
    return "사람 확인", grade, why


CLASS_RANK = ["R1 명확한 희망", "R2 고려 중", "R4 비교 표현", "사람 확인", "경험", "회피"]


def question_mentions(question: str, terms: dict) -> pd.DataFrame:
    """question에서 언급 등급별 분류. 반환: 언급 등급, 해석 등급, 분류, 사람 확인 사유, 문장 수 (등급당 한 행).

    애매한 경우 처리: 같은 등급이 여러 문장에 나오면 CLASS_RANK 순서로 가장 앞선 분류를 쓴다
    (희망 진술이 하나라도 있으면 희망으로 본다. 02b L1과 같은 방향).
    """
    rows = []
    for sent in _SENT_SPLIT.split(_mask_ranges(str(question))):
        ms = sentence_mentions(sent)
        for g in sorted({m[0] for m in ms}):
            cls, g2, why = classify_sentence(sent, g, ms, terms)
            rows.append({"언급 등급": g, "해석 등급": g2, "분류": cls, "사람 확인 사유": why})
    if not rows:
        return pd.DataFrame(columns=["언급 등급", "해석 등급", "분류", "사람 확인 사유", "문장 수"])
    d = pd.DataFrame(rows)
    d["rank"] = d["분류"].map(CLASS_RANK.index)
    best = d.sort_values("rank").groupby("언급 등급", as_index=False).first()
    best["문장 수"] = best["언급 등급"].map(d["언급 등급"].value_counts())
    return best.drop(columns="rank")


def transfer_expressions(texts: list, terms: dict) -> tuple:
    """cot1~answer에서 이관 표현과 일반 상담 문구를 찾는다 (D56).

    반환: (이관 표현 있음, 일반 상담 문구 있음)
    - 이관: 한 문장 안에 '이관_고지'(범위 초과 고지)와 '이관_절차'(상담·확인 절차)가 함께 있음 (D40).
    - 일반 상담 문구: 이관_절차 표현은 있지만 같은 문장에 고지가 없음 (예: '필요 시 전문가 상담'). 이관으로 보지 않는다.
    """
    transfer = generic = False
    for t in texts:
        for s in _SENT_SPLIT.split(str(t)):
            proc = re.search(terms["이관_절차"], s)
            if proc and re.search(terms["이관_고지"], s):
                transfer = True
            elif proc:
                generic = True
    return transfer, generic
