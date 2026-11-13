"""교정 함수 (D38·D46). 데이터프레임을 받아 교정 결과를 돌려준다.

- 원본은 바꾸지 않는다. 교정 결과는 새 컬럼으로 돌려준다.
- 교정 대상 선정(D45·D46 규칙 ①②)은 노트북에서 한다. 이 함수는 넘겨받은 CoT의 cot2 범위 표기만 고친다.
"""
from __future__ import annotations

import re

import pandas as pd

from src.extract import parse_allowed_grades

# 공개 기준(D43)과, D30에서 확인한 '완화된' 범위
PUBLIC_RANGE = {"위험중립형": frozenset({4, 5, 6}), "안정추구형": frozenset({5, 6})}
WIDE_RANGE = {"위험중립형": frozenset({3, 4, 5, 6}), "안정추구형": frozenset({4, 5, 6})}

_RANGE_CONNECTOR = re.compile(r"~|-|부터|에서")


def _text_set(text: str) -> frozenset:
    """범위 표기 하나가 가리키는 등급 집합. parse_allowed_grades가 cot2 전체에서 뽑아 준 표기만 넣는다.

    표기 하나만 따로 다시 읽으면 '3·4·5·6'처럼 '등급'이 빠진 표기를 놓치므로, 숫자로 직접 계산한다.
    """
    nums = [int(d) for d in re.findall(r"[1-6]", text)]
    if _RANGE_CONNECTOR.search(text) and len(nums) >= 2:
        return frozenset(range(min(nums[0], nums[1]), max(nums[0], nums[1]) + 1))
    return frozenset(nums)


def _fix_text(text: str, wide: frozenset, public: frozenset) -> str:
    """범위 표기 하나를 공개 기준으로 바꾼다. 예) '3~6등급'→'4~6등급', '3, 4, 5, 6등급'→'4, 5, 6등급'."""
    drop = min(wide)          # 넓은 범위에서 빠져야 하는 등급 (위험중립형 3, 안정추구형 4)
    keep_min = min(public)
    if _RANGE_CONNECTOR.search(text):
        # 범위형: 맨 앞 숫자(시작 등급)만 바꾼다
        return re.sub(rf"^(\s*){drop}", rf"\g<1>{keep_min}", text, count=1)
    # 목록형: 맨 앞 원소와 그 뒤 구분자를 지운다
    return re.sub(rf"^(\s*){drop}\s*(?:등급)?\s*(?:[,·ㆍ/]|및|와|과|또는)\s*·?\s*", r"\g<1>", text, count=1)


def correct_allowed_range(df: pd.DataFrame, propensity_col: str = "성향_cot2") -> pd.DataFrame:
    """cot2의 허용 범위 표기를 공개 기준으로 고친다 (D46 규칙 ①: cot2 안의 범위 진술은 모두 교정).

    입력: cot_id, cot2, propensity_col 컬럼이 있는 데이터프레임 (교정 대상만 넘긴다)
    반환: cot_id, 원래 범위, 교정 범위, 교정 표기 수, cot2_교정, 상태
      상태: '교정' / '교정 불가: <이유>' (교정 불가면 cot2_교정은 원문 그대로)

    애매한 경우 처리:
    - 범위 표기 중 집합이 '넓은 범위'와 정확히 같은 것만 고친다. 이미 공개 기준인 표기(예: '5, 6등급')는 그대로 둔다.
    - 넓은 범위에서 빠지는 등급을 포함하지만 넓은 범위와 다른 표기(예: 위험중립형 '3~4등급')가 있으면
      다른 의미(상품군 지칭 등)일 수 있어 고치지 않고 '교정 불가'로 둔다.
    - 같은 표기가 cot2에 여러 번 나오면 모두 바꾼다.
    - '보통 위험(3, 4, 5, 6등급)'처럼 등급 명칭이 붙은 경우 숫자만 고치고 명칭은 그대로 둔다.
    - 교정 뒤 parse_allowed_grades로 다시 읽어 공개 기준과 같지 않으면 '교정 불가'로 되돌린다.
    """
    rows = []
    for _, r in df.iterrows():
        p = r[propensity_col]
        t = str(r["cot2"])
        wide, public = WIDE_RANGE.get(p), PUBLIC_RANGE.get(p)
        before, _, texts = parse_allowed_grades(t, with_text=True)
        out = {"cot_id": r["cot_id"], "원래 범위": before, "교정 범위": public, "교정 표기 수": 0,
               "cot2_교정": t, "상태": ""}
        if wide is None:
            out["상태"] = "교정 불가: 대상 성향 아님"
            rows.append(out)
            continue
        drop = min(wide)
        # 긴 표기부터 바꿔야 '4, 5, 6등급'과 '4, 5, 6'이 함께 있을 때 서로 겹치지 않는다
        sets = [(x, _text_set(x)) for x in sorted(dict.fromkeys(texts), key=len, reverse=True)]
        odd = [x for x, s in sets if s and drop in s and s != wide]
        if odd:
            out["상태"] = f"교정 불가: 넓은 범위와 다른 표기 {odd}"
            rows.append(out)
            continue
        new, n = t, 0
        for x, s in sets:
            if s == wide:
                fixed = _fix_text(x, wide, public)
                n += new.count(x)
                new = new.replace(x, fixed)
        after = parse_allowed_grades(new)[0]
        if n == 0:
            out["상태"] = "교정 불가: 넓은 범위 표기를 못 찾음"
        elif after != public:
            out["상태"] = f"교정 불가: 교정 뒤 범위가 {sorted(after) if after else None}"
        else:
            out.update({"교정 표기 수": n, "cot2_교정": new, "상태": "교정"})
        rows.append(out)
    return pd.DataFrame(rows)
