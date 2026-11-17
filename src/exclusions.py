"""엄격본 제외 목록을 만든다 (D30·D47·D52·D55).데이터프레임을 받아 제외 목록 데이터프레임을 돌려준다.

원문 문장은 넣지 않는다. 사유는 분류 코드·판단 값 같은 짧은 라벨만 쓴다.
"""
from __future__ import annotations

import pandas as pd

# D47 '내 분류' 코드 정의
CLASS_CODE = {
    "1": "고객 요청·경험을 이유로 범위 밖 추천",
    "2": "예외 설명 없이 범위 밖 추천",
    "3": "성향 판단 오류",
    "4": "성향과 허용 범위 짝 불일치",
}

D52_REASON = "원천 등급 과소 기재(베트남 레버리지)"
D30_REASON = "넓은 허용 범위의 완화 등급 실제 추천(안정추구형 4등급·위험중립형 3등급)"

# 공개 기준 (D43)
D43_RANGE = {"공격투자형": {1, 2, 3, 4, 5, 6}, "적극투자형": {2, 3, 4, 5, 6}, "위험중립형": {4, 5, 6},
             "안정추구형": {5, 6}, "안정형": {6}}
D55_NOTE = {6941: "D33"}  # D55로 판단 보류를 제외로 갱신한 CoT → 근거 결정에 함께 적는다


def d55_code(cot2_propensity: str, allowed: str) -> str:
    """D55 사유 코드 하나를 조건으로 정한다.

    - cot2_propensity: cot2의 성향 (propensity_in 결과: 성향명 / '복수' / '없음')
    - allowed: 파싱한 허용 등급 문자열 (예: '4-5-6', '추출 실패')
    순서 (처음 맞는 것 하나)
    1. cot2 성향 '복수' → 코드 3 (성향 판단 오류)
    2. cot2 성향 '없음' → '성향 누락'
    3. 파싱 허용 범위가 그 성향의 D43 범위와 다름 → 코드 4 (성향과 허용 범위 짝 불일치)
    4. 허용 범위 추출 실패, 또는 그 외(범위는 D43과 같은데 범위 밖 추천) → 코드 2 (예외 설명 없이 범위 밖 추천)
    애매한 경우 처리: 허용 범위가 '추출 실패'면 짝을 비교할 수 없어 3번을 건너뛰고 코드 2로 둔다.
    """
    if cot2_propensity == "복수":
        return "3"
    if cot2_propensity == "없음":
        return "성향 누락"
    try:
        grades = {int(x) for x in str(allowed).split("-")}
    except ValueError:
        return "2"
    if cot2_propensity in D43_RANGE and grades != D43_RANGE[cot2_propensity]:
        return "4"
    return "2"


def d55_exclusions(out_cots: pd.DataFrame, already: set) -> pd.DataFrame:
    """D55: 공개 기준(D43) 밖 추천이 있는 CoT 중 아직 제외 목록에 없는 것에 사유 코드를 자동으로 붙인다.

    입력
    - out_cots: 공개 기준 밖 추천이 있는 CoT (cot_id, 성향_cot2, 허용등급), CoT당 한 행
    - already: 이미 제외 목록에 있는 cot_id (범위밖_분류표·D52의 사람 판정, D30)
    반환: cot_id, 사유 코드, 제외 사유, 근거 결정, 출처

    애매한 경우 처리: 사람 판정이 이미 있는 CoT(범위밖_분류표, D52)와 D30 제외 CoT는 건드리지 않는다 (사람 판정 우선).
    """
    x = out_cots[~out_cots["cot_id"].isin(already)].drop_duplicates("cot_id").copy()
    x["사유 코드"] = [d55_code(p, a) for p, a in zip(x["성향_cot2"], x["허용등급"])]
    x["제외 사유"] = x["사유 코드"].map(lambda c: CLASS_CODE.get(c, c)) + " (D55 자동 부여)"
    x["근거 결정"] = x["cot_id"].map(lambda c: ", ".join(["D55"] + ([D55_NOTE[c]] if c in D55_NOTE else [])))
    x["출처"] = "D55"
    return x[["cot_id", "사유 코드", "제외 사유", "근거 결정", "출처"]].sort_values("cot_id").reset_index(drop=True)


def build_exclusions(review: pd.DataFrame, d52_ids: list, split_of: pd.Series, d30_ids: list = (),
                     d55: pd.DataFrame | None = None) -> pd.DataFrame:
    """범위밖_분류표, D52 목록, D30 목록, D55 자동 부여 목록으로 엄격본 제외 목록을 만든다.

    입력
    - review: 범위밖_분류표 (cot_id, 판단, 처리, 내 분류)
    - d52_ids: D52 베트남 레버리지 추천 CoT의 cot_id
    - d30_ids: D30 제외 CoT의 cot_id (교정 전 원본 cot2 기준 넓은 범위 + 완화 등급 실제 추천, 308건)
    - d55: d55_exclusions 결과 (없으면 넣지 않음)
    - split_of: cot_id → 분할(Training/Validation)
    반환: cot_id, 분할, 제외 사유, 근거 결정, 출처 (cot_id 순)

    애매한 경우 처리
    - 처리 칸이 '제외'이거나 '엄격본 제외'로 시작하면 엄격본 제외로 본다 (2017: '엄격본 제외, 교정본 교정').
      '교정 + 태그'(194)처럼 제외가 아닌 처리는 넣지 않는다.
    - 사유: 내 분류가 숫자 코드면 D47 정의로 바꾸고, 글자 값이면 그대로 쓴다. 판단 값을 괄호로 붙인다.
    - 6799는 D44(원천 오류), 2017은 D46(범위 표기)을 근거 결정에 함께 적는다.
    - 같은 cot_id가 여러 출처에 있으면(예: 6429) 한 행으로 합치고 사유는 ' | ', 근거·출처는 '; '로 잇는다.
      (6799의 내 분류 값 '원천 오류 + 부적합 정당화'에 ' + '가 들어 있어 사유 구분자로 쓰지 않는다)
    """
    rows = []
    r = review.copy()
    r["cot_id"] = r["cot_id"].astype(int)
    strict = r["처리"].eq("제외") | r["처리"].str.startswith("엄격본 제외")
    for _, x in r[strict].iterrows():
        cls = str(x["내 분류"]).strip()
        reason = CLASS_CODE.get(cls, cls)
        if str(x["판단"]).strip():
            reason += f" (판단: {x['판단']})"
        if x["처리"] != "제외":
            reason += f" [{x['처리']}]"
        basis = ["D47"]
        if x["cot_id"] == 6799:
            basis.append("D44")
        if x["cot_id"] == 2017:
            basis.append("D46")
        rows.append({"cot_id": x["cot_id"], "제외 사유": reason, "근거 결정": ", ".join(basis), "출처": "범위밖_분류표"})
    for c in d52_ids:
        rows.append({"cot_id": int(c), "제외 사유": D52_REASON, "근거 결정": "D52", "출처": "D52"})
    for c in d30_ids:
        rows.append({"cot_id": int(c), "제외 사유": D30_REASON, "근거 결정": "D30", "출처": "D30"})
    if d55 is not None:
        rows += d55[["cot_id", "제외 사유", "근거 결정", "출처"]].to_dict("records")

    df = pd.DataFrame(rows)
    out = (df.groupby("cot_id", sort=True)
             .agg({"제외 사유": " | ".join, "근거 결정": "; ".join, "출처": "; ".join})
             .reset_index())
    out.insert(1, "분할", out["cot_id"].map(split_of))
    return out
