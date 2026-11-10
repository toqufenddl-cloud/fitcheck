"""CoT 텍스트에서 사실 확인에 필요한 값을 뽑아내는 함수 모음.

- 판정(좋다/나쁘다)은 하지 않는다. "무엇이 적혀 있는가"만 꺼낸다.
- 함수마다 '애매한 경우 처리'를 주석으로 적는다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 공통: 문자열 정규화
# ---------------------------------------------------------------------------

def norm_full(s) -> str:
    """공백, 괄호류, 하이픈, 밑줄, 따옴표, 가운뎃점, 쉼표, 마침표를 지운다 (상품명 비교용)."""
    return re.sub(r"[\s\(\)\[\]\{\}\-_'\"·,.]", "", str(s))


# ---------------------------------------------------------------------------
# 상품명 → 상품 JSON 매칭 (D10 확정 규칙)
# ---------------------------------------------------------------------------

def grade_num(risk_grade) -> float:
    """상품 JSON의 risk_grade('2등급(높은위험)')에서 숫자만 꺼낸다. 없으면 NaN."""
    m = re.match(r"\s*([1-6])\s*등급", str(risk_grade))
    return float(m.group(1)) if m else np.nan


def build_product_index(prod: pd.DataFrame) -> dict:
    """D10 매칭에 쓸 이름 → 상품 키(file_name) 사전을 만든다.

    - 증권: product_name, 보험: product_full_name 을 '대표 이름'으로 쓴다.
    - 같은 이름이 여러 상품에 있을 수 있으므로 값은 상품 키의 리스트다.
    """
    p = prod.copy()
    p["display_name"] = np.where(p["domain"] == "증권", p["product_name"], p["product_full_name"])
    p["grade"] = p["risk_grade"].map(grade_num) if "risk_grade" in p else np.nan

    def to_index(names: pd.Series) -> dict:
        d = {}
        for key, name in zip(p["file_name"], names):
            if pd.notna(name):
                d.setdefault(name, []).append(key)
        return d

    ins = p["domain"] == "보험"
    return {
        "table": p.set_index("file_name"),
        "exact": to_index(p["display_name"]),
        "norm": to_index(p["display_name"].map(norm_full)),
        "mother": to_index(p["mother_product_name"].where(ins)),
        "norm_set": set(p["display_name"].map(norm_full)),
    }


def _unmatched_cause(name: str, idx: dict) -> str:
    """매칭되지 않은 이름에 원인 코드를 붙인다.

    애매한 경우 처리: 두 원인에 모두 해당하면 '본인→고객 치환 의심'을 우선한다.
    """
    if norm_full(str(name).replace("고객", "본인")) in idx["norm_set"]:
        return "판정불가:본인→고객 치환 의심"
    if len(norm_full(name)) <= 10 or str(name).strip().startswith("["):
        return "판정불가:상품명 조각"
    return "판정불가:기타"


def match_product(name: str, idx: dict) -> tuple[str, list]:
    """상품명 하나를 D10 순서(정확 → 정규화 → 보험 모상품명)로 매칭한다.

    반환: (매칭 단계, 상품 키 리스트). 실패하면 (원인 코드, []).
    애매한 경우 처리: 한 단계에서 후보가 여러 개면(이름 중복) 모두 돌려준다.
    등급이 후보마다 다를 수 있으므로, 등급을 쓸 때는 product_grade()로 확인한다.
    """
    if name in idx["exact"]:
        return "1 정확 일치", idx["exact"][name]
    n = norm_full(name)
    if n in idx["norm"]:
        return "2 정규화 일치", idx["norm"][n]
    if name in idx["mother"]:
        return "3 보험 모상품명", idx["mother"][name]
    return _unmatched_cause(name, idx), []


def product_grade(keys: list, idx: dict) -> float:
    """상품 키 리스트의 위험등급을 돌려준다.

    애매한 경우 처리:
    - 키가 없거나 등급 정보가 없으면(보험 상품 등) NaN.
    - 후보 상품끼리 등급이 다르면 하나를 고르지 않고 NaN을 돌려준다 (등급 복수).
    """
    if not keys:
        return np.nan
    g = idx["table"].loc[keys, "grade"].dropna().unique()
    return float(g[0]) if len(g) == 1 else np.nan


# ---------------------------------------------------------------------------
# 위험등급 표현 추출
# ---------------------------------------------------------------------------

# 범위·목록 표현 (단일 등급으로 보지 않는다)
_RANGE_PATTERNS = [
    r"[1-6]\s*(?:등급)?\s*[~∼〜\-–]\s*[1-6]\s*등급",                         # 2~6등급, 2등급~6등급
    # 2·3·4등급, 2등급, 3등급 (목록의 마지막 원소는 반드시 '등급'으로 끝나야 한다.
    # 그래야 '6등급, 1년 기준'의 '1'을 목록으로 잘못 읽지 않는다)
    r"[1-6]\s*(?:등급)?(?:\s*(?:[·ㆍ,/]|및|와|과|또는)\s*[1-6]\s*(?:등급)?)*\s*(?:[·ㆍ,/]|및|와|과|또는)\s*[1-6]\s*등급",
    r"[1-6]\s*등급\s*(?:이상|이하|까지|부터|초과|미만|이내)",                  # 2등급 이상
]
_GRADE_ANY = re.compile(r"[1-6]\s*등급")


def _mask_ranges(text: str) -> str:
    """범위·목록 표현을 '#'로 덮어서 단일 등급 검색에서 빠지게 한다."""
    for p in _RANGE_PATTERNS:
        text = re.sub(p, lambda m: "#" * len(m.group(0)), text)
    return text


def parse_allowed_grades(cot2: str):
    """cot2에서 '허용 위험등급' 집합을 뽑는다 (범위·목록 표현을 펼친다).

    방법: '허용·범위·투자할 수·가능·포함' 중 하나가 들어 있는 문장에서 등급 표현을 모은다.
    - 'N~M등급' → N..M, 'N·M·K등급' / 'N등급, M등급' → {N, M, K}, 단일 'N등급' → {N}
    - 'N등급 이상/이하'는 각각 N..6 / 1..N 으로 펼친다.
    애매한 경우 처리:
    - 키워드 문장이 없으면 cot2 전체에서 모은다 (source='전체').
    - 등급 표현이 하나도 없으면 None (추출 실패).
    - 서로 다른 범위가 여러 번 나오면 합집합으로 둔다 (어느 것이 '최종'인지 판단하지 않는다).
    """
    sents = re.split(r"(?<=[.!?])\s+|\n", str(cot2))
    key = [s for s in sents if re.search(r"허용|범위|투자할 수|가능|포함", s)]
    source = "키워드 문장" if key and any(_GRADE_ANY.search(s) for s in key) else "전체"
    text = " ".join(key) if source == "키워드 문장" else str(cot2)

    grades = set()
    for a, b in re.findall(r"([1-6])\s*(?:등급)?\s*[~∼〜\-–]\s*([1-6])\s*등급", text):
        grades |= set(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
    for n, op in re.findall(r"([1-6])\s*등급\s*(이상|이하)", text):
        grades |= set(range(int(n), 7)) if op == "이상" else set(range(1, int(n) + 1))
    # 나머지 숫자 등급 (목록·단일)
    rest = re.sub(r"[1-6]\s*(?:등급)?\s*[~∼〜\-–]\s*[1-6]\s*등급|[1-6]\s*등급\s*(?:이상|이하)", " ", text)
    for m in re.finditer(r"[1-6](?:\s*(?:등급)?\s*(?:[·ㆍ,/]|및|와|과|또는)\s*[1-6])*\s*등급", rest):
        grades |= {int(d) for d in re.findall(r"[1-6]", m.group(0))}
    return (frozenset(grades), source) if grades else (None, source)


_CHAIN_GAP = 8  # 상품명 사이 간격이 이 글자 수 이하면 '함께 나열된 상품'으로 본다 (예: "A"와 "B"는)


def _find_all(text: str, name: str) -> list:
    """상품명이 나온 모든 위치 (시작, 끝). 정확 일치가 없으면 공백을 무시해 다시 찾는다."""
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(name), text)]
    if not spans:
        pat = r"\s*".join(map(re.escape, re.sub(r"\s+", "", name)))
        spans = [(m.start(), m.end()) for m in re.finditer(pat, text)] if pat else []
    return spans


def grade_after_name(text: str, name: str, other_names: list) -> tuple[str, float]:
    """text(cot3 등)에서 상품명 뒤에 적힌 단일 위험등급을 뽑는다.

    구간 정하기:
    - 상품명이 나온 위치의 끝에서 시작한다.
    - 바로 뒤(8자 이내)에 목록의 다른 상품명이 이어지면 함께 나열된 것으로 보고 건너뛴다.
      예: "A"와 "B"는 모두 4등급 → A의 구간은 B 뒤부터 이어진다.
    - 그다음 나오는 목록의 다른 상품명 앞(없으면 텍스트 끝)에서 끝난다.
    - 상품명이 여러 번 나오면 등장 순서대로 보고, 처음으로 등급이 추출되는 구간을 쓴다.
    반환: (상태, 등급)
    - '추출': 구간에서 범위 표현을 제외한 첫 단일 'N등급'
    - '상품명 없음': 텍스트에 상품명이 없음 (정확 일치, 실패 시 공백 무시 비교)
    - '등급 표현 없음': 어느 구간에도 'N등급'이 전혀 없음
    - '있지만 추출 실패': 'N등급'은 있으나 모두 범위·목록 표현
    애매한 경우 처리:
    - 범위 표현(N~M, N·M, N 이상/이하)은 상품의 등급이 아니라 허용 범위일 수 있어 제외한다.
    - 구간 안의 첫 단일 등급을 그 상품의 등급으로 본다. 다른 맥락의 등급일 가능성은 남는다.
    - 함께 나열된 상품은 같은 등급 표현을 공유한다 ("모두 4등급"인지 "각각"인지 구분하지 않는다).
    """
    text = str(text)
    spans = _find_all(text, name)
    if not spans:
        return "상품명 없음", np.nan

    others = [s for o in other_names if o != name for s in _find_all(text, o)]
    saw_grade_word = False
    for _, end in spans:
        # 바로 이어서 나열된 다른 상품명 건너뛰기
        moved = True
        while moved:
            moved = False
            for s, e in others:
                if end <= s <= end + _CHAIN_GAP and e > end:
                    end, moved = e, True
        nxt = min([s for s, _ in others if s >= end] + [len(text)])
        seg = text[end:nxt]
        if not _GRADE_ANY.search(seg):
            continue
        saw_grade_word = True
        m = _GRADE_ANY.search(_mask_ranges(seg))
        if m:
            return "추출", float(re.match(r"([1-6])", m.group(0)).group(1))
    return ("있지만 추출 실패" if saw_grade_word else "등급 표현 없음"), np.nan


# ---------------------------------------------------------------------------
# 투자성향, 연령, 성별
# ---------------------------------------------------------------------------

PROPENSITY = ["공격투자형", "적극투자형", "위험중립형", "안정추구형", "안정형"]


def propensity_in(text) -> str:
    """텍스트에 나온 투자성향 명칭을 돌려준다.

    애매한 경우 처리: 0개면 '없음', 2개 이상이면 '복수'. (어느 것이 고객 성향인지 판단하지 않는다)
    """
    found = [p for p in PROPENSITY if p in str(text)]
    return found[0] if len(found) == 1 else ("없음" if not found else "복수")


def extract_age_decade(text):
    """질문 본문에서 연령대를 뽑는다 (첫 번째 표현 기준).

    인식: 'N0대'(초반/중반/후반 포함), 'NN세', 'NN살'. 'NN세'는 십의 자리로 바꾼다.
    애매한 경우 처리: 자녀·부모 등 다른 사람의 나이일 수 있으나 구분하지 않는다. 없으면 None.
    """
    m = re.search(r"([1-9])0\s*대|(\d{2})\s*(?:세|살)", str(text))
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else int(m.group(2)) // 10


def extract_gender(text):
    """질문 본문에서 성별 표현을 뽑는다 ('남성/남자' → 남, '여성/여자' → 여).

    애매한 경우 처리: '남자친구', '여자아이'처럼 다른 사람을 가리킬 수 있으나 구분하지 않는다.
    둘 다 나오면 '복수', 없으면 None.
    """
    t = str(text)
    male = bool(re.search(r"남성|남자", t))
    female = bool(re.search(r"여성|여자", t))
    if male and female:
        return "복수"
    return "남" if male else ("여" if female else None)


def meta_age_match(meta_age: str, decade) -> str:
    """메타 age와 본문 연령대를 비교한다.

    애매한 경우 처리: 메타가 'N0대 이상'이면 본문 연령대가 N 이상일 때 일치로 본다.
    """
    if decade is None:
        return "본문에 없음"
    m = re.match(r"([1-9])0대(\s*이상)?", str(meta_age))
    if not m:
        return "메타 형식 불명"
    n, plus = int(m.group(1)), bool(m.group(2))
    ok = decade >= n if plus else decade == n
    return "일치" if ok else "불일치"


# ---------------------------------------------------------------------------
# answer 관련
# ---------------------------------------------------------------------------

def quoted_strings(text) -> list:
    """큰따옴표(" " 또는 “ ”) 안의 문자열을 모두 뽑는다.

    애매한 경우 처리: 따옴표 안이 상품명이 아닐 수도 있다 (강조 표현 등). 걸러내지 않는다.
    """
    return [s.strip() for s in re.findall(r"[\"“]([^\"“”]+)[\"”]", str(text))]


_END_OK = set(".!?…。")
_CLOSERS = "\"'”’)]}」』"


def ends_with_punct(text) -> bool:
    """문장 부호로 끝나는지 확인한다.

    애매한 경우 처리: 끝의 공백과 닫는 따옴표·괄호는 건너뛰고 그 앞 글자를 본다.
    """
    t = str(text).rstrip()
    while t and t[-1] in _CLOSERS:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] in _END_OK


# ---------------------------------------------------------------------------
# 유사 중복 묶음
# ---------------------------------------------------------------------------

def near_duplicate_groups(texts: list, threshold: float = 0.9, block: int = 500) -> np.ndarray:
    """글자 n-그램 TF-IDF 코사인 유사도가 threshold 이상인 문서를 묶는다.

    - 메모리를 아끼려고 block행씩 나눠 유사도를 계산한다.
    - 유사한 쌍을 연결해 묶음(연결 요소)을 만든다. A~B, B~C이면 A·B·C가 한 묶음이다.
    반환: 문서마다 묶음 번호 (혼자인 문서도 고유 번호를 가진다).
    애매한 경우 처리: 연결로 묶기 때문에 A와 C의 직접 유사도는 0.9 미만일 수 있다.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit_transform(texts)  # 행 단위 L2 정규화됨
    n = X.shape[0]
    rows, cols = [], []
    for s in range(0, n, block):
        sim = (X[s:s + block] @ X.T).tocoo()
        keep = (sim.data >= threshold) & (sim.row + s < sim.col)   # 자기 자신·중복 쌍 제외
        rows.append(sim.row[keep] + s)
        cols.append(sim.col[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels
