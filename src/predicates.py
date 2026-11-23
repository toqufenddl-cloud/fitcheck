"""규칙 표(src/rules/rules.csv)의 '조건' 칸이 가리키는 함수들.

모든 함수는 (ctx, **params)를 받아 (mask, detail)을 돌려준다.
- mask: 증권 CoT 전체(cot_id 색인)에 대한 bool Series
- detail: 사유 코드 등 CoT별 값이 필요한 경우의 Series, 없으면 None
ctx는 judge.prepare()가 만든다. 애매한 경우 처리는 각 함수 주석에 적는다.
"""
from __future__ import annotations

import pandas as pd

REGISTRY = {}


def predicate(fn):
    REGISTRY[fn.__name__] = fn
    return fn


def _ids(ctx):
    return ctx["cots"].index


def _from_ids(ctx, ids):
    return pd.Series(_ids(ctx).isin(set(ids)), index=_ids(ctx))


@predicate
def always(ctx):
    return pd.Series(True, index=_ids(ctx)), None


@predicate
def human_review(ctx, 처리):
    """범위밖_분류표의 사람 판정. 처리 칸이 값 중 하나와 같거나 그 값으로 시작하면 해당 ('엄격본 제외, 교정본 교정' 포함).
    detail: 내 분류 (숫자 코드 또는 글자 값)."""
    r = ctx["review"]
    keep = r["처리"].map(lambda v: any(v == p or v.startswith(p) for p in 처리.split("|")))
    sub = r[keep]
    detail = pd.Series(sub["내 분류"].values, index=sub["cot_id"].astype(int)).reindex(_ids(ctx))
    return _from_ids(ctx, sub["cot_id"].astype(int)), detail


@predicate
def recommends(ctx, 상품):
    """추천 상품명에 문자열이 들어 있는 CoT (클래스 무관, D52)."""
    rec = ctx["recs"]
    return _from_ids(ctx, rec.loc[rec["rec_name"].str.contains(상품, regex=False), "cot_id"]), None


@predicate
def wide_range_relaxed_grade(ctx):
    """D30: 교정 전 원본 cot2가 넓은 범위이고 완화 등급을 실제 추천.
    안정추구형(cot2) + 허용 4-5-6 + 4등급 추천, 위험중립형(cot2) + 허용 3-4-5-6 + 3등급 추천.
    애매한 경우 처리: 성향은 D30과 같게 cot2 성향을 쓴다(D21 아님). 등급 없는 추천은 보지 않는다."""
    m = ctx["recs"].merge(ctx["cots"][["성향_cot2", "허용등급"]], left_on="cot_id", right_index=True)
    hit = (((m["성향_cot2"] == "안정추구형") & (m["허용등급"] == "4-5-6") & (m["json_grade"] == 4)) |
           ((m["성향_cot2"] == "위험중립형") & (m["허용등급"] == "3-4-5-6") & (m["json_grade"] == 3)))
    return _from_ids(ctx, m.loc[hit, "cot_id"]), None


@predicate
def wide_range_only(ctx):
    """D30 태그: 넓은 범위 표기(위 조합)인데 완화 등급 추천은 없는 CoT.
    애매한 경우 처리: 추천 목록이 비어 있으면 '추천이 모두 엄격 범위 안'이라고 볼 수 없으므로 넣지 않는다 (2026-11-17 수정, 5890)."""
    c = ctx["cots"]
    wide = (((c["성향_cot2"] == "안정추구형") & (c["허용등급"] == "4-5-6")) |
            ((c["성향_cot2"] == "위험중립형") & (c["허용등급"] == "3-4-5-6")))
    relaxed, _ = wide_range_relaxed_grade(ctx)
    has_rec, _ = no_rec(ctx)
    return wide & ~relaxed & ~has_rec, None


@predicate
def no_rec(ctx):
    """추천 목록(product_names)이 비어 있는 CoT (D34: 문자열 깨짐 4건 등). 범위 안으로 통과시키지 않는다."""
    return ~_ids(ctx).to_series(index=_ids(ctx)).isin(set(ctx["recs"]["cot_id"])), None


@predicate
def out_of_range(ctx):
    """D43(as_of 기준표) 밖 추천이 하나라도 있는 CoT. 성향은 D21.
    애매한 경우 처리: 등급 없는 추천(매칭 실패·후보 등급 다름)과 성향 '비교 불가' CoT는 보지 않는다."""
    return _from_ids(ctx, ctx["recs"].loc[ctx["recs"]["밖"], "cot_id"]), None


@predicate
def cot2_propensity(ctx, 값):
    return ctx["cots"]["성향_cot2"].isin(값.split("|")), None


@predicate
def range_mismatch(ctx):
    """cot2 성향(단일)과 파싱 허용 범위가 기준표와 다름 (D55 코드 4).
    애매한 경우 처리: 허용 범위 '추출 실패'는 비교하지 않는다(→ 코드 2)."""
    c, rng = ctx["cots"], ctx["range"]

    def mis(p, a):
        if p not in rng or a == "추출 실패":
            return False
        return {int(x) for x in a.split("-")} != rng[p]
    return pd.Series([mis(p, a) for p, a in zip(c["성향_cot2"], c["허용등급"])], index=c.index), None


@predicate
def no_propensity(ctx):
    return ctx["cots"]["성향_D21"] == "비교 불가", None


@predicate
def no_graded_rec(ctx):
    return ~_ids(ctx).to_series(index=_ids(ctx)).isin(set(ctx["recs"].loc[ctx["recs"]["json_grade"].notna(), "cot_id"])), None


@predicate
def d19_level(ctx, 값):
    return ctx["cots"]["D19"] == 값, None


@predicate
def recommends_mismatch_product(ctx):
    """D51: 등록 등급과 설명 속 등급(body_grade)이 다른 상품을 추천한 CoT. 상품 키 기준."""
    rec = ctx["recs"]
    return _from_ids(ctx, rec.loc[rec["keys"].map(lambda ks: any(k in ctx["mismatch_keys"] for k in ks)), "cot_id"]), None


@predicate
def recommends_multi_grade_product(ctx):
    """설명 속 등급이 둘 이상(운용전환 전후 등)인 상품을 추천한 CoT. 비교하지 않고 따로 표시만 한다."""
    rec = ctx["recs"]
    return _from_ids(ctx, rec.loc[rec["keys"].map(lambda ks: any(k in ctx["multi_keys"] for k in ks)), "cot_id"]), None


@predicate
def transfer_class(ctx, 값):
    return ctx["cots"]["이관 분류"].isin(값.split("|")), None


@predicate
def transfer_outcome(ctx, 값):
    return ctx["cots"]["이관 결과"].isin(값.split("|")), None


@predicate
def in_corrections_log(ctx):
    return _from_ids(ctx, ctx["corrections"]["cot_id"].astype(int)), None


@predicate
def in_tags_file(ctx):
    t = ctx["tags"]
    detail = t.groupby(t["cot_id"].astype(int))["태그"].agg(", ".join).reindex(_ids(ctx))
    return _from_ids(ctx, t["cot_id"].astype(int)), detail


@predicate
def out_of_cot2_range(ctx):
    """D57: 추천 상품 등급이 cot2에 적힌(파싱한) 허용 범위 밖인 CoT. 교정 전 원본 cot2 기준.
    애매한 경우 처리: 허용 범위 '추출 실패'와 등급 없는 추천은 보지 않는다."""
    c, rec = ctx["cots"], ctx["recs"]
    allowed = c["허용등급"].map(lambda a: None if a == "추출 실패" else {int(x) for x in a.split("-")})
    r = rec[rec["json_grade"].notna()]
    hit = [cid for cid, g in zip(r["cot_id"], r["json_grade"])
           if allowed.get(cid) is not None and int(g) not in allowed[cid]]
    return _from_ids(ctx, hit), None


@predicate
def range_text_damaged(ctx):
    """D57: 허용 범위 문장이 깨져 파싱 결과를 믿을 수 없는 CoT. 예) '허용 위험등급은 3. 5, 6등급'(194, D20 원문 깨짐).
    패턴: 'N. M, K'처럼 숫자 뒤 마침표 다음에 등급 목록이 이어짐.
    애매한 경우 처리: 이 패턴 밖의 손상은 잡지 못한다 (2026-11-18 기준 해당 1건)."""
    return ctx["cots"]["cot2"].str.contains(r"[1-6]\s*\.\s+(?:/\s*)?[1-6]\s*[,·]\s*[1-6]", regex=True), None


@predicate
def c2_class(ctx, 값):
    """기준 ② 분류 (src/criteria12.judge12). 값의 '|'는 '또는'."""
    return ctx["cots"]["② 분류"].isin(값.split("|")), None
