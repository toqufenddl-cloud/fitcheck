"""판정 엔진. 규칙 표(src/rules/*.csv)를 읽어 증권 CoT에 적용하고 CoT당 한 행의 판정 결과를 돌려준다.

- 규칙과 엔진을 나눈다. 기준·시행일이 바뀌면 rules.csv / d43_range.csv만 고친다.
- 같은 기준(rules.csv의 '기준' 칸) 안에서는 우선순위가 가장 낮은 숫자부터 보고 처음 맞는 규칙이 판정·처리·사유를 정한다.
  맞은 규칙은 모두 '적용 규칙'에 남긴다 (조건이 always()인 기본값 규칙은 빼고).
- 처리 우선순위: 제외 > 교정 > 태그 > 표시. '처리'는 가장 앞선 것 하나, '처리 전체'는 모두.
- 원문 문장은 결과에 넣지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import predicates
from src.extract import (PROPENSITY, body_grade, build_product_index, match_product, parse_allowed_grades,
                         product_grade, propensity_in, risk_mention_after_name)
from src.criteria12 import judge12
from src.transfer import CLASS_RANK, load_terms, question_mentions, transfer_expressions

RULE_DIR = Path(__file__).resolve().parent / "rules"
ROOT = Path(__file__).resolve().parents[1]
ACTION_ORDER = ["제외", "교정", "태그", "표시"]
P5_FROM = "2026-11-23"   # 판정기준서 v1.1 (D64) 시행일


# ---------------------------------------------------------------------------
# 규칙 표
# ---------------------------------------------------------------------------

def load_range(as_of: str, standard: str = "공개기준", path: Path = RULE_DIR / "d43_range.csv") -> dict:
    """as_of 날짜에 시행 중인 허용 범위표 (성향 → 등급 집합).
    애매한 경우 처리: 같은 성향에 시행 중인 행이 둘 이상이면 시행일이 가장 늦은 행을 쓴다."""
    t = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    t = t[(t["기준명"] == standard) & (t["시행일"] <= as_of) & ((t["종료일"] == "") | (t["종료일"] >= as_of))]
    t = t.sort_values("시행일").groupby("성향").tail(1)
    return {p: {int(x) for x in g.split("-")} for p, g in zip(t["성향"], t["허용 등급"])}


def load_rules(as_of: str, path: Path = RULE_DIR / "rules.csv") -> pd.DataFrame:
    t = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    t = t[t["시행일"] <= as_of].copy()
    t["우선순위"] = t["우선순위"].astype(int)
    return t.sort_values(["기준", "우선순위"]).reset_index(drop=True)


_COND = re.compile(r"^\s*(\w+)\((.*)\)\s*$")


def parse_condition(cond: str) -> list:
    """'a(k=v) & b()' → [('a', {'k': 'v'}), ('b', {})]. 값 안의 '|'는 '또는'으로 함수가 해석한다."""
    out = []
    for part in cond.split(" & "):
        m = _COND.match(part)
        if not m:
            raise ValueError(f"조건 형식 오류: {part!r}")
        args = {}
        for kv in filter(None, m.group(2).split(";")):
            k, v = kv.split("=", 1)
            args[k.strip()] = v.strip()
        out.append((m.group(1), args))
    return out


# ---------------------------------------------------------------------------
# 입력 준비
# ---------------------------------------------------------------------------

def prepare(cot: pd.DataFrame, prod: pd.DataFrame, as_of: str, standard: str = "공개기준",
            review_path: Path = ROOT / "data" / "review" / "범위밖_분류표.csv",
            log_path: Path = ROOT / "outputs" / "corrections_log.csv",
            tags_path: Path = ROOT / "outputs" / "tags.csv",
            transfer_review_path: Path = ROOT / "data" / "review" / "사람확인_판정.csv") -> dict:
    """규칙 함수가 쓰는 표를 만든다 (증권 CoT만, D18). 원본은 바꾸지 않는다."""
    rng = load_range(as_of, standard)
    sc = cot[cot["category"] == "증권"].copy().set_index("cot_id")
    sc["성향_cot1"] = sc["cot1"].map(propensity_in)
    sc["성향_cot2"] = sc["cot2"].map(propensity_in)
    # D21: cot2 단일 성향 우선, 없으면 cot1 단일 성향
    sc["성향_D21"] = np.where(sc["성향_cot2"].isin(PROPENSITY), sc["성향_cot2"],
                            np.where(sc["성향_cot1"].isin(PROPENSITY), sc["성향_cot1"], "비교 불가"))
    # 기준 ①② (판정기준서 v1.1 1·2장, D64)
    sc = sc.join(judge12(sc))
    # 판정기준서 2장: D21로 성향을 정할 수 없지만 P5로 단일 성향이 정해지면 그 성향으로 범위를 다시 판정한다.
    # 애매한 경우 처리: D21로 이미 단일 성향인 CoT는 바꾸지 않는다 (P5와 D21이 다른 CoT는 결과에 따로 표시만 한다).
    sc["P5 보충"] = (as_of >= P5_FROM) & (sc["성향_D21"] == "비교 불가") & sc["성향_P5"].isin(PROPENSITY)
    sc.loc[sc["P5 보충"], "성향_D21"] = sc.loc[sc["P5 보충"], "성향_P5"]
    # D65 (판정기준서 v1.2, 사용자 결정): D21로 단일 성향인데 P5가 다르면 P5를 따른다.
    # P5가 복수면 후보 중 가장 낮은 성향(P8)으로 ③을 판정한다 (②는 judge12에서 이미 ②-b).
    # 애매한 경우 처리: D21이 '비교 불가'인 CoT(사람 판정이 있는 6799 등)는 여기서 바꾸지 않는다 (새 규칙을 만들지 않음).
    #   P5가 '없음'이고 D21이 단일인 CoT가 생기면 성향을 '비교 불가'로 둔다 (P5를 따름).
    def _lowest(c):
        xs = [x for x in str(c).split("|") if x in PROPENSITY]
        return max(xs, key=PROPENSITY.index) if xs else "비교 불가"
    d21_single = sc["성향_D21"].isin(PROPENSITY) & ~sc["P5 보충"]
    sc["P5 우선"] = (as_of >= P5_FROM) & d21_single & (sc["성향_P5"] != sc["성향_D21"])
    sc["성향_D21 원래"] = sc["성향_D21"]
    sc.loc[sc["P5 우선"], "성향_D21"] = [
        _lowest(c) if p == "복수" else (p if p in PROPENSITY else "비교 불가")
        for p, c in zip(sc.loc[sc["P5 우선"], "성향_P5"], sc.loc[sc["P5 우선"], "P5 후보"])]
    sc["허용등급"] = sc["cot2"].map(lambda t: (lambda s: "-".join(map(str, sorted(s))) if s else "추출 실패")(parse_allowed_grades(t)[0]))

    # 추천 상품 (D10 매칭, 등급은 상품 JSON 등록 등급)
    idx = build_product_index(prod)
    recs = (sc.reset_index()[["cot_id", "product_names"]].explode("product_names").dropna()
            .rename(columns={"product_names": "rec_name"}).reset_index(drop=True))
    recs["keys"] = recs["rec_name"].map(lambda n: match_product(n, idx)[1])
    recs["json_grade"] = recs["keys"].map(lambda k: product_grade(k, idx))
    p21 = recs["cot_id"].map(sc["성향_D21"])
    recs["밖"] = [pd.notna(g) and p in rng and int(g) not in rng[p] for g, p in zip(recs["json_grade"], p21)]

    # D19: 추천 상품마다 answer에 숫자 등급 또는 공식 명칭
    r19 = recs.merge(sc[["answer", "product_names"]], left_on="cot_id", right_index=True)
    r19["충족"] = r19.apply(lambda r: risk_mention_after_name(r["answer"], r["rec_name"], r["product_names"]), axis=1) \
                     .isin(["숫자 등급", "공식 명칭", "둘 다"])
    lvl = r19.groupby("cot_id")["충족"].agg(lambda s: "전부 충족" if s.all() else ("전무" if not s.any() else "일부"))
    sc["D19"] = lvl.reindex(sc.index).fillna("판정 불가")

    # D51: 설명 속 등급(body_grade)과 등록 등급
    ps = prod[prod["domain"] == "증권"]
    bg = ps["investment_strategy"].map(body_grade)
    tbl = idx["table"]
    fld = ps["file_name"].map(tbl["grade"])
    is_int = bg.map(lambda x: isinstance(x, int))
    mismatch_keys = set(ps.loc[is_int & (bg.where(is_int) != fld), "file_name"])
    multi_keys = set(ps.loc[bg == "본문 등급 복수", "file_name"])

    # 이관 (D50·D56)
    terms = load_terms()
    tr = [transfer_row(cid, r, recs, rng, terms) for cid, r in sc.iterrows()]
    tr = pd.DataFrame(tr).set_index("cot_id")
    sc = sc.join(tr)

    review = pd.read_csv(review_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    corrections = pd.read_csv(log_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    tags = pd.read_csv(tags_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    # 사람 확인 사례의 사람 판정 (D67, P3). 파일이 없으면 빈 표 (규칙만으로 판정).
    tcols = ["cot_id", "내 분류", "이관 판정", "메모", "출처"]
    transfer_review = (pd.read_csv(transfer_review_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
                       if Path(transfer_review_path).exists() else pd.DataFrame(columns=tcols))
    return {"cots": sc, "recs": recs, "range": rng, "review": review, "corrections": corrections, "tags": tags,
            "transfer_review": transfer_review,
            "mismatch_keys": mismatch_keys, "multi_keys": multi_keys, "as_of": as_of, "standard": standard}


def transfer_row(cid, r, recs, rng, terms) -> dict:
    """CoT 하나의 이관 분류·결과.

    - 언급 등급별 분류(src/transfer.question_mentions) 중 '성향 밖'(해석 등급이 기준표 범위 밖)인 것만 결과를 정한다.
    - 결함: 성향 밖 + R1·R2·R4 + 그 등급 상품을 실제 추천 + cot1~answer에 이관 표현 없음 (D50·D56)
      R4는 해석 등급(4)이 성향 안이어도, 문자 그대로의 3등급이 성향 밖이고 3등급을 추천했으면 결함으로 본다.
      경험(D50 갱신 2026-11-18): 성향 밖 등급의 '경험' 언급만 있는데 그 등급을 추천하고 이관 표현이 없으면 결함
      (경험은 현재 의도를 대신하지 않으므로 상담사가 현재 의도를 확인해야 한다).
    - 사람 확인: 성향 밖 언급이 '사람 확인'으로 분류됨, 또는 성향을 알 수 없는데(D21 비교 불가) 등급 언급이 있음
    - R3 대상: 성향 밖 R1·R2·R4인데 question에 그 등급의 '경험' 문장이 없음 (상담사 비교 상담 대상, 표시만)
    애매한 경우 처리: 언급 등급이 여러 개면 하나라도 결함이면 결함, 아니면 하나라도 사람 확인이면 사람 확인.
    """
    qm = question_mentions(r["question"], terms)
    transfer, generic = transfer_expressions([r["cot1"], r["cot2"], r["cot3"], r["answer"]], terms)
    out = {"cot_id": cid, "이관 표현": transfer, "일반 상담 문구": generic, "이관 분류": "언급 없음",
           "이관 결과": "해당 없음", "성향 밖 언급": "", "R3 대상": False, "사람 확인 사유": ""}
    if qm.empty:
        return out
    out["이관 분류"] = min(qm["분류"], key=CLASS_RANK.index)
    p = r["성향_D21"]
    if p not in rng:
        out["이관 결과"] = "사람 확인"
        out["사람 확인 사유"] = "성향 불명(D21 비교 불가)"
        return out
    rec_g = set(recs.loc[recs["cot_id"] == cid, "json_grade"].dropna().astype(int))
    outside = qm[~qm["해석 등급"].isin(rng[p])]
    out["성향 밖 언급"] = ",".join(f"{g}:{c}" for g, c in zip(outside["해석 등급"], outside["분류"]))
    wish = outside[outside["분류"].isin(["R1 명확한 희망", "R2 고려 중", "R4 비교 표현"])]
    exp_grades = set(qm.loc[qm["분류"] == "경험", "언급 등급"])
    out["R3 대상"] = bool(len(wish)) and not set(wish["언급 등급"]) <= exp_grades
    # R4: '예금보다 다소 높은 위험'을 문자 그대로 3등급으로 읽어 성향 밖 3등급을 추천한 경우도 결함 (D50 R4, 2552)
    r4_lit = qm[(qm["분류"] == "R4 비교 표현") & ~qm["언급 등급"].isin(rng[p])]
    if len(r4_lit):
        out["성향 밖 언급"] = ",".join(filter(None, [out["성향 밖 언급"]] +
                                         [f"{g}(문자):R4 비교 표현" for g in r4_lit["언급 등급"]]))
    # 경험 규칙 (D50 갱신 2026-11-18): 성향 밖 등급의 경험만을 근거로 그 등급을 바로 추천하면 결함
    exp_out = outside[outside["분류"] == "경험"]
    if not transfer and ((len(wish) and (set(wish["해석 등급"]) & rec_g)) or (set(r4_lit["언급 등급"]) & rec_g)
                         or (set(exp_out["해석 등급"]) & rec_g)):
        out["이관 결과"] = "결함"
    elif (outside["분류"] == "사람 확인").any():
        out["이관 결과"] = "사람 확인"
        out["사람 확인 사유"] = ", ".join(sorted(set(outside.loc[outside["분류"] == "사람 확인", "사람 확인 사유"])))
    return out


# ---------------------------------------------------------------------------
# 규칙 적용
# ---------------------------------------------------------------------------

def apply_rules(ctx: dict, rules: pd.DataFrame) -> pd.DataFrame:
    ids = ctx["cots"].index
    res = pd.DataFrame(index=ids)
    applied = pd.Series([[] for _ in ids], index=ids)
    for crit, grp in rules.groupby("기준", sort=False):
        verdict = pd.Series("", index=ids, dtype=object)
        action = pd.Series("", index=ids, dtype=object)
        reason = pd.Series("", index=ids, dtype=object)
        decided = pd.Series(False, index=ids)
        for _, rule in grp.sort_values("우선순위").iterrows():
            mask, detail = pd.Series(True, index=ids), None
            for name, args in parse_condition(rule["조건"]):
                m, d = predicates.REGISTRY[name](ctx, **args)
                mask &= m.reindex(ids).fillna(False).astype(bool)
                detail = d if d is not None else detail
            if rule["조건"].strip() != "always()":
                for cid in ids[mask]:
                    applied[cid].append(rule["rule_id"])
            new = mask & ~decided
            verdict[new] = rule["판정"]
            action[new] = rule["처리"]
            code = rule["사유 코드"]
            if code.startswith("@") and detail is not None:
                reason[new] = detail.reindex(ids)[new].fillna("").astype(str)
            elif rule["rule_id"] == "TAG-FILE" and detail is not None:
                reason[new] = detail.reindex(ids)[new].fillna("").astype(str)
            else:
                reason[new] = code
            decided |= mask
        res[f"{crit} 판정"] = verdict
        res[f"{crit} 처리"] = action
        res[f"{crit} 사유"] = reason
    res["적용 규칙"] = applied.map(lambda xs: "|".join(xs))
    return res


def judge(cot: pd.DataFrame, prod: pd.DataFrame, as_of: str = "2026-11-17", standard: str = "공개기준",
          drop_rules: tuple = (), ctx: dict | None = None, **paths):
    """증권 CoT를 판정해 CoT당 한 행을 돌려준다. 반환: (결과 데이터프레임, ctx)

    - drop_rules: 빼고 돌릴 rule_id (04 검증: 사람 판정 규칙 R3-HUMAN을 뺀 자동 규칙만의 결과)
    - ctx: 이미 만든 ctx를 다시 쓸 때 (prepare 생략)
    """
    ctx = ctx or prepare(cot, prod, as_of, standard, **paths)
    rules = load_rules(as_of)
    rules = rules[~rules["rule_id"].isin(drop_rules)]
    r = apply_rules(ctx, rules)
    c = ctx["cots"]
    out = pd.DataFrame({
        "cot_id": c.index,
        "분할": c["split"].values,
        "기준① 판정": r["① 판정"].values,
        "기준① 확인 수": c["① 확인 수"].values,
        "기준① 법정 미확인": c["① 법정 미확인"].values,
        "기준② 판정": r["② 판정"].values,
        "성향_P5": c["성향_P5"].values,
        "P5 보충": c["P5 보충"].values,
        "P5 우선": c["P5 우선"].values,
        "기준③ 판정": r["③ 판정"].values,
        "기준③ 처리": r["③ 처리"].values,
        "사유 코드": r["③ 사유"].values,
        "D30 태그": r["③태그 판정"].values,
        "기준④ 판정": r["④ 판정"].values,
        "원천 확인 필요": (r["원천 판정"] == "원천 확인 필요").values,
        "원천 등급 복수": c.index.isin(r.index[r["적용 규칙"].str.contains("SRC-D51-MULTI")]),
        "이관 분류": c["이관 분류"].values,
        "이관 판정": r["이관 판정"].values,
        "성향 밖 언급": c["성향 밖 언급"].values,
        "R3 대상": c["R3 대상"].values,
        "사람 확인 사유": c["사람 확인 사유"].values,
        "이관 표현": c["이관 표현"].values,
        "일반 상담 문구": c["일반 상담 문구"].values,
        "교정": (r["교정 처리"] == "교정").values,
        "기록 태그": r["기록 태그 사유"].values,
    })
    acts = pd.DataFrame({k: r[f"{k} 처리"].values for k in ["②", "③", "③태그", "④", "원천", "이관", "교정", "기록 태그"]})
    all_acts = acts.apply(lambda row: [a for a in ACTION_ORDER if a in set(row)], axis=1)
    out["처리"] = all_acts.map(lambda xs: xs[0] if xs else "")
    out["처리 전체"] = all_acts.map("|".join)
    out["적용 규칙"] = r["적용 규칙"].values
    return out, ctx
