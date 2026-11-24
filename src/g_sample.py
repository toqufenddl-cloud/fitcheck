"""G 표본 추정 (판정기준서 v1.3, D67): 상위 30개 밖 추천 상품에서 원천 등급별 층화 무작위 표본을 뽑아 최신 등급을 조회하고,
등급 변경 비율과 '최신 기준 부적합' 추천 수를 전체로 추정한다.

- 모집단: 증권 CoT에서 추천된 상품(추천 상품명 단위) 중 데이터 등록 등급이 있고 상위 30개(outputs/latest_grade_check.csv)에 없는 것.
- 층: 데이터 등록 등급(원천 등급 1~6). 배분: 층 크기 비례, 층마다 최소 2개 (분산 계산용), 합 200. 시드 고정.
- 조회: src.latest_grade.lookup (상품명만, 요청 간격 1초 이상). 결과는 data/llm_cache/에 상품마다 이어 쓴다 (중단 후 이어서 조회).
- 추정 (층화 추정, 정규 근사 95% 신뢰구간, 유한 모집단 수정 포함):
  · 조회 가능 비율: 최신 등급을 얻은 상품('확인'·'클래스 다름') 비율.
  · 등급 변경 비율: 조회 가능 상품 중 등급이 바뀐 비율 (결합 비 추정, 선형화 분산).
  · 최신 기준 부적합 추천 수: 상품별 G4 태그 추천 수의 층화 합계 추정. 조회 불가 상품은 0으로 센다(하한).
    상위 30개 전수 결과(16)를 더한 전체 추정도 함께 낸다.
애매한 경우 처리:
- 등록 등급이 없는 추천 상품은 층을 정할 수 없어 모집단에서 뺀다 (수를 기록).
- '사람 확인(명칭 다름)'은 조회 불가로 센다 (자동으로 같은 상품으로 보지 않음, G 규칙과 같음).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261124
N_SAMPLE = 200


def frame(recs: pd.DataFrame, top30: set) -> tuple[pd.DataFrame, dict]:
    r = recs.dropna(subset=["rec_name"])
    g = r.groupby("rec_name").agg(등급=("json_grade", "first"), 추천=("cot_id", "size")).reset_index()
    info = {"추천 상품": len(g), "상위 30 제외": int(g["rec_name"].isin(top30).sum()), "등급 없음 제외": int(g["등급"].isna().sum())}
    g = g[~g["rec_name"].isin(top30) & g["등급"].notna()].copy()
    g["층"] = g["등급"].astype(int)
    info["모집단 상품"] = len(g)
    return g.rename(columns={"rec_name": "상품명"}).reset_index(drop=True), info


def allocate(sizes: pd.Series, n: int = N_SAMPLE, floor: int = 2) -> pd.Series:
    """층 크기 비례 배분, 최소 floor, 층 크기 상한. 반올림 오차는 큰 나머지 순으로 맞춘다."""
    raw = sizes / sizes.sum() * n
    a = np.maximum(np.floor(raw), floor).astype(int).clip(upper=sizes)
    while a.sum() < n:
        rem = (raw - a).where(a < sizes, -np.inf)
        a[rem.idxmax()] += 1
    while a.sum() > n:
        a[(a - floor).where(a > floor, -1).idxmax()] -= 1
    return a


def draw(fr: pd.DataFrame, n: int = N_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    sizes = fr.groupby("층").size()
    alloc = allocate(sizes, n)
    rng = np.random.default_rng(seed)
    parts = []
    for h, k in alloc.items():
        pool = fr[fr["층"] == h].sort_values("상품명").reset_index(drop=True)
        idx = rng.choice(len(pool), size=int(k), replace=False)
        parts.append(pool.iloc[sorted(idx)].assign(층_N=int(sizes[h]), 층_n=int(k)))
    return pd.concat(parts, ignore_index=True)


def lookup_resumable(names: list, cache: Path, checked: str, delay: float = 1.0) -> pd.DataFrame:
    from src.latest_grade import lookup
    done = pd.read_csv(cache, encoding="utf-8-sig", keep_default_na=False, na_values=[""]) if cache.exists() else pd.DataFrame()
    have = set(done["상품명"]) if len(done) else set()
    for nm in names:
        if nm in have:
            continue
        row = lookup([nm], checked, delay=delay)
        row.to_csv(cache, mode="a", header=not cache.exists(), index=False, encoding="utf-8-sig")
        have.add(nm)
    return pd.read_csv(cache, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def per_product(sample: pd.DataFrame, looked: pd.DataFrame, recs: pd.DataFrame, cots: pd.DataFrame, rng: dict) -> pd.DataFrame:
    from src.latest_grade import g4_rejudge
    lk = looked.drop_duplicates("상품명", keep="last")
    s = sample.merge(lk[["상품명", "최신 등급", "상태", "조회 이름", "조회 클래스", "출처 순위", "URL", "확인일"]], on="상품명", how="left")
    s["조회 가능"] = s["상태"].isin(["확인", "클래스 다름"]) & s["최신 등급"].notna()
    s["등급 변경"] = s["조회 가능"] & (s["최신 등급"] != s["등급"])
    g = g4_rejudge(recs, cots, rng, lk)
    tag = g[g["G4 태그"] != ""].groupby("rec_name").size()
    s["최신 기준 부적합 추천"] = s["상품명"].map(tag).fillna(0).astype(int)
    s["데이터 범위 밖 추천"] = s["상품명"].map(g[g["데이터 시점"] == "범위 밖"].groupby("rec_name").size()).fillna(0).astype(int)
    return s


def _strat_total(s: pd.DataFrame, y: str):
    tot, var = 0.0, 0.0
    for _, g in s.groupby("층"):
        N, n = g["층_N"].iloc[0], len(g)
        tot += N * g[y].mean()
        if n > 1:
            var += N ** 2 * (1 - n / N) * g[y].var(ddof=1) / n
    return tot, var


def estimate(s: pd.DataFrame, top30_g4: int) -> pd.DataFrame:
    s = s.assign(_x=s["조회 가능"].astype(float), _y=s["등급 변경"].astype(float), _t=s["최신 기준 부적합 추천"].astype(float))
    N = s.groupby("층")["층_N"].first().sum()
    ci = lambda e, v: (e - 1.96 * math.sqrt(v), e + 1.96 * math.sqrt(v))
    rows = []
    tx, vx = _strat_total(s, "_x")
    lo, hi = ci(tx / N, vx / N ** 2)
    rows.append(("조회 가능 비율 (확인·클래스 다름)", tx / N * 100, max(lo, 0) * 100, min(hi, 1) * 100, "%", "모집단 상품 대비"))
    ty, _ = _strat_total(s, "_y")
    R = ty / tx if tx else float("nan")
    s["_d"] = s["_y"] - R * s["_x"]
    _, vd = _strat_total(s, "_d")
    vR = vd / tx ** 2 if tx else float("nan")
    lo, hi = ci(R, vR)
    rows.append(("등급 변경 비율 (조회 가능 상품 중)", R * 100, max(lo, 0) * 100, min(hi, 1) * 100, "%", "결합 비 추정"))
    rows.append(("등급 변경 상품 수 추정 (모집단)", ty, *[max(v, 0) for v in ci(ty, _strat_total(s, "_y")[1])], "상품", "조회 불가는 0(하한)"))
    tt, vt = _strat_total(s, "_t")
    lo, hi = ci(tt, vt)
    rows.append(("최신 기준 부적합 추천 수 추정 (상위 30 밖)", tt, max(lo, 0), hi, "추천", "조회 불가는 0(하한)"))
    rows.append(("최신 기준 부적합 추천 수 추정 (전체 = 상위 30 전수 + 표본 추정)", tt + top30_g4, max(lo, 0) + top30_g4, hi + top30_g4,
                 "추천", f"상위 30 전수 {top30_g4}"))
    out = pd.DataFrame(rows, columns=["지표", "추정치", "95% 하한", "95% 상한", "단위", "비고"])
    out[["추정치", "95% 하한", "95% 상한"]] = out[["추정치", "95% 하한", "95% 상한"]].round(1)
    return out
