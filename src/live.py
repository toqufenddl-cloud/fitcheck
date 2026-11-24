"""시연용 실시간 파이프라인 (D69): 기준 문서 검색 → 규칙 생성 → 상품 등급 검색 → 근거 조항 검색 → 규칙 적용.

    from src.live import load_base, pick_sample, run_live
    base = load_base("2026-11-24")
    for ev in run_live(base, pick_sample(base, 30), live=True): print(ev)

run_live는 단계마다 이벤트(dict)를 yield 한다: {"step": 1~5, "kind": "msg"|"warn"|"table"|"progress"|"done", ...}.
Streamlit 앱(app/fitcheck_app.py)과 CLI(python -m src.fitcheck live)가 같은 함수를 쓴다.

원칙
- 웹으로 나가는 것은 공개 문서 URL 내려받기와 상품명 조회뿐이다. 원문 문장은 밖으로 보내지 않는다 (P10, D60).
- 결과 파일에는 cot_id·상품명·등급·조항만 쓴다 (원문 없음). 내려받은 문서는 data/live/ (git 제외).
애매한 경우 처리
- 1단계: 원본 URL이 문서 파일을 직접 가리키지 않는 문서(법령 사이트, 협회 홈페이지)는 내려받지 않고 '보관본 사용'으로 표시한다.
  10초 안에 받지 못하면 '접속 실패'로 두고 보관본으로 계속한다.
- 2단계: 읽은 등급표가 저장 표(d43_range.csv)와 다르면 자동으로 바꾸지 않는다. '변경 감지, 승인 필요'로 표시하고
  저장 표로 판정한다 (D69). 읽기에 실패해도 저장 표로 계속한다. 받은 문서가 없으면 보관본에서 읽고 그렇게 표시한다.
- 3단계: 조회 결과에 G 규칙을 그대로 쓴다: '클래스 다름'은 대표 클래스 등급을 쓰되 표시, '명칭 다름'·'확인 불가'는 최신 등급 없음
  (재판정 불가, 데이터 등급 유지). data/review/등급변경_확인.csv에 사람 확인 등급이 있으면 그 값을 우선하고, 실시간 값과 다르면 표시한다.
  조회가 연속 2번 실패하면 인터넷이 끊긴 것으로 보고 나머지는 저장값(이전 조회 결과)으로 계속한다.
- 4단계: Ollama가 응답하지 않거나 색인이 없으면 검색을 건너뛰고 gold 매핑(src/rag/gold_articles.csv)으로 대체한다.
- 5단계: ③은 공개 기준(2단계 표)으로 데이터 등급과 최신 등급을 각각 판정한다. 최신 등급이 없는 상품은 데이터 등급을 쓴다.
  ④·이관·②는 문서·등급이 바뀌어도 판정이 같으므로 저장된 판정(outputs/judgments.csv와 같은 judge 결과)을 양쪽에 쓴다.
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LIVE_DIR = ROOT / "data" / "live"
CACHE = ROOT / "data" / "llm_cache"
REVIEW = ROOT / "data" / "review" / "등급변경_확인.csv"
SEED = 20261124
RULE_DOCS = ["LAW-FCPA", "REG-KOFIA", "CO-KIS"]
PROPS = ["공격투자형", "적극투자형", "위험중립형", "안정추구형", "안정형"]
QUERY = {
    "범위": "투자성향 고객에게 권유할 수 있는 금융투자상품 위험등급 적합성 원칙",
    "이관": "고객이 투자성향보다 위험이 높은 상품을 원할 때 부적합 사실 고지 확인 절차 상담",
    "설명": "금융상품 설명의무 위험등급 손실 가능성 설명",
    "②-a": "적합성 원칙 일반금융소비자 정보 파악 면담 질문 확인 투자성향 분류",
    "②-b": "적합성 원칙 일반금융소비자 정보 파악 면담 질문 확인 투자성향 분류",
}
TYPE_LABEL = {"범위": "③ 범위 밖", "이관": "이관 없음", "설명": "④ 위험등급 설명 누락", "②-a": "②-a 근거 부족 확정",
              "②-b": "②-b 성향 미확정 추천"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", **kw)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# 준비: 데이터 판정 (저장값 기준)과 표본
# ---------------------------------------------------------------------------

def load_base(as_of: str = "2026-11-24") -> dict:
    """원본을 읽고 데이터 기준 판정을 한 번 돌린다 (약 30초). 반환: cot, j, ctx, as_of, g4_saved."""
    from src.judge import judge
    from src.load_cot import load_cot, load_products
    cot, prod = load_cot(), load_products()
    j, ctx = judge(cot, prod, as_of=as_of)
    g4 = _read(OUT / "latest_grade_rejudge.csv", keep_default_na=False)   # 'G4 태그' 빈 값은 ''로 둔다
    return {"cot": cot, "j": j, "ctx": ctx, "as_of": as_of, "g4_saved": g4}


def _out_ids(base: dict, rng: dict | None = None) -> set:
    """공개 기준(데이터 등급)으로 범위 밖 추천이 있는 CoT."""
    rng = rng or base["ctx"]["range"]
    rec = base["ctx"]["recs"].dropna(subset=["json_grade"])
    p = rec["cot_id"].map(base["ctx"]["cots"]["성향_D21"])
    bad = [pp in rng and int(g) not in rng[pp] for g, pp in zip(rec["json_grade"], p)]
    return set(rec.loc[bad, "cot_id"].astype(int))


def pick_sample(base: dict, n: int = 30, seed: int = SEED) -> list:
    """표본: 최신 기준 부적합 CoT min(n/3, 16) + 공개 기준 범위 밖 최소 3 + 나머지 고정 시드 무작위.
    애매한 경우 처리: n이 작아 두 조건을 다 채울 수 없으면 최신 기준 부적합을 먼저 채운다."""
    rs = np.random.default_rng(seed)
    g4 = base["g4_saved"]
    g4_ids = sorted(set(g4.loc[g4["G4 태그"] != "", "cot_id"].astype(int)))
    k = min(n // 3, len(g4_ids))
    pick = list(rs.choice(g4_ids, size=k, replace=False)) if k else []
    outs = sorted(_out_ids(base) - set(pick))
    m = max(0, min(3, n - len(pick)))
    pick += list(rs.choice(outs, size=min(m, len(outs)), replace=False))
    rest = sorted(set(base["j"]["cot_id"].astype(int)) - set(pick))
    pick += list(rs.choice(rest, size=max(0, n - len(pick)), replace=False))
    return sorted(int(x) for x in pick)


# ---------------------------------------------------------------------------
# 1단계: 기준 문서 검색
# ---------------------------------------------------------------------------

def check_documents(as_of: str, live: bool = True, timeout: float = 10.0) -> pd.DataFrame:
    idx = _read(ROOT / "docs" / "regulations" / "regulations_index.csv", dtype=str, keep_default_na=False).set_index("doc_id")
    rows = []
    for d in RULE_DOCS:
        r = idx.loc[d]
        saved = ROOT / "docs" / "regulations" / r["파일"]
        row = {"doc_id": d, "문서명": r["문서명"], "시행일": r["시행일"], "URL": r["출처 URL"],
               "보관본 sha256[:16]": _sha(saved.read_bytes())[:16] if saved.is_file() else "", "받은 문서 sha256[:16]": "",
               "상태": "", "받은 파일": "", "검색 시각": _now()}
        direct = r["출처 URL"].lower().endswith(".pdf")
        if not live:
            row["상태"] = "저장값 사용 (실시간 검색 안 함)"
        elif not direct:
            row["상태"] = "보관본 사용 (원본 URL이 문서 파일이 아님)" + (f", 시행 {r['시행일']}" if r["시행일"] else "")
        else:
            try:
                import httpx
                resp = httpx.get(r["출처 URL"], timeout=timeout, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (research script)"})
                resp.raise_for_status()
                LIVE_DIR.mkdir(parents=True, exist_ok=True)
                f = LIVE_DIR / f"{d}_{as_of}{Path(r['파일']).suffix}"
                f.write_bytes(resp.content)
                h = _sha(resp.content)[:16]
                row.update({"받은 문서 sha256[:16]": h, "받은 파일": str(f.relative_to(ROOT)),
                            "상태": "변경 없음" if h == row["보관본 sha256[:16]"] else "변경됨 (보관본과 해시 다름)"})
            except Exception as e:   # 접속 실패: 보관본으로 계속
                row["상태"] = f"접속 실패 ({type(e).__name__}), 보관본 사용"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2단계: 규칙 생성 (KIS 준칙 제6조① 등급표)
# ---------------------------------------------------------------------------

_GRADE = re.compile(r"(공격투자형|적극투자형|위험중립형|안정추구형|안정형)\s*:\s*(\d)\s*등급")


def parse_kis_table(pdf: Path) -> dict:
    """제6조① 각 호 '성향: N등급 및 그 보다 낮은 위험의 등급' → {성향: {N..6}}. 다섯 성향을 다 못 읽으면 빈 dict."""
    from pypdf import PdfReader
    t = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)
    m = re.search(r"제\s*6\s*조\s*\(적합성\s*원칙\)\s*①", t)
    if not m:
        return {}
    seg = t[m.start(): m.start() + 2500]
    seg = seg[: seg.find("②")] if "②" in seg else seg
    out = {p: set(range(int(g), 7)) for p, g in _GRADE.findall(seg)}
    return out if set(out) == set(PROPS) else {}


def build_rule_table(docs: pd.DataFrame, saved_rng: dict) -> dict:
    kis = docs.set_index("doc_id").loc["CO-KIS"]
    got = ROOT / kis["받은 파일"] if kis["받은 파일"] else None
    src = "검색한 문서" if got else "보관본"
    pdf = got or ROOT / "docs" / "regulations" / "CO-KIS" / "231012_rule.pdf"
    try:
        parsed = parse_kis_table(pdf) if pdf.is_file() else {}
    except Exception:
        parsed = {}
    tbl = pd.DataFrame({"성향": PROPS,
                        "읽은 허용 등급": ["-".join(map(str, sorted(parsed[p]))) if p in parsed else "" for p in PROPS],
                        "저장 표 허용 등급": ["-".join(map(str, sorted(saved_rng.get(p, [])))) for p in PROPS]})
    tbl["같음"] = tbl["읽은 허용 등급"] == tbl["저장 표 허용 등급"]
    label = f"{src}(KIS 준칙 제6조①, 시행 {kis['시행일']})"
    if not parsed:
        status, msg = "읽기 실패", f"{label}에서 등급표를 읽지 못함 → 저장 표(d43_range.csv)로 판정"
    elif tbl["같음"].all():
        status, msg = "일치", f"{label}의 등급표로 판정"
    else:
        status, msg = "변경 감지, 승인 필요", f"{label}의 등급표가 저장 표와 다름 → 변경 감지, 승인 필요. 승인 전이므로 저장 표로 판정 (D69)"
    return {"table": tbl, "status": status, "message": msg, "range": saved_rng}


# ---------------------------------------------------------------------------
# 3단계: 상품 등급 검색
# ---------------------------------------------------------------------------

def saved_latest() -> pd.DataFrame:
    """이전 조회 결과 모음 (상위 30, G 표본, 시연 캐시). 같은 상품은 마지막 것을 쓴다."""
    parts = []
    for p in [OUT / "latest_grade_check.csv", OUT / "latest_grade_sample.csv", *sorted(CACHE.glob("live_lookup_*.csv"))]:
        if p.exists():
            d = _read(p, keep_default_na=False, na_values=[""])
            parts.append(d.reindex(columns=["상품명", "최신 등급", "상태", "조회 이름", "URL", "확인일"]))
    if not parts:
        return pd.DataFrame(columns=["상품명", "최신 등급", "상태", "조회 이름", "URL", "확인일"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("상품명", keep="last")


def _human() -> pd.DataFrame:
    if not REVIEW.exists():
        return pd.DataFrame(columns=["상품명", "최신 등급", "사람 확인", "확인일"])
    return _read(REVIEW, keep_default_na=False).drop_duplicates("상품명", keep="last").set_index("상품명")


def lookup_products(names: list, as_of: str, live: bool = True, delay: float = 0.5):
    """제너레이터: ('progress', i, n, 상품명) 이벤트를 내고, 마지막에 ('result', DataFrame, 메시지 목록)."""
    from src.latest_grade import lookup
    saved = saved_latest().set_index("상품명")
    cache = CACHE / f"live_lookup_{as_of}.csv"
    rows, msgs, fails, offline = [], [], 0, not live
    for i, nm in enumerate(names, 1):
        row = None
        if not offline:
            try:
                row = lookup([nm], as_of, delay=delay).iloc[0].to_dict()
                row["값 출처"] = "실시간 조회"
                fails = 0
                pd.DataFrame([row]).drop(columns=["값 출처"]).to_csv(cache, mode="a", header=not cache.exists(), index=False,
                                                                      encoding="utf-8-sig")
            except Exception as e:
                fails += 1
                msgs.append(f"조회 실패: {type(e).__name__}")
                if fails >= 2:
                    offline = True
                    msgs.append("연속 2번 실패 → 인터넷 연결 없음으로 보고 나머지는 저장값으로 계속")
        if row is None:
            if nm in saved.index:
                s = saved.loc[nm]
                row = {"상품명": nm, "최신 등급": s["최신 등급"], "상태": s["상태"], "조회 이름": s["조회 이름"], "URL": s["URL"],
                       "확인일": s["확인일"], "값 출처": "저장값"}
            else:
                row = {"상품명": nm, "최신 등급": np.nan, "상태": "저장값 없음", "조회 이름": "", "URL": "", "확인일": "",
                       "값 출처": "저장값 없음"}
        yield ("progress", i, len(names), nm)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["최신 등급"] = pd.to_numeric(df["최신 등급"], errors="coerce")
    # G 규칙: 명칭 다름·확인 불가는 최신 등급 없음 (lookup이 이미 NaN으로 둔다). 클래스 다름은 등급을 쓰되 표시
    df.loc[~df["상태"].isin(["확인", "클래스 다름"]), "최신 등급"] = np.nan
    hm = _human()
    df["사람 확인 등급"] = df["상품명"].map(pd.to_numeric(hm["최신 등급"], errors="coerce")) if len(hm) else np.nan
    df["실시간과 다름"] = df["사람 확인 등급"].notna() & df["최신 등급"].notna() & (df["사람 확인 등급"] != df["최신 등급"])
    df["판정 등급"] = df["사람 확인 등급"].where(df["사람 확인 등급"].notna(), df["최신 등급"])
    df["판정 등급 출처"] = np.where(df["사람 확인 등급"].notna(), "사람 확인", np.where(df["최신 등급"].notna(), df["값 출처"], ""))
    yield ("result", df, msgs)


# ---------------------------------------------------------------------------
# 4단계: 근거 조항 검색
# ---------------------------------------------------------------------------

def ollama_up(timeout: float = 2.0) -> bool:
    from src.rag import OLLAMA_URL
    try:
        import httpx
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _gold(types: list) -> pd.DataFrame:
    g = _read(ROOT / "src" / "rag" / "gold_articles.csv", dtype=str, keep_default_na=False)
    return g[g["유형"].isin(types) & (g["조항"] != "")]


def search_articles(types: list, as_of: str) -> dict:
    """반환: {"hits": DataFrame(유형, 순위, doc_id, 조항, 시행일, 점수, gold 일치, 검색 시각), "mode": "검색"|"gold 매핑", "message"}."""
    idx = _read(ROOT / "docs" / "regulations" / "regulations_index.csv", dtype=str, keep_default_na=False).set_index("doc_id")["시행일"]
    gold = _gold(types)
    if not ollama_up():
        from src.rag import OLLAMA_URL
        msg = (f"Ollama가 응답하지 않음({OLLAMA_URL}) → 임베딩 검색을 건너뛰고 gold 매핑(src/rag/gold_articles.csv)으로 대체. "
               "켜려면 'ollama serve' 실행 후 다시 분석")
        h = gold.assign(순위=None, 점수=None, **{"gold 일치": True, "검색 시각": _now(), "시행일": gold["doc_id"].map(idx)})
        return {"hits": h[["유형", "순위", "doc_id", "조항", "시행일", "점수", "gold 일치", "검색 시각"]], "mode": "gold 매핑", "message": msg}
    from src.rag.index import STORE_DIR, load, search
    if not (STORE_DIR / "index.faiss").exists():
        return {"hits": gold.assign(순위=None, 점수=None, **{"gold 일치": True, "검색 시각": _now(), "시행일": gold["doc_id"].map(idx)}),
                "mode": "gold 매핑", "message": "검색 색인(data/llm_cache/faiss)이 없어 gold 매핑으로 대체"}
    store = load()
    rows = []
    for t in types:
        gs = set(zip(gold.loc[gold["유형"] == t, "doc_id"], gold.loc[gold["유형"] == t, "조항"]))
        for k, (cid, score, meta, _) in enumerate(search(store, QUERY[t], as_of, company="한국투자증권", k=3), 1):
            rows.append({"유형": t, "순위": k, "doc_id": meta.get("doc_id", ""), "조항": meta.get("조항", ""),
                         "시행일": meta.get("시행일", "") or idx.get(meta.get("doc_id", ""), ""), "점수": round(score, 3),
                         "gold 일치": (meta.get("doc_id", ""), meta.get("조항", "")) in gs, "검색 시각": _now()})
    return {"hits": pd.DataFrame(rows), "mode": "검색", "message": f"bge-m3 임베딩 검색, 기준일 {as_of} 시행 조항, 유형별 상위 3개"}


# ---------------------------------------------------------------------------
# 5단계: 규칙 적용
# ---------------------------------------------------------------------------

def apply_rules(base: dict, ids: list, rng: dict, latest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (CoT별 결과, 추천별 결과)."""
    c, j = base["ctx"]["cots"], base["j"].set_index("cot_id")
    rec = base["ctx"]["recs"]
    rec = rec[rec["cot_id"].isin(ids)].copy()
    lt = latest.set_index("상품명")
    rec["성향"] = rec["cot_id"].map(c["성향_D21"])
    rec["최신 등급"] = rec["rec_name"].map(lt["판정 등급"]) if len(lt) else np.nan
    rec["최신 등급 출처"] = rec["rec_name"].map(lt["판정 등급 출처"]) if len(lt) else ""
    rec["판정 등급(최신)"] = rec["최신 등급"].where(rec["최신 등급"].notna(), rec["json_grade"])

    def out(g, p):
        if pd.isna(g) or p not in rng:
            return "판정 불가"
        return "범위 안" if int(g) in rng[p] else "범위 밖"
    rec["데이터 기준"] = [out(g, p) for g, p in zip(rec["json_grade"], rec["성향"])]
    rec["최신 문서 기준"] = [out(g, p) for g, p in zip(rec["판정 등급(최신)"], rec["성향"])]
    rows = []
    for cid in ids:
        r, jr = rec[rec["cot_id"] == cid], j.loc[cid]
        d3 = "범위 밖" if (r["데이터 기준"] == "범위 밖").any() else ("범위 안" if (r["데이터 기준"] == "범위 안").any() else "판정 불가")
        l3 = "범위 밖" if (r["최신 문서 기준"] == "범위 밖").any() else ("범위 안" if (r["최신 문서 기준"] == "범위 안").any() else "판정 불가")
        d4 = jr["기준④ 판정"] in ("일부 기재", "기재 없음")
        tr = jr["이관 판정"] == "결함"
        c2 = jr["기준② 판정"] in ("②-a 근거 부족 확정", "②-b 성향 미확정 추천")
        rows.append({"cot_id": cid, "분할": jr["분할"], "성향": c.at[cid, "성향_D21"],
                     "추천 상품(데이터→최신)": " | ".join(
                         f"{n}[{'' if pd.isna(a) else int(a)}→{'' if pd.isna(b) else int(b)}]"
                         for n, a, b in zip(r["rec_name"], r["json_grade"], r["최신 등급"])),
                     "③ 데이터 기준": d3, "③ 최신 문서 기준": l3, "③ 바뀜": d3 != l3,
                     "저장 판정 ③": jr["기준③ 판정"], "④": jr["기준④ 판정"], "이관": jr["이관 판정"], "②": jr["기준② 판정"],
                     "결함(데이터 기준)": (d3 == "범위 밖") or d4 or tr or c2, "결함(최신 문서 기준)": (l3 == "범위 밖") or d4 or tr or c2})
    return pd.DataFrame(rows), rec


def cards(res: pd.DataFrame, art: dict) -> pd.DataFrame:
    """결함 CoT마다 (유형, 기준 시점) 근거 카드. 문서·조항은 gold 매핑, 검색 결과는 상위 3 목록과 gold 포함 여부로 붙인다."""
    hits, mode = art["hits"], art["mode"]
    gold = _gold(list(QUERY))
    idx = _read(ROOT / "docs" / "regulations" / "regulations_index.csv", dtype=str, keep_default_na=False).set_index("doc_id")["시행일"]
    rows = []
    for _, r in res.iterrows():
        types = []
        if r["③ 데이터 기준"] == "범위 밖" or r["③ 최신 문서 기준"] == "범위 밖":
            types.append(("범위", "데이터" if r["③ 데이터 기준"] == "범위 밖" else "최신 문서만"))
        if r["④"] in ("일부 기재", "기재 없음"):
            types.append(("설명", "양쪽"))
        if r["이관"] == "결함":
            types.append(("이관", "양쪽"))
        if r["②"].startswith("②-a"):
            types.append(("②-a", "양쪽"))
        if r["②"].startswith("②-b"):
            types.append(("②-b", "양쪽"))
        for t, when in types:
            h = hits[hits["유형"] == t] if len(hits) else hits
            top = " ; ".join(f"{a} {b}" for a, b in zip(h["doc_id"], h["조항"])) if mode == "검색" else "(검색 안 함)"
            for g in gold[gold["유형"] == t].itertuples():
                rows.append({"cot_id": r["cot_id"], "결함 유형": TYPE_LABEL[t], "판정 시점": when, "doc_id": g.doc_id, "조항": g.조항,
                             "문서 시행일": idx.get(g.doc_id, ""), "근거 출처": mode,
                             "검색 상위 3": top, "검색에 gold 포함": bool(h["gold 일치"].any()) if mode == "검색" and len(h) else "",
                             "검색 시각": h["검색 시각"].iloc[0] if len(h) else _now()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 전체 흐름
# ---------------------------------------------------------------------------

def run_live(base: dict, ids: list, live: bool = True, delay: float = 0.5):
    as_of, t0 = base["as_of"], time.time()
    yield {"step": 0, "kind": "msg", "text": f"분석 {len(ids)}건, 기준일 {as_of}, {'실시간' if live else '저장값'} 모드"}
    # 1
    docs = check_documents(as_of, live)
    yield {"step": 1, "kind": "table", "title": "기준 문서 검색", "data": docs.drop(columns=["URL"])}
    # 2
    rule = build_rule_table(docs, base["ctx"]["range"])
    yield {"step": 2, "kind": "warn" if rule["status"] != "일치" else "msg", "text": rule["message"]}
    yield {"step": 2, "kind": "table", "title": "성향별 허용 등급표", "data": rule["table"]}
    # 3
    rec = base["ctx"]["recs"]
    names = sorted(rec.loc[rec["cot_id"].isin(ids), "rec_name"].dropna().unique())
    yield {"step": 3, "kind": "msg", "text": f"추천 상품 {len(names)}개 조회 ({'0.5초 간격, 상품명만' if live else '저장값'})"}
    latest, msgs = None, []
    for ev in lookup_products(names, as_of, live, delay):
        if ev[0] == "progress":
            yield {"step": 3, "kind": "progress", "i": ev[1], "n": ev[2], "text": ev[3]}
        else:
            latest, msgs = ev[1], ev[2]
    for m in dict.fromkeys(msgs):
        yield {"step": 3, "kind": "warn", "text": m}
    dg = rec[rec["cot_id"].isin(ids)].drop_duplicates("rec_name").set_index("rec_name")["json_grade"]
    latest["데이터 등급"] = latest["상품명"].map(dg)
    diff = latest["판정 등급"].notna() & latest["데이터 등급"].notna() & (latest["판정 등급"] != latest["데이터 등급"])
    latest["등급 다름"] = diff
    yield {"step": 3, "kind": "msg",
           "text": f"최신 등급 있음 {int(latest['판정 등급'].notna().sum())}/{len(latest)}개, 데이터 등급과 다른 상품 {int(diff.sum())}개, "
                   f"사람 확인 등급 사용 {int(latest['사람 확인 등급'].notna().sum())}개 (실시간과 다름 {int(latest['실시간과 다름'].sum())}개)"}
    yield {"step": 3, "kind": "table", "title": "상품 등급 (G 규칙 적용)",
           "data": latest[["상품명", "데이터 등급", "최신 등급", "상태", "값 출처", "사람 확인 등급", "실시간과 다름", "판정 등급", "등급 다름"]]}
    # 5 (4에서 쓸 결함 유형을 먼저 정한다)
    res, recs = apply_rules(base, ids, rule["range"], latest)
    types = [t for t, on in [("범위", ((res["③ 데이터 기준"] == "범위 밖") | (res["③ 최신 문서 기준"] == "범위 밖")).any()),
                             ("설명", res["④"].isin(["일부 기재", "기재 없음"]).any()), ("이관", (res["이관"] == "결함").any()),
                             ("②-a", res["②"].str.startswith("②-a").any()), ("②-b", res["②"].str.startswith("②-b").any())] if on]
    # 4
    art = search_articles(types, as_of)
    yield {"step": 4, "kind": "warn" if art["mode"] != "검색" else "msg", "text": art["message"]}
    yield {"step": 4, "kind": "table", "title": "결함 유형별 근거 조항", "data": art["hits"]}
    # 5
    cd = cards(res, art)
    summ = pd.DataFrame({
        "지표": ["③ 범위 밖 CoT", "④ 위험등급 설명 누락 CoT", "이관 결함 CoT", "② 결함 CoT", "결함 CoT (하나라도)", "③ 판정이 바뀐 CoT"],
        "데이터 기준": [int((res["③ 데이터 기준"] == "범위 밖").sum()), int(res["④"].isin(["일부 기재", "기재 없음"]).sum()),
                   int((res["이관"] == "결함").sum()), int(res["②"].str.startswith("②-").sum()), int(res["결함(데이터 기준)"].sum()), ""],
        "최신 문서 기준": [int((res["③ 최신 문서 기준"] == "범위 밖").sum()), int(res["④"].isin(["일부 기재", "기재 없음"]).sum()),
                     int((res["이관"] == "결함").sum()), int(res["②"].str.startswith("②-").sum()), int(res["결함(최신 문서 기준)"].sum()),
                     int(res["③ 바뀜"].sum())]})
    summ[["데이터 기준", "최신 문서 기준"]] = summ[["데이터 기준", "최신 문서 기준"]].astype(str)
    yield {"step": 5, "kind": "msg", "text": f"판정 표: {rule['message']}"}
    yield {"step": 5, "kind": "table", "title": "판정 결과 (데이터 기준 vs 최신 문서 기준)", "data": res}
    yield {"step": 5, "kind": "table", "title": "근거 카드", "data": cd}
    secs = round(time.time() - t0, 1)
    yield {"step": 5, "kind": "done", "summary": summ, "result": res, "cards": cd, "latest": latest, "docs": docs,
           "rule": rule, "articles": art, "seconds": secs}


def save_run(ev: dict, as_of: str, n: int, live: bool) -> Path:
    """CLI 실행 결과를 outputs/live/에 저장 (cot_id·상품명·등급·조항만)."""
    d = OUT / "live" / as_of
    d.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}_{'live' if live else 'saved'}"
    ev["result"].to_csv(d / f"result_{tag}.csv", index=False, encoding="utf-8-sig")
    ev["cards"].to_csv(d / f"cards_{tag}.csv", index=False, encoding="utf-8-sig")
    ev["summary"].to_csv(d / f"summary_{tag}.csv", index=False, encoding="utf-8-sig")
    ev["latest"].to_csv(d / f"products_{tag}.csv", index=False, encoding="utf-8-sig")
    ev["docs"].to_csv(d / f"documents_{tag}.csv", index=False, encoding="utf-8-sig")
    return d
