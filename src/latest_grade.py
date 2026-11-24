"""G 규칙 (판정기준서 v1.1 6장, D64): 추천 상품의 최신 공시 등급 조회와 재판정.

- 조회에는 상품명만 쓴다 (원문 문장은 밖으로 나가지 않는다, P10).
- G2 출처 순서: 금융투자협회 전자공시 → 운용사 공식 페이지·투자설명서 → 판매사·펀드 정보 사이트(참고).
  2026-11-23 확인: 협회 전자공시의 펀드 기본정보 화면·응답에는 투자위험등급 항목이 없다 (투자설명서 PDF에만 있음).
  이번 조회는 펀드 정보 사이트 FunETF(3순위, 참고)의 '투자위험' 값을 쓰고, 출처 순위를 결과에 남긴다.
- 조회 결과는 outputs/latest_grade_check.csv에 저장하고, 재판정(G4)은 저장된 표만 읽는다 (재실행 시 웹 조회 없음).
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher

import pandas as pd

BASE = "https://www.funetf.co.kr"
HEAD = {"User-Agent": "Mozilla/5.0 (research script)"}
LABEL = {1: "매우높은위험", 2: "높은위험", 3: "다소높은위험", 4: "보통위험", 5: "낮은위험", 6: "매우낮은위험"}


def _norm(s: str) -> str:
    """이름 비교용 정규화: 공백·괄호·기호, '형', '제·호', 사이트가 붙이는 '운용'·'(모)'·'종류' 표기를 지운다."""
    s = str(s).replace("(운용)", "").replace("_운용", "").replace("(모)", "").replace("(종류)", "").replace("(운)", "")
    s = re.sub(r"운용$", "", s.strip())
    s = re.sub(r"제\s*(\d+)\s*호", r"\1", s)
    return re.sub(r"[\s\-_·\[\]\(\)형]", "", s).lower()


def split_class(name: str) -> tuple[str, str]:
    """데이터 상품명 → (기본 이름, 클래스). 클래스는 마지막 괄호 안 가장 안쪽 값 (예: '(수수료미징구-온라인(C-E))' → 'C-E').
    애매한 경우 처리: 괄호가 없으면 클래스 ''."""
    name = str(name)
    inner = re.findall(r"\(([^()]*)\)", name)
    if not inner or not name.rstrip().endswith(")"):
        return name, ""
    cls = inner[-1]
    base = name[: name.rfind("(" + cls + ")")].rstrip()
    base = re.sub(r"\([^()]*$", "", base).rstrip()   # '(수수료미징구-온라인' 같은 바깥 괄호 앞부분 제거
    return base, cls


def _terms(base: str) -> list:
    """검색어 후보: 공백 없는 이름 앞 10자, 공백·괄호로 나눈 4자 이상 조각(긴 순)."""
    toks = [t for t in re.split(r"[\s\[\]\(\)]+", base) if len(t) >= 4]
    return [re.sub(r"\s+", "", base)[:10]] + sorted(toks, key=len, reverse=True)[:3]


def _search(client, term: str) -> list:
    r = client.get(f"{BASE}/api/public/main/search/all",
                   params={"schVal": term, "reSchVal": "", "schKeyword": "", "schStockItem": ""}, headers=HEAD, timeout=30)
    return r.json().get("fundList", {}).get("content", [])


def _classes(client, code: str) -> list:
    """대표 펀드 화면의 클래스 목록 [(코드, 클래스 표시)]."""
    html = client.get(f"{BASE}/product/fund/view/{code}", headers=HEAD, timeout=30).text
    out = []
    for c, txt in re.findall(r'class="class-item" href="/product/fund/view/([0-9A-Z]{12})"[^>]*>(.*?)</a>', html, re.S):
        out.append((c, re.sub(r"<[^>]+>|\s+", " ", txt).strip()))
    return out


def _grade(client, code: str) -> tuple[float, str]:
    html = client.get(f"{BASE}/product/fund/view/{code}", headers=HEAD, timeout=30).text
    name = re.search(r'prd-info-top-title">([^<]+)<', html)
    g = re.search(r"투자위험\s*(?:<[^>]+>\s*)*([1-6])등급", html)
    return (float(g.group(1)) if g else float("nan")), (name.group(1).strip() if name else "")


def lookup(names: list, checked: str, delay: float = 0.5) -> pd.DataFrame:
    """상품명 목록을 조회한다. 반환 열: 상품명, 조회 이름, 조회 클래스, 최신 등급, 상태, 출처 순위, URL, 확인일.
    애매한 경우 처리:
    - 펀드가 아닌 상품(파생결합증권 등)은 펀드 공시 대상이 아니므로 조회하지 않고 '최신 확인 불가' (G5).
    - 검색은 기본 이름 전체 → 앞 12자 순으로 하고, 결과 중 기본 이름과 정규화 이름이 같은 펀드만 '일치'로 본다.
      이름이 바뀐 것으로 보이는 경우(정규화 이름이 다름)는 자동으로 같은 상품으로 보지 않고 '사람 확인(명칭 다름)'.
    - 클래스가 목록에 없으면 같은 펀드 대표 클래스 등급을 쓰고 '클래스 다름' (G3).
    delay: 요청 사이 최소 간격(초). 상위 30 조회는 0.4~0.5초, G 표본(D67)은 1초 이상.
    """
    import httpx
    rows = []
    with httpx.Client(follow_redirects=True) as cl:
        for nm in names:
            base, cls = split_class(nm)
            row = {"상품명": nm, "조회 이름": "", "조회 클래스": "", "최신 등급": float("nan"), "상태": "", "출처 순위": "",
                   "URL": "", "확인일": checked}
            if "파생결합증권" in nm:
                row["상태"] = "최신 확인 불가(펀드 공시 대상 아님)"; rows.append(row); continue
            hits = {}
            for term in _terms(base):
                for h in _search(cl, term):
                    hits[h["repFundCd"]] = h
                time.sleep(max(0.4, delay))
            nb = _norm(base)
            same = [h for h in hits.values() if _norm(h["repFundNm"]) == nb]
            near = sorted(hits.values(), key=lambda h: -SequenceMatcher(None, _norm(h["repFundNm"]), nb).ratio())
            if not same:
                if near and SequenceMatcher(None, _norm(near[0]["repFundNm"]), nb).ratio() >= 0.85:
                    row.update({"상태": "사람 확인(명칭 다름)", "조회 이름": near[0]["repFundNm"]})
                else:
                    row["상태"] = "최신 확인 불가(검색 결과 없음)"
                rows.append(row); continue
            rep = same[0]
            cl_list = _classes(cl, rep["fundCd"]); time.sleep(max(0.5, delay))
            want = _norm(cls)
            lab = lambda t: _norm(re.sub(r"^(Class|종류)\s*", "", t.split(" - ")[0].strip()))
            pick = [c for c, t in cl_list if lab(t) == want] if want else []
            code = pick[0] if pick else rep["fundCd"]
            g, shown = _grade(cl, code); time.sleep(max(0.5, delay))
            row.update({"조회 이름": shown, "조회 클래스": cls if pick else rep.get("classNm", ""), "최신 등급": g,
                        "상태": "확인" if pick else "클래스 다름", "출처 순위": "3 펀드 정보 사이트(참고)",
                        "URL": f"{BASE}/product/fund/view/{code}"})
            rows.append(row)
    return pd.DataFrame(rows)


def g4_rejudge(recs: pd.DataFrame, cots: pd.DataFrame, rng: dict, latest: pd.DataFrame) -> pd.DataFrame:
    """G4: 최신 등급으로 범위를 다시 판정한다. 추천 행(recs: cot_id, rec_name, json_grade)과 CoT 성향(성향_D21)을 쓴다.
    반환: 조회 대상 상품을 추천한 행마다 데이터 시점·최신 시점 범위 판정.
    애매한 경우 처리: 최신 등급이 없거나(확인 불가·사람 확인) 성향을 알 수 없으면 '재판정 불가'. 클래스 다름도 등급은 쓰되 표시를 남긴다.
    """
    lt = latest.set_index("상품명")
    r = recs[recs["rec_name"].isin(lt.index)].copy()
    r["성향"] = r["cot_id"].map(cots["성향_D21"])
    r["최신 등급"] = r["rec_name"].map(lt["최신 등급"])
    r["조회 상태"] = r["rec_name"].map(lt["상태"])

    def judge(g, p):
        if pd.isna(g) or p not in rng:
            return "재판정 불가"
        return "범위 안" if int(g) in rng[p] else "범위 밖"
    r["데이터 시점"] = [judge(g, p) for g, p in zip(r["json_grade"], r["성향"])]
    r["최신 시점"] = [judge(g, p) for g, p in zip(r["최신 등급"], r["성향"])]
    r["G4 태그"] = (r["데이터 시점"] == "범위 안") & (r["최신 시점"] == "범위 밖")
    r["G4 태그"] = r["G4 태그"].map({True: "최신 기준 부적합", False: ""})
    return r[["cot_id", "rec_name", "성향", "json_grade", "최신 등급", "조회 상태", "데이터 시점", "최신 시점", "G4 태그"]]
