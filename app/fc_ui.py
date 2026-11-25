"""FitCheck 화면 공용 부분 (로컬 앱 app/fitcheck_app.py와 외부용 app/fitcheck_public.py가 같이 쓴다).

시안 색·CSS·HTML 조각·숫자로 채우는 정해진 문장·저장 집계 읽기. 원문을 읽는 코드는 여기에 없다.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.live import load_base, pick_sample, run_live  # noqa: E402
from src.score import ITEMS, LAW, NAME, WEIGHT, item_pass, live_frame, summarize  # noqa: E402

OUT = ROOT / "outputs"
STEPS = {0: "준비", 1: "1. 기준 문서 검색", 2: "2. 규칙 생성", 3: "3. 상품 등급 검색", 4: "4. 근거 조항 검색", 5: "5. 규칙 적용"}

# ---------------------------------------------------------------------------
# 시안 색 (docs/design/diagnosis_mockup.html)
# ---------------------------------------------------------------------------
NAVY, BLUE, BG, LINE, INK, SUB, BODY = "#13213D", "#2F6FED", "#F4F6FA", "#DCE2EC", "#0F172A", "#4A5568", "#334155"
GRAY_BAR, TRACK, AMBER, BLUE_TXT, RED = "#94A3B8", "#EEF1F6", "#B45309", "#1D4ED8", "#B91C1C"
# st.html이 svg를 지우므로 아이콘은 글자로 (색과 글자를 같이 쓴다)
ICON_OK = '<span aria-hidden="true" style="font-weight:800">✓</span>'
ICON_WARN = '<span aria-hidden="true" style="font-weight:800">!</span>'
ICON_X = '<span aria-hidden="true" style="font-weight:800">✕</span>'
# 등급 배지: (배경, 글자, 아이콘). 색과 글자를 같이 쓴다
BADGE = {"우수": ("#DBEAFE", "#1E3A8A", ICON_OK), "양호": ("#DCFCE7", "#166534", ICON_OK),
         "보통": ("#FEF3C7", "#92400E", ICON_WARN), "미흡": ("#FFEDD5", "#9A3412", ICON_WARN),
         "저조": ("#FEE2E2", "#991B1B", ICON_X)}
BANDS = [("저조", 60, "#FCA5A5"), ("미흡", 10, "#FDBA74"), ("보통", 10, "#FCD34D"), ("양호", 10, "#86EFAC"), ("우수", 10, "#93C5FD")]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap');
/* 글꼴은 config.toml [theme]와 여기서. 아이콘 글꼴(Material Symbols)을 덮지 않도록 span 전체에는 주지 않는다 */
html, body, .stApp, [class^="fc-"], .stMarkdown, .stCaption, label, button p {{
  font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}}
/* 한글 줄바꿈: 단어 중간에서 줄이 바뀌지 않게 */
html, body, .stApp, p, div, span, label, td, th, h1, h2, h3, li {{
  word-break: keep-all; overflow-wrap: break-word; text-wrap: pretty; line-break: strict;
}}
.stApp {{ background: {BG}; color: {INK}; }}
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ max-width: 1200px; padding-top: 1.2rem; padding-bottom: 3rem; }}
/* 탭 = 시안 nav */
.stTabs [role="tablist"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; padding: 0 16px; gap: 4px; }}
.stTabs [data-testid="stTab"] {{ padding: 14px 18px; height: auto; border-bottom: 3px solid transparent; }}
.stTabs [data-testid="stTab"] p {{ font-size: 15px; color: {SUB}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ border-bottom-color: {BLUE}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] p {{ color: {NAVY}; font-weight: 700; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF; }}
.fc-head {{ background: {NAVY}; color: #FFF; padding: 20px 32px; border-radius: 14px; display: flex; flex-wrap: wrap; gap: 16px;
  align-items: center; justify-content: space-between; }}
.fc-logo {{ width: 40px; height: 40px; border-radius: 10px; background: {BLUE}; display: flex; align-items: center; justify-content: center; }}
.fc-chip {{ background: #1F3159; padding: 6px 12px; border-radius: 999px; font-size: 13px; }}
.fc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 4px 0 20px; }}
.fc-grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin: 4px 0 20px; }}
.fc-card {{ background: #FFF; border: 1px solid {LINE}; border-radius: 14px; padding: 22px 24px; display: flex; flex-direction: column; gap: 10px; }}
.fc-dark {{ background: {NAVY}; color: #FFF; border-radius: 14px; padding: 22px 24px; display: flex; flex-direction: column; gap: 12px; }}
.fc-label {{ font-size: 14px; color: {SUB}; font-weight: 500; }}
.fc-dark .fc-label, .fc-dark .fc-note {{ color: #B8C4DC; }}
.fc-big {{ font-size: 56px; font-weight: 800; letter-spacing: -1.5px; line-height: 1; }}
.fc-badge {{ display: inline-flex; align-items: center; gap: 6px; font-weight: 700; font-size: 15px; padding: 5px 12px; border-radius: 8px; }}
.fc-sent {{ font-size: 14px; color: {BODY}; line-height: 1.55; }}
.fc-op {{ font-size: 19px; font-weight: 700; line-height: 1.45; }}
.fc-note {{ font-size: 13px; line-height: 1.5; color: {SUB}; }}
.fc-sec {{ background: #FFF; border: 1px solid {LINE}; border-radius: 14px; padding: 20px 24px; display: flex; flex-direction: column; gap: 12px;
  margin: 0 0 20px; }}
.fc-h2 {{ margin: 0; font-size: 17px; font-weight: 700; color: {INK}; }}
.fc-row {{ display: grid; grid-template-columns: minmax(180px, 220px) minmax(0, 1fr) minmax(220px, 320px); gap: 20px; align-items: center;
  padding: 14px 0; border-top: 1px solid {TRACK}; }}
.fc-track {{ flex: 1; height: 12px; background: {TRACK}; border-radius: 6px; overflow: hidden; }}
.fc-num {{ width: 92px; font-size: 13px; text-align: right; }}
.fc-step {{ width: 28px; height: 28px; border-radius: 8px; background: {NAVY}; color: #FFF; font-weight: 800; font-size: 14px;
  display: flex; align-items: center; justify-content: center; flex: none; }}
.fc-foot {{ font-size: 12px; color: {SUB}; line-height: 1.6; padding: 0 4px; }}
.fc-stat {{ font-size: 32px; font-weight: 800; letter-spacing: -0.8px; line-height: 1.1; }}
</style>
"""


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def _csv(p: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig", **kw)


def guide_version() -> str:
    m = re.search(r"\(v([\d.]+),", (ROOT / "docs" / "판정기준서.md").read_text(encoding="utf-8").splitlines()[0])
    return m.group(1) if m else "?"


@st.cache_resource(show_spinner="데이터 판정 준비 중 (약 30초)…")
def base_for(as_of: str) -> dict:
    return load_base(as_of)


@st.cache_data
def facts(_mtime: float) -> dict:
    """화면 문장에 쓰는 저장 집계 (원문 없음). 없으면 빈 값."""
    f = {}
    p = OUT / "score_cot.csv"
    f["corr_pass"] = _csv(p).query("판 == '교정판'").set_index("cot_id") if p.exists() else None
    lp = OUT / "corrections_log.csv"
    f["n_add"] = int((_csv(lp, dtype=str, keep_default_na=False)["위치"] == "answer 위험등급 문장").sum()) if lp.exists() else 0
    dp = OUT / "d4_cause_summary.csv"
    if dp.exists():
        d = _csv(dp, dtype=str, keep_default_na=False)
        d["n"] = pd.to_numeric(d["추천_전체"])
        miss = d[d["answer"] != "있음"]
        f["d4_cot3"] = round(miss.loc[miss["cot3"] == "있음", "n"].sum() / miss["n"].sum() * 100) if len(miss) else None
    sc = _csv(OUT / "scorecard.csv", dtype=str, keep_default_na=False)
    g = sc[sc["구분"].str.startswith("G 표본")]
    row = g[g["지표"].str.startswith("보고 문구")]
    m = re.search(r"최소 약 ([\d,]+)건.*CI ([\d,]+)[–~-]([\d,]+)", row["지표"].iloc[0]) if len(row) else None
    f["g_est"] = m.groups() if m else None
    ch = g[g["지표"].str.startswith("등급 변경 비율")]
    f["g_change"] = ch["전체"].iloc[0] if len(ch) else None
    ns = g[g["지표"].str.startswith("표본 상품 수")]
    f["g_n"] = int(float(ns["전체"].iloc[0])) if len(ns) else None
    return f


# ---------------------------------------------------------------------------
# HTML 조각 (시안 구성)
# ---------------------------------------------------------------------------

def html(s: str) -> None:
    st.html(s)


def badge(g: str) -> str:
    bg, fg, ic = BADGE.get(g, ("#EEF1F6", BODY, ""))
    return f'<span class="fc-badge" style="background:{bg};color:{fg}">{ic}{escape(g)}</span>'


def score_card(label: str, s: dict, sentence: str, delta: float | None = None) -> str:
    d = ""
    if delta is not None:
        d = f'<span style="font-size:15px;font-weight:700;color:{BLUE_TXT if delta >= 0 else RED}">{delta:+.1f}</span>'
    return (f'<div class="fc-card"><div class="fc-label">{escape(label)}</div>'
            f'<div style="display:flex;align-items:baseline;gap:10px"><span class="fc-big">{s["점수"]:.1f}</span>'
            f'<span style="font-size:16px;color:{SUB}">/ 100점</span>{d}</div>'
            f'<div>{badge(s["등급"])}</div><div class="fc-sent">{escape(sentence)}</div></div>')


def dark_card(title: str, body: str, note: str = "") -> str:
    n = f'<div class="fc-note">{escape(note)}</div>' if note else ""
    return f'<div class="fc-dark"><div class="fc-label">{escape(title)}</div><div class="fc-op">{escape(body)}</div>{n}</div>'


def band(marks: list) -> str:
    """marks: [(이름, 점수, 등급)]. 두 점수가 가까우면 표시를 위아래로 나눈다."""
    marks = sorted(marks, key=lambda m: m[1])
    tops, pins = [], ""
    for i, (nm, sc, g) in enumerate(marks):
        top = 16 if i and abs(sc - marks[i - 1][1]) < 12 and tops[-1] == 0 else 0
        tops.append(top)
        pins += (f'<div style="position:absolute;top:{top}px;left:{max(min(sc, 97), 3)}%;transform:translateX(-50%);font-size:12px;'
                 f'font-weight:700;color:{BADGE[g][1]};white-space:nowrap">{escape(nm)} {sc:.1f} ▾</div>')
    cols = " ".join(f"{w}fr" for _, w, _ in BANDS)
    segs = "".join(f'<div style="background:{c};{"border-radius:7px 0 0 7px;" if i == 0 else ""}'
                   f'{"border-radius:0 7px 7px 0;" if i == len(BANDS) - 1 else ""}"></div>' for i, (_, _, c) in enumerate(BANDS))
    labels = ["저조 · 60 미만", "미흡 · 60", "보통 · 70", "양호 · 80", "우수 · 90"]
    labs = "".join(f"<div>{t}</div>" for t in labels)
    pad = 46 if max(tops) else 30
    return (f'<div class="fc-sec"><div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:baseline">'
            f'<h2 class="fc-h2">등급 구간</h2><span style="font-size:13px;color:{SUB}">금융감독원 미스터리쇼핑 5단계 등급 차용</span></div>'
            f'<div style="position:relative;padding-top:{pad}px">{pins}'
            f'<div style="display:grid;grid-template-columns:{cols};gap:3px;height:14px">{segs}</div>'
            f'<div style="display:grid;grid-template-columns:{cols};gap:3px;margin-top:6px;font-size:12px;color:{BODY}">{labs}</div>'
            f'</div></div>')


def loss_color(loss: float) -> str:
    return RED if loss > 10 else (AMBER if loss > 2 else BODY)


def item_bars(title: str, a: dict, b: dict, names: tuple, why: dict) -> str:
    """항목별 막대. a = 회색(원본·데이터 기준), b = 파랑(교정판·최신 문서 기준). 오른쪽 감점·이행률은 a 기준."""
    ta, tb = a["항목"].set_index("키"), b["항목"].set_index("키")
    leg = (f'<span style="display:inline-flex;align-items:center;gap:6px"><span style="width:14px;height:10px;border-radius:3px;'
           f'background:{GRAY_BAR}"></span>{escape(names[0])}</span>'
           f'<span style="display:inline-flex;align-items:center;gap:6px"><span style="width:14px;height:10px;border-radius:3px;'
           f'background:{BLUE}"></span>{escape(names[1])}</span>')
    rows = ""
    for k, n, w, law in ITEMS:
        ra, rb = ta.loc[k], tb.loc[k]
        bars = ""
        for r, col, num in ((ra, GRAY_BAR, f"color:{BODY}"), (rb, BLUE, "color:#1E3A8A;font-weight:700")):
            pct = 0 if pd.isna(r["이행률"]) else r["이행률"]
            bars += (f'<div style="display:flex;align-items:center;gap:10px"><div class="fc-track"><div style="height:12px;'
                     f'background:{col};border-radius:6px;width:{pct}%"></div></div>'
                     f'<span class="fc-num" style="{num}">{r["평균 득점"]:.1f} / {w:g}</span></div>')
        rows += (f'<div class="fc-row"><div style="display:flex;flex-direction:column;gap:4px">'
                 f'<div style="font-size:15px;font-weight:700">{escape(n)}</div>'
                 f'<div style="font-size:12px;color:{SUB}">배점 {w:g}점 · {escape(law)}</div></div>'
                 f'<div style="display:flex;flex-direction:column;gap:6px">{bars}</div>'
                 f'<div style="display:flex;flex-direction:column;gap:4px">'
                 f'<div style="font-size:13px;font-weight:700;color:{loss_color(ra["감점"])}">감점 {ra["감점"]:.1f}점 · 이행 {ra["이행률"]:.1f}%</div>'
                 f'<div style="font-size:13px;color:{BODY};line-height:1.5">{escape(why.get(k, ""))}</div></div></div>')
    return (f'<div class="fc-sec" style="gap:0;padding:22px 24px"><div style="display:flex;flex-wrap:wrap;justify-content:space-between;'
            f'gap:8px;align-items:baseline;padding-bottom:14px"><h2 class="fc-h2">{escape(title)}</h2>'
            f'<div style="display:flex;gap:16px;font-size:13px;color:{BODY}">{leg}</div></div>{rows}</div>')


def todo_cards(todos: list) -> str:
    cards = ""
    for i, (t, body, status, color) in enumerate(todos, 1):
        cards += (f'<div class="fc-card" style="padding:20px 22px;gap:8px"><div style="display:flex;align-items:center;gap:10px">'
                  f'<span class="fc-step">{i}</span><span style="font-size:15px;font-weight:700">{escape(t)}</span></div>'
                  f'<div style="font-size:13px;color:{BODY};line-height:1.55">{escape(body)}</div>'
                  f'<div style="font-size:12px;font-weight:700;color:{color}">{escape(status)}</div></div>')
    return f'<h2 class="fc-h2" style="margin:4px 0 10px">먼저 할 일</h2><div class="fc-grid3">{cards}</div>'


def stat_cards(stats: list) -> str:
    return '<div class="fc-grid">' + "".join(
        f'<div class="fc-card" style="padding:18px 22px;gap:6px"><div class="fc-label">{escape(a)}</div>'
        f'<div class="fc-stat">{b}</div><div class="fc-note">{escape(c)}</div></div>' for a, b, c in stats) + "</div>"


# ---------------------------------------------------------------------------
# 정해진 문장 (숫자로 채운다, LLM 아님)
# ---------------------------------------------------------------------------
SHORT = {"①": "필수정보 누락", "②": "성향 근거 부족", "③": "부적합 추천", "이관": "이관 누락", "④": "위험등급 설명 누락"}
WORST = {"①": "상담 10건 중 {k}건이 법정 필수정보를 다 확인하지 않습니다.",
         "②": "상담 10건 중 {k}건이 근거 없이 성향을 정하거나 성향 없이 추천합니다.",
         "③": "상담 10건 중 {k}건이 성향보다 위험한 상품을 추천합니다.",
         "이관": "상담 10건 중 {k}건이 기준 초과를 알리지 않고 추천합니다.",
         "④": "답변 10건 중 {k}건이 상품의 위험등급을 제대로 설명하지 않습니다."}


def counts(j: pd.DataFrame) -> dict:
    s = lambda c: j[c].fillna("").astype(str)  # noqa: E731
    v4 = s("기준④ 판정"); n4 = int(v4.isin(["전부 기재", "일부 기재", "기재 없음"]).sum()) or 1
    return {"①": int(s("기준① 법정 미확인").ne("").sum()), "②a": int(s("기준② 판정").str.startswith("②-a").sum()),
            "②b": int(s("기준② 판정").str.startswith("②-b").sum()), "③": int(s("기준③ 판정").str.startswith("범위 밖").sum()),
            "이관": int(s("이관 판정").eq("결함").sum()), "④없음": round(v4.eq("기재 없음").sum() / n4 * 100),
            "④일부": round(v4.eq("일부 기재").sum() / n4 * 100)}


def worst_sentence(s: dict) -> str:
    t = s["항목"].sort_values("감점", ascending=False).iloc[0]
    miss = 1 - t["이행률"] / 100
    k = round(miss * 10)
    if k == 0:
        return WORST[t["키"]].replace("10건 중 {k}건", f"100건 중 {round(miss * 100)}건")
    return WORST[t["키"]].format(k=k)


def why_lines(c: dict, corr: dict | None) -> dict:
    cm = corr["항목"].set_index("키")["미이행 건수"] if corr else None
    w3 = "교정판에서는 제외" if cm is not None and cm["③"] == 0 else (f"교정판에 {cm['③']:,}건 남음" if cm is not None else "")
    wt = "교정판에서는 0건" if cm is not None and cm["이관"] == 0 else ""
    return {"①": f"{c['①']:,}건이 목적·재산·경험 중 하나를 확인하지 않음. 교정으로 채울 수 없음",
            "②": f"{c['②a'] + c['②b']:,}건이 근거 없이 성향을 확정({c['②a']:,})하거나 성향 없이 추천({c['②b']:,})",
            "③": f"{c['③']:,}건이 성향보다 위험한 상품 추천" + (f". {w3}" if w3 else ""),
            "이관": f"{c['이관']:,}건이 기준 초과를 알리지 않고 그대로 추천" + (f". {wt}" if wt else ""),
            "④": f"{c['④없음']}%가 위험등급을 아예 설명하지 않고 {c['④일부']}%는 일부만 설명"}


def opinion(o: dict, cr: dict, n1_corr: int) -> str:
    ok = ("우수", "양호")
    s1 = "원본은 그대로 학습에 쓰면 안 됩니다." if o["등급"] not in ok else "원본도 학습에 쓸 수 있는 수준입니다."
    if cr["등급"] in ok:
        s2 = (f"교정판은 학습에 쓸 수 있지만, 필수정보가 빠진 {n1_corr:,}건은 데이터를 다시 받아야 합니다." if n1_corr
              else "교정판은 학습에 쓸 수 있습니다.")
    else:
        s2 = f"교정판도 아직 {cr['등급']} 등급이라 학습에 쓰기 전에 남은 결함을 고쳐야 합니다."
    return f"{s1} {s2}"


def corr_sentence(cr: dict, n_add: int, n_ex: int) -> str:
    t = cr["항목"].sort_values("감점", ascending=False)
    t = t[t["감점"] >= 1]
    rest = f" {SHORT[t.iloc[0]['키']]}은 남아 있습니다." if len(t) else ""
    return f"위험등급 문장 {n_add:,}개를 넣고 부적합 추천이 있는 {n_ex:,}건을 뺐습니다.{rest}"


def todos(o: dict, c: dict, f: dict) -> list:
    t = o["항목"].sort_values("감점", ascending=False).head(3).reset_index(drop=True)
    out = []
    for i, r in t.iterrows():
        k, loss = r["키"], r["감점"]
        lead = f"감점이 가장 큰 항목 (−{loss:.1f}점)." if i == 0 else ""
        tail = "" if i == 0 else f" (−{loss:.1f}점)"
        if k == "④":
            d4 = f" 등급은 추론 단계에 있다가 답변 정리 단계에서 빠졌습니다({f['d4_cot3']}%)." if f.get("d4_cot3") is not None else ""
            out.append(("답변에 위험등급 문장 넣기", (lead or f"감점 −{loss:.1f}점.") + d4,
                        f"교정판에서 해결됨 · 문장 {f['n_add']:,}개 추가", BLUE_TXT))
        elif k == "①":
            out.append(("빠진 고객 정보 다시 받기", f"{lead} {c['①']:,}건이 목적·재산·경험 중 하나를 확인하지 않았습니다{tail}. "
                        "데이터만으로는 채울 수 없습니다.".strip(), "데이터 제공처에 재요청 필요", AMBER))
        elif k == "②":
            out.append(("근거 없는 성향 판단 표시하기", f"{lead} {c['②a'] + c['②b']:,}건이 근거 없이 성향을 확정하거나 성향 없이 "
                        f"추천했습니다{tail}.".strip(), "태그로 표시, 학습 제외 검토", AMBER))
        elif k == "③":
            g = (f" 표본 {f['g_n']}개 중 등급을 찾은 상품의 {f['g_change']}%가 데이터와 달랐습니다." if f.get("g_change") else "")
            out.append(("최신 상품 등급으로 다시 검사", f"{lead} {c['③']:,}건이 성향보다 위험한 상품을 추천했습니다{tail}.{g}".strip(),
                        "실시간 분석 탭에서 확인", AMBER))
        else:
            out.append(("부적합 안내 누락 표시하기", f"{lead} {c['이관']:,}건이 기준 초과를 알리지 않고 추천했습니다{tail}.".strip(),
                        "태그로 표시", AMBER))
    return [(a, b.strip(), s, col) for a, b, s, col in out]


FOOT = ("채점 방식: 금융감독원 미스터리쇼핑 방식 차용. CoT 1건을 판매 상담 1회로 보고, 항목을 지키면 배점을 주고 못 지키면 0점(일부 이행도 0점). "
        "측정하는 5개 항목(성향 파악 영역 4개 각 7.5점, 투자 위험 설명 10점)을 100점으로 환산했습니다. 판정할 수 없는 항목은 그 CoT에서 빼고 "
        "환산합니다 (판정기준서 v{v} 11장, D70). 데이터: AI Hub 「금융상품·서비스 및 소비자 특성 데이터」(한국지능정보사회진흥원). "
        "AI 학습용 생성 데이터이며 실제 고객 행동이 아닙니다.")


