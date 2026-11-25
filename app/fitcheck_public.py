"""FitCheck 외부용 시연 앱 (보고서 스타일, 탭 3개). 실행: run_public.ps1 (앱 127.0.0.1:8501 + ngrok 고정 도메인).

로컬 앱(app/fitcheck_app.py)은 그대로 두고, 판정·점수 함수(src/)와 데이터 준비(app/fc_ui.base_for 등)만 같이 쓴다.
화면 디자인(색·글꼴·카드·표)은 이 파일의 CSS가 정한다. 테마 색은 run_public.ps1의 --theme.* 옵션으로 준다
(.streamlit/config.toml은 로컬 앱 테마라서 바꾸지 않는다).

탭
- 진단 결과: 저장된 전체 결과 (품질 점수 원본 → 교정판, 항목별 감점 원본 vs 교정판, 주요 지표)
- 실시간 검색: 표본 30건 (CLI `python -m src.fitcheck live --n 30`과 같은 pick_sample 고정 시드) 단계별 진행
- 사례 보기: 시연용 cot_id 6건, 점수·항목별 판정·근거 조항·시행일·추천 상품, 개별 실시간 검색

공개 범위
- 보이는 것: 점수, 판정, 근거 조항(요지)·시행일, 상품명과 데이터·최신 등급, 실시간 검색 단계 기록
- 빼는 것: 원문(question·cot·answer)을 읽는 코드 자체가 없다. 파일 경로·주소·예외 이름은 clean()이 지우고,
  오류는 일반 문구로만 보인다 (run_public.ps1의 --client.showErrorDetails none). 내려받기 없음.
- 시작 화면 비밀번호 (.streamlit/secrets.toml의 public_password, git 제외). 세션당 5번 틀리면 1분 잠금.

애매한 경우 처리
- 30건 실시간 검색은 앱 전체에서 동시에 1개만, 끝난 뒤 5분 간격. 기다리는 사람에게는 남은 시간과 마지막 결과를 보여 준다.
- 판정(⑤)은 run_live가 30건을 한 번에 계산한다. 표에는 계산이 끝난 뒤 1건씩 추가해 보여 준다 (건별 계산이 아님).
- 중간에 실패하면 거기까지 그린 행은 두고, 나머지 행은 저장값 실행 결과로 채워 '저장값'으로 표시한다.
- 행의 출처: 그 상담의 추천 상품을 모두 실시간으로 조회했으면 '실시간', 하나라도 저장값이면 '저장값'.
- 사례 보기의 개별 실시간 검색은 세션당 1분에 1번, 30건 검색이 도는 동안에는 막는다.
"""
from __future__ import annotations

import hmac
import re
import sys
import threading
import time
from datetime import datetime
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_ui import ITEMS, ROOT, _csv, base_for, facts, item_pass, live_frame, pick_sample, run_live, st, summarize  # noqa: E402
import pandas as pd  # noqa: E402
from src.score import cot_scores  # noqa: E402

AS_OF = "2026-11-24"
OUT = ROOT / "outputs"
N_BATCH = 30
BATCH_GAP = 300        # 30건 실시간 검색: 끝난 뒤 다음 실행까지 (초, 앱 전체)
CASE_GAP = 60          # 사례 실시간 검색 간격 (초, 세션당)
MAX_FAIL, LOCK = 5, 60
DEMO = {1547: "최신 등급으로 ③ 판정이 바뀜", 2365: "범위 밖 추천 + 이관 결함", 43: "위험등급 설명 누락",
        82: "모든 항목 이행", 145: "필수정보 일부 미확인", 5851: "근거 부족으로 성향 확정"}
STEPS = {1: "① 기준 문서 검색", 2: "② 규칙 생성", 3: "③ 상품 등급 조회", 4: "④ 근거 조항 검색", 5: "⑤ 판정"}
SHORT = {"CO-KIS": "KIS 준칙", "LAW-FCPA": "금소법", "REG-KOFIA": "표준준칙"}
GENERIC = "일시적인 문제로 화면을 표시하지 못했습니다. 잠시 뒤 다시 시도해 주세요."
SOURCE = ("출처: AI Hub 「금융상품·서비스 및 소비자 특성 데이터」(한국지능정보사회진흥원) · 한국투자증권 투자권유준칙(2026-09-21 시행) · "
          "금융소비자 보호에 관한 법률. AI 학습용 생성 데이터이며 실제 고객 행동이 아닙니다.")
OR, OR2, INK, CARD, LINE, BEIGE, RES = "#FF5A1F", "#FF7A45", "#111111", "#FBF9F3", "#D8D1C2", "#CBC2B0", "#C2410C"

CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp td, .stApp th, .stApp summary,
.stApp h1, .stApp h2, .stApp h3, .fc, .fc div, .fc span {{ font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; }}
.stApp p, .stApp div, .stApp td, .stApp th, .stApp li, .stApp label, .stApp summary {{ word-break: keep-all; overflow-wrap: break-word; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}
.fc-top {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 12px 24px; flex-wrap: wrap;
  border-bottom: 2px solid {INK}; padding-bottom: 14px; margin-bottom: 14px; }}
.fc-kicker {{ font-size: 12px; font-weight: 700; letter-spacing: 2px; color: {INK}; }}
.fc-title {{ font-size: 30px; font-weight: 900; letter-spacing: -0.8px; line-height: 1.25; margin-top: 6px; color: {INK}; }}
.fc-date {{ font-size: 14px; font-weight: 700; text-align: right; color: {INK}; }}
.fc-sec {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 24px 0 12px; font-size: 18px; font-weight: 900;
  letter-spacing: -0.3px; color: {INK}; }}
.fc-chip {{ display: inline-block; background: {OR}; color: {INK}; font-weight: 700; font-size: 13px; padding: 3px 8px; border-radius: 0;
  letter-spacing: 0; line-height: 1.4; }}
.fc-chip.k {{ background: {INK}; color: #FFFFFF; }}
.fc-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 4px 0 8px; }}
.fc-kpi {{ background: {INK}; color: #FFFFFF; border-top: 3px solid {OR}; padding: 14px 16px; }}
.fc-kpi .l {{ font-size: 13px; color: #E8E2D4; font-weight: 500; }}
.fc-kpi .v {{ font-size: 32px; font-weight: 900; color: {OR2}; letter-spacing: -1px; line-height: 1.15; margin-top: 6px; }}
.fc-kpi .v small {{ font-size: 15px; font-weight: 700; color: #FFFFFF; letter-spacing: 0; margin-left: 4px; }}
.fc-kpi .s {{ font-size: 12.5px; color: #BDB5A4; margin-top: 6px; line-height: 1.5; }}
.fc-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 0; padding: 16px 18px; margin: 0 0 12px; color: {INK}; }}
.fc-card p {{ margin: 0 0 6px; font-size: 14.5px; line-height: 1.65; }}
.fc-note {{ font-size: 13px; color: #5A5446; line-height: 1.6; }}
.fc-scroll {{ overflow-x: auto; }}
.fc-tbl {{ width: 100%; border-collapse: collapse; font-size: 14px; color: {INK}; }}
.fc-tbl th {{ text-align: left; font-weight: 700; font-size: 13px; padding: 8px 10px; border-bottom: 1px solid {INK}; white-space: nowrap; }}
.fc-tbl td {{ padding: 8px 10px; border-bottom: 1px solid {LINE}; vertical-align: top; }}
.fc-tbl td.res {{ font-weight: 700; color: {RES}; }}
.fc-tbl td.num {{ text-align: right; white-space: nowrap; }}
.fc-tbl tr.hl td {{ background: #FFE9DF; }}
.fc-ok {{ color: {INK}; font-weight: 900; }}
.fc-ng {{ color: {OR}; font-weight: 900; }}
.fc-na {{ color: #8A8272; }}
.fc-src {{ display: inline-block; font-size: 12px; font-weight: 700; padding: 1px 6px; border: 1px solid {INK}; }}
.fc-src.saved {{ background: {BEIGE}; }}
.fc-bars {{ display: grid; gap: 14px; }}
.fc-bar-row {{ display: grid; grid-template-columns: minmax(130px, 190px) minmax(0, 1fr); gap: 6px 14px; align-items: center; }}
.fc-bar-name {{ font-size: 14px; font-weight: 700; }}
.fc-bar-name small {{ display: block; font-size: 12px; font-weight: 400; color: #5A5446; }}
.fc-bar-line {{ display: grid; grid-template-columns: 56px minmax(0, 1fr) 64px; gap: 8px; align-items: center; font-size: 12.5px; }}
.fc-bar-track {{ height: 14px; background: transparent; border-left: 1px solid {INK}; }}
.fc-bar-fill {{ height: 14px; }}
.fc-legend {{ display: flex; gap: 16px; font-size: 13px; margin-bottom: 12px; flex-wrap: wrap; }}
.fc-legend span::before {{ content: ""; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px;
  background: var(--c); }}
.fc-foot {{ border-top: 1px solid {INK}; margin-top: 30px; padding-top: 10px; font-size: 12.5px; color: #4A4538; line-height: 1.6; }}
/* 탭: 선택 = 검정 바탕 흰 글씨, 나머지 = 테두리만 */
.stTabs [role="tablist"] {{ gap: 6px; border-bottom: none; box-shadow: none; flex-wrap: wrap; }}
.stTabs [data-testid="stTab"] {{ border: 1px solid {INK}; border-radius: 0; padding: 8px 16px; height: auto; background: transparent; }}
.stTabs [data-testid="stTab"] p {{ font-size: 15px; font-weight: 700; color: {INK}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background: {INK}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"], .stTabs [data-testid="stTabsHighlight"], .stTabs .react-aria-SelectionIndicator {{ display: none !important; }}
/* 버튼: 주황 바탕, 검정 굵은 글씨, 각진 모서리 */
.stButton button, [data-testid="stFormSubmitButton"] button {{ background: {OR}; color: {INK}; border: 1px solid {INK}; border-radius: 0;
  font-weight: 700; }}
.stButton button p, [data-testid="stFormSubmitButton"] button p {{ color: {INK}; font-weight: 700; }}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: {OR2}; color: {INK}; border-color: {INK}; }}
.stButton button:disabled {{ background: {BEIGE}; opacity: 0.8; }}
[data-testid="stExpander"] details, [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 0; }}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {LINE}; }}
[data-testid="stExpander"] summary p {{ font-weight: 700; }}
[data-baseweb="select"] > div, [data-baseweb="input"], .stTextInput input {{ border-radius: 0; }}
[data-testid="stProgress"] div[role="progressbar"] > div > div {{ background: {OR}; }}
@media (max-width: 640px) {{ .fc-title {{ font-size: 24px; }} .fc-date {{ text-align: left; }} .fc-kpi .v {{ font-size: 28px; }} }}
</style>"""


# ---------------------------------------------------------------------------
# 공용 조각
# ---------------------------------------------------------------------------
def html(s: str) -> None:
    st.html(f'<div class="fc">{s}</div>')


def _ctl(e: Exception) -> bool:
    """st.stop·st.rerun이 쓰는 제어 예외는 그대로 올린다."""
    return type(e).__name__ in ("StopException", "RerunException", "RerunData")


def top() -> None:
    html('<div class="fc-top"><div><div class="fc-kicker">FITCHECK · LIVE DEMO</div>'
         '<div class="fc-title">금융상품 추천 학습 데이터 품질 진단</div></div>'
         f'<div class="fc-date">기준일 {AS_OF}</div></div>')


def sec(title: str, chip: str, sub: str = "") -> str:
    s = f'<span class="fc-chip k">{escape(sub)}</span>' if sub else ""
    return f'<div class="fc-sec"><span class="fc-chip">{escape(chip)}</span>{s}<span>{escape(title)}</span></div>'


def kpis(items: list) -> str:
    """items: [(라벨, 값, 단위, 설명)]"""
    return '<div class="fc-kpis">' + "".join(
        f'<div class="fc-kpi"><div class="l">{escape(l)}</div><div class="v">{escape(v)}<small>{escape(u)}</small></div>'
        f'<div class="s">{escape(s)}</div></div>' for l, v, u, s in items) + "</div>"


def table(df: pd.DataFrame, res_cols=(), num_cols=(), hl=None, raw_cols=()) -> str:
    """보고서 표. res_cols = 결과 열(굵게 주황 갈색), hl = 강조할 행(불리언 목록), raw_cols = 이미 HTML인 열."""
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = ""
    for i, (_, r) in enumerate(df.iterrows()):
        cells = ""
        for c in df.columns:
            v = r[c]
            txt = "" if v is None or (isinstance(v, float) and pd.isna(v)) else (str(v) if c in raw_cols else escape(str(v)))
            cls = " ".join(x for x in ["res" if c in res_cols else "", "num" if c in num_cols else ""] if x)
            cells += f'<td class="{cls}">{txt}</td>' if cls else f"<td>{txt}</td>"
        rows += f'<tr class="hl">{cells}</tr>' if hl is not None and hl[i] else f"<tr>{cells}</tr>"
    return f'<div class="fc-scroll"><table class="fc-tbl"><tr>{head}</tr>{rows}</table></div>'


def mark(v) -> str:
    """판정 기호: 이행 ✓(검정), 미이행 ✗(주황), 판정 불가 –. 색과 기호를 같이 쓴다."""
    if v is None or pd.isna(v):
        return '<span class="fc-na">– 판정 불가</span>'
    return '<span class="fc-ok">✓ 이행</span>' if v == 1 else '<span class="fc-ng">✗ 미이행</span>'


def mark_short(v) -> str:
    if v is None or pd.isna(v):
        return '<span class="fc-na">–</span>'
    return '<span class="fc-ok">✓</span>' if v == 1 else '<span class="fc-ng">✗</span>'


_PATHY = re.compile(r"\s*\([^)]*(?:[/\\]|\.csv|\.json|https?:|localhost|127\.0\.0\.1|:\d{2,5})[^)]*\)")


def clean(text: str) -> str:
    """메시지에서 파일 경로·주소·예외 이름을 지운다 (괄호 안 경로, '조회 실패: XxxError')."""
    t = _PATHY.sub("", str(text))
    t = re.sub(r"조회 실패: \w+", "조회 실패", t)
    t = re.sub(r"\((?:[A-Z]\w*(?:Error|Exception|Timeout)\w*)\)", "", t)
    t = re.sub(r"[A-Za-z]:\\\S+|(?:data|src|outputs|docs)/\S+", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


@st.cache_data
def doc_names() -> dict:
    idx = _csv(ROOT / "docs" / "regulations" / "regulations_index.csv", dtype=str, keep_default_na=False)
    return dict(zip(idx["doc_id"], idx["문서명"]))


@st.cache_data
def article_gist() -> dict:
    """(doc_id, 조항) → 요지. gold 조항은 설명, 나머지는 조항 제목 + 본문 앞부분 (규정 문서이며 CoT 원문이 아니다)."""
    out = {}
    try:
        from src.rag.regulations import load_chunks
        for _, r in load_chunks().iterrows():
            body = re.sub(r"\s+", " ", str(r["본문"])).strip()
            body = re.sub(r"^제\s*\d+\s*조(의\s*\d+)?\s*(\([^)]*\))?\s*", "", body)
            out.setdefault((r["doc_id"], r["조항"]), f"{r['조항 제목']}: {body[:70]}…" if body else str(r["조항 제목"]))
    except Exception:
        pass
    g = _csv(ROOT / "src" / "rag" / "gold_articles.csv", dtype=str, keep_default_na=False)
    for _, r in g.iterrows():
        if r["조항"]:
            out[(r["doc_id"], r["조항"])] = r["설명"].strip('"')
    return out


def _grade(v) -> str:
    try:
        return "" if v is None or pd.isna(v) or v == "" else f"{int(float(v))}등급"
    except (TypeError, ValueError):
        return ""


def parse_products(s: str) -> list:
    """'상품명[데이터→최신] | …' → [(상품명, 데이터, 최신)]."""
    out = []
    for q in [x.strip() for x in str(s).split("|") if x.strip()]:
        m = re.search(r"\[([^\[\]]*)→([^\[\]]*)\]$", q)
        out.append((q[:m.start()] if m else q, m.group(1) if m else "", m.group(2) if m else ""))
    return out


def n_changed(s: str) -> int:
    return sum(1 for _, a, b in parse_products(s) if a and b and a != b)


def articles_of(cards: pd.DataFrame, cid: int) -> str:
    c = cards[cards["cot_id"].astype(int) == cid].drop_duplicates(["doc_id", "조항"])
    return ", ".join(f"{SHORT.get(d, d)} {a}" for d, a in zip(c["doc_id"], c["조항"])) or "없음"


# ---------------------------------------------------------------------------
# 비밀번호
# ---------------------------------------------------------------------------
def gate() -> None:
    ss = st.session_state
    if ss.get("ok"):
        return
    try:
        secret = str(st.secrets["public_password"])
    except Exception:
        secret = ""
    top()
    if not secret:
        st.error("시연 설정이 아직 끝나지 않았습니다. 운영자에게 문의해 주세요.")
        st.stop()
    locked = ss.get("lock_until", 0) - time.time()
    with st.form("login", border=True):
        pw = st.text_input("비밀번호", type="password", disabled=locked > 0)
        sent = st.form_submit_button("들어가기", type="primary", disabled=locked > 0)
    if locked > 0:
        st.warning(f"비밀번호를 여러 번 틀렸습니다. {int(locked) + 1}초 뒤 다시 시도해 주세요.")
    elif sent:
        if hmac.compare_digest(pw.encode(), secret.encode()):
            ss["ok"] = True
            st.rerun()
        time.sleep(1)
        ss["fail"] = ss.get("fail", 0) + 1
        if ss["fail"] >= MAX_FAIL:
            ss["fail"], ss["lock_until"] = 0, time.time() + LOCK
        st.error("비밀번호가 맞지 않습니다.")
    html(f'<div class="fc-foot">{escape(SOURCE)}</div>')
    st.stop()


# ---------------------------------------------------------------------------
# 단계 표시 (st.status)
# ---------------------------------------------------------------------------
def step_view(step: int, rec: dict) -> None:
    """단계 하나의 실제 내용 (경로·원문 없음)."""
    for m in rec.get("msgs", []):
        st.markdown(escape(clean(m)))
    for w in rec.get("warns", []):
        st.warning(clean(w))
    for title, d in rec.get("tables", []):
        if step == 1:
            t = d.reindex(columns=["문서명", "시행일", "상태"]).copy()
            t["상태"] = t["상태"].map(lambda v: re.sub(r"접속 실패.*", "접속 실패 → 보관본 사용", clean(v)))
            html(table(t))
        elif step == 2:
            t = d.reindex(columns=["성향", "읽은 허용 등급", "저장 표 허용 등급", "같음"]).copy()
            t["같음"] = t["같음"].map(lambda v: "✓ 같음" if str(v) == "True" else "✗ 다름")
            html(table(t, res_cols=("읽은 허용 등급",)))
        elif step == 3:
            t = d.reindex(columns=["상품명", "데이터 등급", "최신 등급", "판정 등급", "등급 다름", "값 출처"]).copy()
            ch = t["등급 다름"].map(lambda v: str(v) == "True")
            for c in ["데이터 등급", "최신 등급", "판정 등급"]:
                t[c] = t[c].map(_grade)
            t["등급 다름"] = ch.map(lambda v: "▲ 바뀜" if v else "")
            t["값 출처"] = t["값 출처"].map(lambda v: "실시간" if v == "실시간 조회" else "저장값" if v else "")
            order = (~ch).argsort(kind="stable")
            t, ch = t.iloc[order].reset_index(drop=True), ch.iloc[order].reset_index(drop=True)
            html(table(t, res_cols=("판정 등급",), hl=list(ch)))
            if ch.any():
                st.caption(f"강조 행 {int(ch.sum())}개: 데이터 등급과 최신 등급이 다른 상품입니다. 판정 등급은 사람이 확인한 등급을 먼저 씁니다 (G 규칙).")
        elif step == 4:
            names, gist = doc_names(), article_gist()
            t = d.reindex(columns=["유형", "순위", "doc_id", "조항", "시행일"]).copy()
            t["문서"] = t["doc_id"].map(lambda x: names.get(x, x))
            t["조항 요지"] = [gist.get((a, b), "") for a, b in zip(t["doc_id"], t["조항"])]
            html(table(t[["유형", "순위", "문서", "조항", "조항 요지", "시행일"]], res_cols=("조항",)))
        elif step == 5:
            html(table(d, res_cols=tuple(c for c in d.columns if c.startswith("점수") or c.startswith("득점 (최신")),
                       raw_cols=tuple(d.columns)))


def stream_steps(events, on_done) -> tuple:
    """run_live 이벤트를 단계마다 st.status 하나(진행 중 → 완료·걸린 시간)로 그린다.
    ⑤는 on_done(done 이벤트, ⑤ 상자)이 그린다. 반환: (단계 기록, done 이벤트). 예외는 부른 쪽에서 잡는다."""
    recs, boxes, cur, bar, done = {}, {}, None, None, None

    def close(k):
        recs[k]["secs"] = round(time.time() - recs[k].pop("t0"), 1)
        boxes[k].update(label=f"{STEPS[k]} · 완료 {recs[k]['secs']}초", state="complete", expanded=k >= 3)

    for ev in events:
        k, kind = ev["step"], ev["kind"]
        if k == 0:
            continue
        if kind == "done":
            done = ev
            break
        if k != cur:
            if cur is not None:
                close(cur)
            cur, bar = k, None
            recs[k] = {"t0": time.time(), "msgs": [], "warns": [], "tables": []}
            boxes[k] = st.status(f"{STEPS[k]} · 진행 중…", expanded=True)
        rec = recs[k]
        if k == 5:
            boxes[k].update(state="running")
            continue   # ⑤ 판정 메시지·표는 on_done이 건별로 그린다
        with boxes[k]:
            if kind == "progress":
                if bar is None:
                    bar = st.progress(0.0)
                bar.progress(ev["i"] / ev["n"], text=f"조회 {ev['i']}/{ev['n']}")
            elif kind in ("msg", "warn"):
                rec["msgs" if kind == "msg" else "warns"].append(ev["text"])
                step_view(k, {"msgs" if kind == "msg" else "warns": [ev["text"]]})
            elif kind == "table":
                rec["tables"].append((ev["title"], ev["data"]))
                step_view(k, {"tables": [(ev["title"], ev["data"])]})
        boxes[k].update(state="running")   # with 블록을 나오면 st.status가 '완료'로 바뀌므로 다시 진행 중으로
    if done is not None and cur == 5:
        with boxes[5]:
            recs[5]["tables"] = on_done(done)
    if cur is not None:
        close(cur)
    return recs, done


def replay_steps(recs: dict) -> None:
    for k, rec in recs.items():
        with st.status(f"{STEPS[k]} · 완료 {rec.get('secs', '?')}초", state="complete", expanded=False):
            step_view(k, rec)


# ---------------------------------------------------------------------------
# 판정 결과 (건별 행, 사례 표)
# ---------------------------------------------------------------------------
def frames(res: pd.DataFrame, J: pd.DataFrame) -> tuple:
    fa, fb = (live_frame(res, J, x) for x in ("데이터 기준", "최신 문서 기준"))
    return fa, fb


def batch_rows(done: dict, J: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """30건 결과 → 표 행 (cot_id, 점수 데이터→최신, 항목별 판정(최신 문서 기준), 등급 바뀐 상품 수, 근거 조항, 출처)."""
    res, cards = done["result"], done["cards"]
    fa, fb = frames(res, J)
    sa, sb = cot_scores(fa).set_index("cot_id"), cot_scores(fb).set_index("cot_id")
    lt = done.get("latest")
    src = dict(zip(lt["상품명"], lt["값 출처"])) if lt is not None and "값 출처" in lt else {}
    rows = []
    for _, r in res.iterrows():
        cid = int(r["cot_id"])
        names = [n for n, _, _ in parse_products(r["추천 상품(데이터→최신)"])]
        live_all = bool(names) and all(src.get(n) == "실시간 조회" for n in names)
        s_ = source or ("실시간" if live_all else "저장값")
        row = {"cot_id": cid, "점수 (데이터 → 최신)": f"{sa.loc[cid, '점수']:g} → {sb.loc[cid, '점수']:g}"}
        for k, *_ in ITEMS:
            row[k] = mark_short(sb.loc[cid, k])
        row |= {"바뀐 상품": f"{n_changed(r['추천 상품(데이터→최신)'])}개", "근거 조항": escape(articles_of(cards, cid)),
                "출처": f'<span class="fc-src{" saved" if s_ != "실시간" else ""}">{s_}</span>'}
        rows.append(row)
    return pd.DataFrame(rows)


def batch_summary(done: dict, J: pd.DataFrame) -> dict:
    res = done["result"]
    fa, fb = frames(res, J)
    a, b = summarize(item_pass(fa)), summarize(item_pass(fb))
    new_out = int(((res["③ 바뀜"].astype(str) == "True") & (res["③ 최신 문서 기준"] == "범위 밖")).sum())
    return {"a": a, "b": b, "결함": int(res["결함(최신 문서 기준)"].astype(str).eq("True").sum()),
            "결함 데이터": int(res["결함(데이터 기준)"].astype(str).eq("True").sum()), "새 부적합": new_out, "n": len(res)}


def summary_view(sm: dict, secs: float, at: str, n_live: int) -> None:
    html(kpis([
        ("평균 품질 점수 (최신 문서 기준)", f"{sm['b']['점수']:.1f}", f"점 {sm['b']['등급']}", f"데이터 기준 {sm['a']['점수']:.1f}점 {sm['a']['등급']}"),
        ("결함 있는 상담", f"{sm['결함']}", f"/ {sm['n']}건", f"데이터 기준 {sm['결함 데이터']}건"),
        ("최신 등급으로 새로 부적합", f"{sm['새 부적합']}", "건", "데이터 기준 범위 안 → 최신 기준 범위 밖"),
        ("걸린 시간", f"{secs:g}", "초", f"검색 시각 {at} · 실시간 {n_live}/{sm['n']}건"),
    ]))


# ---------------------------------------------------------------------------
# 앱 전체 공유 상태 (30건 실시간 검색: 동시에 1개, 끝난 뒤 5분)
# ---------------------------------------------------------------------------
@st.cache_resource
def shared() -> dict:
    return {"lock": threading.Lock(), "running": False, "started": 0.0, "ended": 0.0, "last": None}


def try_start(sh: dict) -> str:
    """시작할 수 있으면 '' (running 표시), 아니면 이유."""
    with sh["lock"]:
        now = time.time()
        if sh["running"]:
            return f"다른 사용자가 실행 중입니다 ({int(now - sh['started'])}초 전에 시작). 끝나면 결과가 여기에 나옵니다."
        wait = BATCH_GAP - (now - sh["ended"]) if sh["ended"] else 0
        if wait > 0:
            return f"30건 실시간 검색은 5분 간격입니다. {int(wait) // 60}분 {int(wait) % 60}초 뒤 다시 실행할 수 있습니다."
        sh["running"], sh["started"] = True, now
        return ""


@st.cache_resource(show_spinner="저장된 결과 준비 중…")
def saved_run(_base: dict, ids: tuple) -> dict:
    last = None
    for ev in run_live(_base, list(ids), live=False):
        if ev["kind"] == "done":
            last = ev
    return last


def run_batch(base: dict, J: pd.DataFrame, ids: list) -> dict:
    """표본 30건 실시간 검색. ⑤에서 표에 1건씩 추가한다. 중간 실패 시 나머지 행은 저장값."""
    t0, at = time.time(), datetime.now().strftime("%Y-%m-%d %H:%M")
    shown: list = []
    state = {"done": None}

    def on_done(done):
        state["done"] = done
        rows = batch_rows(done, J)
        st.markdown(f"상담 {len(rows)}건 판정 (데이터 기준 vs 최신 문서 기준). 표에 1건씩 추가합니다.")
        bar, ph = st.progress(0.0), st.empty()
        for i in range(len(rows)):
            shown.append(rows.iloc[i])
            ph.html(f'<div class="fc">{table(pd.DataFrame(shown), res_cols=("점수 (데이터 → 최신)",), raw_cols=tuple(rows.columns))}</div>')
            bar.progress((i + 1) / len(rows), text=f"판정 {i + 1}/{len(rows)}")
            time.sleep(0.08)
        return []

    recs, failed = {}, False
    try:
        recs, done = stream_steps(run_live(base, ids, live=True), on_done)
        if state["done"] is None:
            failed = True
    except Exception as e:
        if _ctl(e):
            raise
        failed = True
    done = state["done"]
    rows = pd.DataFrame(shown) if shown else None
    if failed or rows is None or len(rows) < len(ids):
        sv = saved_run(base, tuple(ids))
        srows = batch_rows(sv, J, source="저장값")
        have = set(rows["cot_id"]) if rows is not None else set()
        rows = pd.concat([rows, srows[~srows["cot_id"].isin(have)]], ignore_index=True) if rows is not None else srows
        done = done or sv
        st.warning("실시간 검색을 끝까지 마치지 못해 나머지는 저장값으로 채웠습니다 ('저장값' 표시).")
    rows = rows.sort_values("cot_id").reset_index(drop=True)
    return {"recs": recs, "rows": rows, "summary": batch_summary(done, J), "secs": round(time.time() - t0, 1), "at": at,
            "n_live": int(rows["출처"].str.contains(">실시간<").sum()), "ids": ids}


def batch_result(last: dict) -> None:
    html(sec("마지막 실행 결과", "요약", f"{last['at']}"))
    summary_view(last["summary"], last["secs"], last["at"], last["n_live"])
    html(sec("상담별 판정", "결과", f"{len(last['rows'])}건"))
    html('<div class="fc-note" style="margin-bottom:8px">항목 기호: ✓ 이행, ✗ 미이행, – 판정 불가 (최신 문서 기준). '
         '점수는 데이터 기준 → 최신 문서 기준 100점 환산입니다.</div>')
    html(table(last["rows"], res_cols=("점수 (데이터 → 최신)",), raw_cols=tuple(last["rows"].columns)))


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
def _loss(x) -> str:
    x = float(x)
    return f"−{x:.2f}" if x >= 0.005 else "0.00"


def tab_summary(J: pd.DataFrame, F: dict) -> None:
    o = summarize(item_pass(J))
    cr = summarize(F["corr_pass"]) if F.get("corr_pass") is not None else None
    n_out = int((J["기준③ 처리"] == "제외").sum())
    n_d4 = int((J["기준④ 판정"] == "기재 없음").sum())
    n_tr = int((J["이관 판정"] == "결함").sum())
    n_2a = int(J["기준② 판정"].astype(str).str.startswith("②-").sum())
    html(sec("품질 점수", "핵심", "원본 → 교정판"))
    html(kpis([
        ("원본 품질 점수", f"{o['점수']:.1f}", f"점 {o['등급']}", f"증권 CoT {o['CoT']:,}건 평균 (100점 환산)"),
        ("교정판 품질 점수", f"{cr['점수']:.1f}" if cr else "-", f"점 {cr['등급']}" if cr else "", f"교정 후 다시 판정한 {cr['CoT']:,}건" if cr else "교정판 점수 없음"),
        ("점수 변화", f"{cr['점수'] - o['점수']:+.1f}" if cr else "-", "점", "범위 밖 제외 + 위험등급 문장 추가 후"),
        ("④ 위험 설명 이행률", f"{o['항목'].set_index('키').loc['④', '이행률']:g}", "%", "원본에서 감점이 가장 큰 항목입니다"),
    ]))
    if cr is not None:
        ta, tb = o["항목"].set_index("키"), cr["항목"].set_index("키")
        mx = max(ta["배점"].max(), 1)
        bars = ""
        for k, nm, w, law in ITEMS:
            la, lb = float(ta.loc[k, "감점"]), float(tb.loc[k, "감점"])
            bars += (f'<div class="fc-bar-row"><div class="fc-bar-name">{escape(nm)}<small>배점 {w:g} · {escape(law)}</small></div><div>'
                     f'<div class="fc-bar-line"><span>원본</span><div class="fc-bar-track"><div class="fc-bar-fill" '
                     f'style="width:{la / mx * 100:.1f}%;background:{OR}"></div></div><b>{_loss(la)}</b></div>'
                     f'<div class="fc-bar-line"><span>교정판</span><div class="fc-bar-track"><div class="fc-bar-fill" '
                     f'style="width:{lb / mx * 100:.1f}%;background:{BEIGE}"></div></div><b>{_loss(lb)}</b></div></div></div>')
        html(sec("항목별 감점", "그래프", "원본 vs 교정판"))
        html(f'<div class="fc-card"><div class="fc-legend"><span style="--c:{OR}">원본</span><span style="--c:{BEIGE}">교정판</span>'
             f'<span style="--c:transparent">막대 길이 = 감점 (최대 25점 기준)</span></div><div class="fc-bars">{bars}</div></div>')
        t = pd.DataFrame([{"항목": nm, "배점": f"{w:g}", "원본 이행률": f"{ta.loc[k, '이행률']:g}%", "원본 감점": _loss(ta.loc[k, '감점']),
                           "교정판 이행률": f"{tb.loc[k, '이행률']:g}%", "교정판 감점": _loss(tb.loc[k, '감점'])} for k, nm, w, _ in ITEMS])
        html(table(t, res_cols=("교정판 감점",), num_cols=("배점", "원본 이행률", "원본 감점", "교정판 이행률", "교정판 감점")))
    html(sec("주요 지표", "지표", f"증권 CoT {len(J):,}건"))
    html(kpis([
        ("③ 범위 밖 추천 (제외)", f"{n_out:,}", "건", "성향 허용 등급보다 위험한 상품을 추천한 상담입니다"),
        ("④ 위험등급 설명 없음", f"{n_d4:,}", "건", "교정판에서 위험등급 문장을 추가했습니다"),
        ("이관 결함", f"{n_tr:,}", "건", "부적합 상품을 안내 없이 권유한 상담입니다"),
        ("② 근거 없는 성향 판단", f"{n_2a:,}", "건", "필수정보가 부족한데 성향을 확정했거나 미확정인데 추천했습니다"),
    ]))
    html('<div class="fc-card"><p>채점 방식: 금융감독원 미스터리쇼핑 방식을 빌렸습니다. CoT 1건을 판매 상담 1회로 보고, 항목을 지키면 배점을 주고 '
         '못 지키면 0점(일부 이행도 0점)입니다. ①②③·이관 각 18.75점, ④ 25점으로 100점이며, 판정할 수 없는 항목은 그 CoT에서 빼고 환산합니다. '
         '등급: 우수 90점 이상, 양호 80, 보통 70, 미흡 60, 저조 60점 미만.</p></div>')


def tab_batch(base: dict, J: pd.DataFrame) -> None:
    sh = shared()
    ids = pick_sample(base, N_BATCH)
    html(sec("표본 30건 실시간 검색", "실시간", "CLI live --n 30과 같은 표본"))
    html(f'<div class="fc-card"><p>로컬 명령 <b>python -m src.fitcheck live --n 30</b>과 같은 방식(고정 시드)으로 뽑은 30건입니다. '
         f'기준 문서를 받아 등급표를 만들고, 추천 상품의 최신 위험등급을 상품명으로 조회한 뒤, 근거 조항을 찾아 다시 판정합니다. '
         f'보통 5분 남짓 걸립니다.</p><div class="fc-note">표본 cot_id: {", ".join(map(str, ids))}</div></div>')
    go = st.button("표본 30건 실시간 검색", type="primary", key="batch_go")
    if go:
        why = try_start(sh)
        if why:
            st.info(why)
        else:
            try:
                html(sec("진행 상황", "단계", "① → ⑤"))
                last = run_batch(base, J, ids)
                with sh["lock"]:
                    sh["last"] = last
                st.success(f"30건 실시간 검색을 마쳤습니다 ({last['secs']:g}초).")
                batch_result(last)
                return
            finally:
                with sh["lock"]:
                    sh["running"], sh["ended"] = False, time.time()
    else:
        now = time.time()
        if sh["running"]:
            st.info(f"다른 사용자가 실행 중입니다 ({int(now - sh['started'])}초 전에 시작). 끝나면 아래에 결과가 나옵니다.")
        elif sh["ended"] and BATCH_GAP - (now - sh["ended"]) > 0:
            w = int(BATCH_GAP - (now - sh["ended"]))
            st.info(f"다음 실행까지 {w // 60}분 {w % 60}초 남았습니다. 아래는 마지막 실행 결과입니다.")
    last = sh["last"]
    if last is None:
        st.caption("아직 실행 기록이 없습니다. 버튼을 누르면 단계별 진행이 이 아래에 나옵니다.")
        return
    with st.expander(f"마지막 실행 단계 기록 ({last['at']}, {last['secs']:g}초)"):
        replay_steps(last["recs"])
    batch_result(last)


def case_tables(J: pd.DataFrame, cid: int, done: dict | None) -> None:
    """사례: 점수, 항목별 판정, 추천 상품, 근거 조항·시행일."""
    row = J[J["cot_id"] == cid]
    jr = row.iloc[0]
    p = item_pass(row)
    s1 = summarize(p)
    kp = [("품질 점수 (저장 판정)", f"{s1['점수']:g}", f"점 {s1['등급']}", f"성향 {jr['성향_P5'] or '미확정'} · {jr['분할']}")]
    if done is not None:
        res = done["result"][done["result"]["cot_id"].astype(int) == cid]
        fa, fb = frames(res, J)
        a, b = summarize(item_pass(fa)), summarize(item_pass(fb))
        kp += [("데이터 기준", f"{a['점수']:g}", f"점 {a['등급']}", f"③ {res['③ 데이터 기준'].iloc[0]}"),
               ("최신 문서 기준", f"{b['점수']:g}", f"점 {b['등급']}", f"③ {res['③ 최신 문서 기준'].iloc[0]}")]
    html(kpis(kp))
    val = {"①": jr["기준① 법정 미확인"] or "법정 3항목 확인", "②": jr["기준② 판정"], "③": jr["기준③ 판정"],
           "이관": jr["이관 판정"], "④": jr["기준④ 판정"]}
    t = pd.DataFrame([{"항목": nm, "근거": law, "판정": mark(p.iloc[0][k]), "세부": escape(str(val[k])),
                       "득점": "제외" if pd.isna(p.iloc[0][k]) else f"{w * p.iloc[0][k]:g} / {w:g}"} for k, nm, w, law in ITEMS])
    html(sec("항목별 판정", "판정", "저장 판정"))
    html(table(t, res_cols=("득점",), raw_cols=("판정", "세부"), num_cols=("득점",)))
    if done is None:
        return
    res = done["result"][done["result"]["cot_id"].astype(int) == cid]
    pr = parse_products(res["추천 상품(데이터→최신)"].iloc[0])
    pt = pd.DataFrame([{"상품명": n, "데이터 등급": f"{a}등급" if a else "", "최신 등급": f"{b}등급" if b else "최신 등급 없음",
                        "변화": "▲ 바뀜" if a and b and a != b else ""} for n, a, b in pr])
    html(sec("추천 상품", "상품", "데이터 등급 → 최신 등급"))
    html(table(pt, res_cols=("최신 등급",), hl=[x == "▲ 바뀜" for x in pt["변화"]]))
    cards = done["cards"][done["cards"]["cot_id"].astype(int) == cid].drop_duplicates(["결함 유형", "doc_id", "조항"])
    html(sec("근거 조항과 시행일", "근거"))
    if not len(cards):
        html('<div class="fc-card"><p>이 상담에는 결함이 없어 근거 조항을 붙이지 않았습니다.</p></div>')
    else:
        names, gist = doc_names(), article_gist()
        ct = pd.DataFrame([{"결함 유형": r["결함 유형"], "문서": names.get(r["doc_id"], r["doc_id"]), "조항": r["조항"],
                            "조항 요지": gist.get((r["doc_id"], r["조항"]), ""), "시행일": r["문서 시행일"]} for _, r in cards.iterrows()])
        html(table(ct, res_cols=("조항",)))


def tab_case(base: dict, J: pd.DataFrame) -> None:
    ss, sh = st.session_state, shared()
    html(sec("시연 사례", "사례", "6건"))
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    cid = c1.selectbox("시연 사례", list(DEMO), format_func=lambda i: f"{i} · {DEMO[i]}")
    wait = CASE_GAP - (time.time() - ss.get("case_at", 0))
    go = c2.button("이 사례 실시간 검색", type="primary", disabled=wait > 0 or sh["running"], key="case_go")
    if sh["running"]:
        st.caption("30건 실시간 검색이 도는 동안에는 사례 실시간 검색을 쉽니다.")
    elif wait > 0:
        st.caption(f"사례 실시간 검색은 1분에 1번입니다. {int(wait) + 1}초 뒤 다시 누를 수 있습니다.")
    lv = ss.get("case_live")
    if go and wait <= 0 and not sh["running"]:
        ss["case_at"] = time.time()
        done, recs = None, {}

        def on_done(d):
            r = d["result"][d["result"]["cot_id"].astype(int) == int(cid)]
            fa, fb = frames(r, J)
            pa, pb = item_pass(fa), item_pass(fb)
            a, b = summarize(pa), summarize(pb)
            rows = [{"항목": nm, "데이터 기준": mark(pa.iloc[0][k]), "최신 문서 기준": mark(pb.iloc[0][k]),
                     "득점 (데이터)": "제외" if pd.isna(pa.iloc[0][k]) else f"{w * pa.iloc[0][k]:g} / {w:g}",
                     "득점 (최신)": "제외" if pd.isna(pb.iloc[0][k]) else f"{w * pb.iloc[0][k]:g} / {w:g}"} for k, nm, w, _ in ITEMS]
            rows.append({"항목": "<b>점수 (100점 환산)</b>", "데이터 기준": a["등급"], "최신 문서 기준": b["등급"],
                         "득점 (데이터)": f"{a['점수']:g}", "득점 (최신)": f"{b['점수']:g}"})
            t = pd.DataFrame(rows)
            step_view(5, {"tables": [("항목별 판정", t)]})
            return [("항목별 판정", t)]

        t0 = time.time()
        try:
            html(sec("진행 상황", "단계", "① → ⑤"))
            recs, done = stream_steps(run_live(base, [int(cid)], live=True), on_done)
        except Exception as e:
            if _ctl(e):
                raise
            done = None
        ss["case_live"] = lv = {"cid": cid, "done": done, "recs": recs, "secs": round(time.time() - t0, 1)}
        if done is None:
            st.warning("실시간 검색을 마치지 못해 저장된 결과를 보여 드립니다.")
        else:
            st.caption(f"전체 {lv['secs']:g}초")
    elif lv and lv["cid"] == cid and lv["recs"]:
        with st.expander(f"지난 실시간 검색 단계 기록 ({lv['secs']:g}초)"):
            replay_steps(lv["recs"])
    done = lv["done"] if lv and lv["cid"] == cid and lv["done"] is not None else saved_run(base, tuple(DEMO))
    label = "실시간 검색 결과" if lv and lv["cid"] == cid and lv["done"] is not None else f"저장된 결과 (기준일 {AS_OF})"
    html(f'<div class="fc-note" style="margin:10px 0 4px"><span class="fc-chip k">{escape(label)}</span></div>')
    case_tables(J, int(cid), done)


def page() -> None:
    base = base_for(AS_OF)
    J = base["j"]
    F = facts((OUT / "scorecard.csv").stat().st_mtime + ((OUT / "score_cot.csv").stat().st_mtime if (OUT / "score_cot.csv").exists() else 0))
    top()
    t1, t2, t3 = st.tabs(["진단 결과", "실시간 검색", "사례 보기"])
    with t1:
        tab_summary(J, F)
    with t2:
        tab_batch(base, J)
    with t3:
        tab_case(base, J)
    html(f'<div class="fc-foot">{escape(SOURCE)}</div>')


st.set_page_config(page_title="FitCheck 시연", layout="wide", initial_sidebar_state="collapsed")
st.html(CSS)
gate()
try:
    page()
except Exception as e:  # 상세 오류·경로를 화면에 내지 않는다
    if _ctl(e):
        raise
    st.error(GENERIC)
