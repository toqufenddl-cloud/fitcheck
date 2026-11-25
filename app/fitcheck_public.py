"""FitCheck 외부용 시연 앱. 실행: run_public.ps1 (앱 127.0.0.1:8501 + ngrok 고정 도메인).

로컬 앱(app/fitcheck_app.py)과 화면 부품(app/fc_ui.py)을 같이 쓰되, 외부에 보여도 되는 것만 남긴다.
- 원문을 읽는 코드가 없다 (원문 보기 토글, question·cot1·cot2·cot3·answer 접근 없음).
- 화면 출력은 점수, 판정, 근거 조항, 시행일만. 상품명·단계 기록·파일 경로·내려받기는 없다.
- 시작할 때 비밀번호 (.streamlit/secrets.toml의 public_password, git 제외). 세션당 5번 틀리면 1분 잠금.
- 상담 선택은 시연용 cot_id 목록만 (자유 입력·파일 업로드 없음).
- 기본은 저장값 결과. '실시간 검색'은 세션당 1분에 1번, 실패하면 저장값 결과로 대신한다.
- 오류는 일반 문구로만 보인다 (상세는 run_public.ps1의 --client.showErrorDetails none으로도 막는다).
애매한 경우 처리: 실시간 검색이 중간에 실패하거나 결과가 비면 이유를 말하지 않고 저장값 결과를 보여 준다.
"""
from __future__ import annotations

import hmac
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_ui import (CSS, FOOT, ICON_OK, ICON_WARN, ICON_X, ITEMS, ROOT, SUB, BODY, TRACK, WEIGHT,  # noqa: E402
                   _csv, band, base_for, corr_sentence, counts, dark_card, escape, facts, guide_version, html,
                   item_bars, item_pass, live_frame, opinion, run_live, score_card, st, summarize, why_lines,
                   worst_sentence)
import pandas as pd  # noqa: E402

AS_OF = "2026-11-24"
OUT = ROOT / "outputs"
# 시연용 cot_id (docs/demo_script.md 세 사례 + 대표 사례). 유형 설명은 판정 결과에서 온 것이다
DEMO = {1547: "최신 등급으로 ③ 판정이 바뀜", 2365: "범위 밖 추천 + 이관 결함", 43: "위험등급 설명 누락",
        82: "모든 항목 이행", 145: "필수정보 일부 미확인", 5851: "근거 부족으로 성향 확정"}
LIVE_GAP = 60          # 실시간 검색 간격 (초, 세션당)
MAX_FAIL, LOCK = 5, 60  # 비밀번호 실패 허용 횟수, 잠금 시간 (초)
GENERIC = "일시적인 문제로 화면을 표시하지 못했습니다. 잠시 뒤 다시 시도해 주세요."
EXTRA_CSS = """<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
.fc-tbl { width: 100%; border-collapse: collapse; font-size: 14px; }
.fc-tbl th, .fc-tbl td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #DCE2EC; }
.fc-tbl th { color: #4A5568; font-weight: 700; font-size: 13px; }
</style>"""


def _ctl(e: Exception) -> bool:
    """st.stop·st.rerun이 쓰는 제어 예외는 그대로 올려야 한다."""
    return type(e).__name__ in ("StopException", "RerunException", "RerunData")


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
    html('<header class="fc-head"><div style="display:flex;align-items:center;gap:14px"><div class="fc-logo">'
         '<span style="font-size:22px;font-weight:800;color:#FFFFFF">✓</span></div><div>'
         '<div style="font-size:22px;font-weight:800">FitCheck</div>'
         '<div style="font-size:13px;color:#B8C4DC">금융상품 추천 학습 데이터 품질 진단 · 시연</div></div></div></header>')
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
    st.stop()


# ---------------------------------------------------------------------------
# 결과 (점수·판정·근거 조항·시행일)
# ---------------------------------------------------------------------------
@st.cache_data
def doc_names() -> dict:
    idx = _csv(ROOT / "docs" / "regulations" / "regulations_index.csv", dtype=str, keep_default_na=False)
    return dict(zip(idx["doc_id"], idx["문서명"]))


def _pick(ev) -> dict | None:
    """run_live 이벤트에서 화면에 쓸 것만 남긴다 (result는 점수 계산용, 상품명 열은 화면에 쓰지 않는다)."""
    if not ev or ev.get("result") is None or not len(ev["result"]):
        return None
    keep = ["cot_id", "③ 데이터 기준", "③ 최신 문서 기준", "③ 바뀜", "④", "이관", "②"]
    cards = ev["cards"].reindex(columns=["cot_id", "결함 유형", "doc_id", "조항", "문서 시행일", "검색 시각"])
    return {"result": ev["result"][keep].copy(), "cards": cards.copy()}


def consume(events) -> dict | None:
    last = None
    for ev in events:
        if ev["kind"] == "done":
            last = ev
    return _pick(last)


@st.cache_resource(show_spinner="저장된 결과 준비 중…")
def saved_all(_base: dict, as_of: str) -> dict | None:
    return consume(run_live(_base, list(DEMO), live=False))


def for_cot(r: dict | None, cid: int) -> dict | None:
    if r is None:
        return None
    res = r["result"][r["result"]["cot_id"].astype(int) == cid]
    if not len(res):
        return None
    return {"result": res, "cards": r["cards"][r["cards"]["cot_id"].astype(int) == cid]}


def evidence_table(cards: pd.DataFrame) -> str:
    if not len(cards):
        return f'<div style="font-size:14px;color:{SUB}">이 상담에는 결함이 없어 근거 조항을 붙이지 않았습니다.</div>'
    names = doc_names()
    rows = "".join(f'<tr><td>{escape(str(r["결함 유형"]))}</td><td>{escape(names.get(r["doc_id"], r["doc_id"]))} {escape(str(r["조항"]))}</td>'
                   f'<td>{escape(str(r["문서 시행일"]))}</td></tr>'
                   for _, r in cards.drop_duplicates(["결함 유형", "doc_id", "조항"]).iterrows())
    return f'<table class="fc-tbl"><tr><th>결함 유형</th><th>근거 조항</th><th>시행일</th></tr>{rows}</table>'


def result_block(r: dict, J: pd.DataFrame, mode: str) -> None:
    res = r["result"]
    a, b = (summarize(item_pass(live_frame(res, J, x)))
            for x in ("데이터 기준", "최신 문서 기준"))
    x = res.iloc[0]
    diff = round(b["점수"] - a["점수"], 1)
    html('<div class="fc-grid">'
         + score_card("데이터 기준", a, f"③ 판정: {x['③ 데이터 기준']}")
         + score_card("최신 문서 기준", b, f"③ 판정: {x['③ 최신 문서 기준']}", diff)
         + dark_card("결과 출처", mode, "최신 문서 기준은 검색한 기준 문서와 최신 상품 등급으로 ③을 다시 판정한 것입니다.")
         + "</div>")
    html(f'<div class="fc-sec"><h2 class="fc-h2">근거 조항과 시행일</h2>{evidence_table(r["cards"])}</div>')


def item_rows(J: pd.DataFrame, cid: int) -> tuple:
    row = J[J["cot_id"] == cid]
    p = item_pass(row)
    jr = row.iloc[0]
    val = {"①": jr["기준① 법정 미확인"] or "법정 3항목 확인", "②": jr["기준② 판정"], "③": jr["기준③ 판정"],
           "이관": jr["이관 판정"], "④": jr["기준④ 판정"]}
    lines = ""
    for k, nm, w, law in ITEMS:
        v = p.iloc[0][k]
        chip = (("이행", "#DCFCE7", "#166534", ICON_OK) if v == 1 else ("미이행", "#FEE2E2", "#991B1B", ICON_X) if v == 0
                else ("판정 불가", TRACK, BODY, ICON_WARN))
        pts = "제외" if pd.isna(v) else f"{w * v:g} / {w:g}"
        lines += (f'<div class="fc-row" style="grid-template-columns:minmax(140px,220px) 110px minmax(0,1fr) 80px">'
                  f'<div style="font-size:15px;font-weight:700">{escape(nm)}<div style="font-size:12px;color:{SUB};font-weight:400">'
                  f'{escape(law)}</div></div>'
                  f'<span class="fc-badge" style="background:{chip[1]};color:{chip[2]};font-size:13px">{chip[3]}{chip[0]}</span>'
                  f'<div style="font-size:13px;color:{BODY}">{escape(str(val[k]))}</div>'
                  f'<div style="font-size:13px;text-align:right;font-weight:700">{pts}</div></div>')
    return summarize(p), jr, lines


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
def page() -> None:
    base = base_for(AS_OF)
    J = base["j"]
    F = facts((OUT / "scorecard.csv").stat().st_mtime + ((OUT / "score_cot.csv").stat().st_mtime if (OUT / "score_cot.csv").exists() else 0))
    ver = guide_version()
    html(f'<header class="fc-head"><div style="display:flex;align-items:center;gap:14px"><div class="fc-logo">'
         f'<span style="font-size:22px;font-weight:800;color:#FFFFFF">✓</span></div><div>'
         f'<div style="font-size:22px;font-weight:800;letter-spacing:-0.3px">FitCheck</div>'
         f'<div style="font-size:13px;color:#B8C4DC">금융상품 추천 학습 데이터 품질 진단 · 시연</div></div></div>'
         f'<div style="display:flex;flex-wrap:wrap;gap:8px"><span class="fc-chip">기준일 {AS_OF}</span>'
         f'<span class="fc-chip">증권 CoT {len(J):,}건</span><span class="fc-chip">판정기준서 v{ver}</span></div></header>')
    tab_d, tab_c = st.tabs(["진단 결과", "사례 보기"])

    with tab_d:
        o = summarize(item_pass(J))
        if F["corr_pass"] is None:
            html(score_card("원본 데이터", o, worst_sentence(o)))
        else:
            cr = summarize(F["corr_pass"])
            n1c = int((F["corr_pass"]["①"] == 0).sum())
            html('<div class="fc-grid">'
                 + score_card("원본 데이터", o, worst_sentence(o))
                 + score_card("FitCheck 교정판", cr, corr_sentence(cr, F["n_add"], len(J) - cr["CoT"] - cr["점수 없음"]),
                              round(cr["점수"] - o["점수"], 1))
                 + dark_card("진단 의견", opinion(o, cr, n1c)) + "</div>")
            html(band([("원본", o["점수"], o["등급"]), ("교정판", cr["점수"], cr["등급"])]))
            html(item_bars("항목별 점수와 감점 이유", o, cr, ("원본", "교정판"), why_lines(counts(J), cr)))
        html(f'<div class="fc-foot">{escape(FOOT.format(v=ver))}</div>')

    with tab_c:
        ss = st.session_state
        with st.container(border=True):
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            cid = c1.selectbox("시연 사례", list(DEMO), format_func=lambda i: f"{i} · {DEMO[i]}")
            wait = LIVE_GAP - (time.time() - ss.get("live_at", 0))
            go = c2.button("실시간 검색", type="primary", disabled=wait > 0,
                           help="최신 기준 문서와 상품 등급을 검색해 다시 판정합니다. 1분에 1번")
            if wait > 0:
                st.caption(f"실시간 검색은 1분에 1번입니다. {int(wait) + 1}초 뒤 다시 누를 수 있습니다.")
        s1, jr, lines = item_rows(J, cid)
        html('<div class="fc-grid" style="grid-template-columns:repeat(auto-fit,minmax(280px,1fr))">'
             + score_card(f"cot_id {cid} · {jr['분할']}", s1, f"성향 {jr['성향_P5'] or '미확정'}")
             + f'<div class="fc-sec" style="margin:0;gap:0"><h2 class="fc-h2" style="padding-bottom:10px">항목별 판정 (저장 판정)</h2>{lines}</div>'
             + "</div>")
        if go:
            ss["live_at"] = time.time()
            r = None
            try:
                with st.spinner("기준 문서·상품 등급 검색 중… (보통 10~30초)"):
                    r = for_cot(consume(run_live(base, [int(cid)], live=True)), int(cid))
            except Exception as e:
                if _ctl(e):
                    raise
            ss["live"] = (cid, r) if r is not None else (cid, None)
            st.rerun()
        lv = ss.get("live")
        if lv and lv[0] == cid and lv[1] is not None:
            result_block(lv[1], J, f"실시간 검색 결과 (검색 시각 {str(lv[1]['cards']['검색 시각'].iloc[0])[:16] if len(lv[1]['cards']) else '기록 없음'})")
            return
        if lv and lv[0] == cid and lv[1] is None:
            st.info("실시간 검색을 마치지 못해 저장된 결과를 보여 줍니다.")
        r = for_cot(saved_all(base, AS_OF), int(cid))
        if r is None:
            st.info(GENERIC)
        else:
            result_block(r, J, "저장된 결과 (기준일 " + AS_OF + ")")


st.set_page_config(page_title="FitCheck 시연", layout="wide", initial_sidebar_state="collapsed")
html(CSS)
html(EXTRA_CSS)
gate()
try:
    page()
except Exception as e:  # 상세 오류·경로를 화면에 내지 않는다
    if _ctl(e):
        raise
    st.error(GENERIC)
