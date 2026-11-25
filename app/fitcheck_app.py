"""FitCheck 시연 앱 (이 컴퓨터 전용, D69). 실행: python -m streamlit run app/fitcheck_app.py

디자인은 docs/design/diagnosis_mockup.html 시안을 따른다 (색·글꼴·간격·모서리·카드 구성).
- 진단 결과: 원본·교정판 품질 점수(D70)와 등급, 진단 의견, 등급 구간, 항목별 막대, 먼저 할 일
- 실시간 분석: 표본 n건에 '기준 문서 검색 → 규칙 생성 → 상품 등급 검색 → 근거 조항 검색 → 규칙 적용', 결과 점수 카드
- 사례 보기: cot_id 하나의 항목별 이행과 점수, 같은 5단계. '원문 보기'는 기본 꺼짐 (원문은 이 화면에만, 저장·전송하지 않음)
- 전체 데이터: 7,201건 전체 판정(저장된 등급 사용)과 내려받기
진단 의견·감점 이유·먼저 할 일은 LLM이 아니라 숫자로 채우는 정해진 문장이다 (아래 문장 함수).
서버는 127.0.0.1에서만 열고 사용 통계 전송을 끈다 (.streamlit/config.toml).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_ui import *  # noqa: E402,F403  (공용 화면 부분: 색·CSS·HTML 조각·정해진 문장)

# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FitCheck 진단", layout="wide", initial_sidebar_state="collapsed")
html(CSS)
as_of = st.sidebar.date_input("기준일", value=date(2026, 11, 24)).isoformat()
base = base_for(as_of)
J = base["j"]
F = facts((OUT / "scorecard.csv").stat().st_mtime + ((OUT / "score_cot.csv").stat().st_mtime if (OUT / "score_cot.csv").exists() else 0))
VER = guide_version()

html(f'<header class="fc-head"><div style="display:flex;align-items:center;gap:14px"><div class="fc-logo">'
     f'<span style="font-size:22px;font-weight:800;color:#FFFFFF">✓</span></div><div>'
     f'<div style="font-size:22px;font-weight:800;letter-spacing:-0.3px">FitCheck</div>'
     f'<div style="font-size:13px;color:#B8C4DC">금융상품 추천 학습 데이터 품질 진단</div></div></div>'
     f'<div style="display:flex;flex-wrap:wrap;gap:8px"><span class="fc-chip">기준일 {as_of}</span>'
     f'<span class="fc-chip">증권 CoT {len(J):,}건</span><span class="fc-chip">기준 문서 3종 · 판정기준서 v{VER}</span></div></header>')


def show(events, box):
    """run_live 이벤트를 st.status 안에 차례로 그린다. 끝나면 접는다. 마지막 done 이벤트를 돌려준다."""
    last, cur, bar = None, None, None
    for ev in events:
        if ev["step"] != cur:
            cur = ev["step"]
            box.update(label=f"진행 중: {STEPS[cur]}")
            box.markdown(f"**{STEPS[cur]}**")
            bar = None
        k = ev["kind"]
        if k == "msg":
            box.write(ev["text"])
        elif k == "warn":
            box.warning(ev["text"])
        elif k == "progress":
            if bar is None:
                bar = box.progress(0.0)
            bar.progress(ev["i"] / ev["n"], text=f"{ev['i']}/{ev['n']} {ev['text'][:40]}")
        elif k == "table":
            box.caption(ev["title"])
            box.dataframe(ev["data"], hide_index=True, width="stretch")
        elif k == "done":
            last = ev
    box.update(label=f"5단계 완료 ({last['seconds']}초) · 펼쳐서 단계 기록 보기", state="complete", expanded=False)
    return last


def live_result(ev: dict) -> None:
    """실시간 분석 결과: 점수 카드 2개(데이터 기준 vs 최신 문서 기준) + 진단 의견, 항목별 막대, 바뀐 사례, 근거 카드(접기)."""
    res = ev["result"]
    a, b = (summarize(item_pass(live_frame(res, J, x))) for x in ("데이터 기준", "최신 문서 기준"))
    n_d, n_l = int((res["③ 데이터 기준"] == "범위 밖").sum()), int((res["③ 최신 문서 기준"] == "범위 밖").sum())
    n_ch = int(res["③ 바뀜"].sum())
    diff = round(b["점수"] - a["점수"], 1)
    op = (f"최신 등급으로 보면 점수가 {abs(diff):.1f}점 {'낮아집니다' if diff < 0 else '높아집니다'}. ③ 판정이 바뀐 CoT {n_ch}건."
          if diff else "최신 등급으로 보아도 점수가 같습니다.")
    html('<div class="fc-grid">'
         + score_card(f"데이터 기준 ({len(res)}건)", a, f"데이터에 적힌 등급으로 판정. ③ 범위 밖 CoT {n_d}건.")
         + score_card(f"최신 문서 기준 ({len(res)}건)", b, f"검색한 문서와 최신 상품 등급으로 판정. ③ 범위 밖 CoT {n_l}건.", diff)
         + dark_card("진단 의견", op, ev["rule"]["message"] + " 규칙 변경은 감지만 하고, 승인 전에는 저장 표로 판정합니다 (D69).")
         + "</div>")
    why = {k: "두 기준 같음" for k in WEIGHT}
    why["③"] = f"범위 밖 CoT {n_d}건 → {n_l}건 (바뀐 CoT {n_ch}건)"
    html(item_bars("항목별 점수: 데이터 기준 vs 최신 문서 기준", a, b, ("데이터 기준", "최신 문서 기준"), why))
    ch = res[res["③ 바뀜"]]
    with st.container(border=True):
        st.markdown("**최신 등급으로 ③ 판정이 바뀐 사례**")
        if len(ch):
            st.dataframe(ch[["cot_id", "성향", "추천 상품(데이터→최신)", "③ 데이터 기준", "③ 최신 문서 기준"]], hide_index=True, width="stretch")
        else:
            st.caption("없음")
    with st.expander(f"근거 카드 {len(ev['cards'])}행 (문서·조항·시행일·검색 시각)"):
        st.dataframe(ev["cards"], hide_index=True, width="stretch")
    with st.expander("결과 요약 표"):
        st.dataframe(ev["summary"], hide_index=True, width="stretch")


tab_d, tab_a, tab_c, tab_all = st.tabs(["진단 결과", "실시간 분석", "사례 보기", "전체 데이터"])

with tab_d:
    o = summarize(item_pass(J))
    cnt = counts(J)
    if F["corr_pass"] is None:
        st.warning("교정판 점수가 없습니다. python -m src.fitcheck run --as-of " + as_of + " 로 만든다")
        cr = None
    else:
        cr = summarize(F["corr_pass"])
        n1c = int((F["corr_pass"]["①"] == 0).sum())
        note = ""
        if F["g_est"]:
            est, lo, hi = F["g_est"]
            note = f"최신 상품 등급을 반영하면 부적합 추천이 최소 약 {est}건 더 있는 것으로 추정됩니다 (표본 추정, 95% CI {lo}~{hi})."
        html('<div class="fc-grid">'
             + score_card("원본 데이터", o, worst_sentence(o))
             + score_card("FitCheck 교정판", cr, corr_sentence(cr, F["n_add"], len(J) - cr["CoT"] - cr["점수 없음"]),
                          round(cr["점수"] - o["점수"], 1))
             + dark_card("진단 의견", opinion(o, cr, n1c), note) + "</div>")
        html(band([("원본", o["점수"], o["등급"]), ("교정판", cr["점수"], cr["등급"])]))
        html(item_bars("항목별 점수와 감점 이유", o, cr, ("원본", "교정판"), why_lines(cnt, cr)))
        html(todo_cards(todos(o, cnt, F)))
    html(f'<div class="fc-foot">{escape(FOOT.format(v=VER))}</div>')

with tab_a:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
        n = c1.number_input("분석 건수", min_value=3, max_value=200, value=30, step=1,
                            help="최신 기준 부적합 사례 min(n/3, 16)건과 공개 기준 범위 밖 최소 3건을 꼭 넣는다")
        mode = c2.radio("등급·문서", ["실시간", "저장값"], horizontal=True, help="실시간 30건은 약 5분 (상품명 조회 0.5초 간격)")
        go = c3.button("분석 실행", type="primary")
    if go:
        ids = pick_sample(base, int(n))
        st.caption(f"표본 cot_id: {', '.join(map(str, ids))}")
        with st.status("시작", expanded=True) as box:
            st.session_state["a"] = show(run_live(base, ids, live=(mode == "실시간")), box)
    if "a" in st.session_state:
        live_result(st.session_state["a"])
    else:
        st.caption("분석 실행을 누르면 5단계 기록이 아래에 나오고, 끝나면 점수 카드가 나온다.")

with tab_c:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
        cid = c1.selectbox("cot_id", J["cot_id"].astype(int).tolist(), index=None, placeholder="cot_id 선택 또는 입력")
        mode_c = c2.radio("등급·문서 ", ["실시간", "저장값"], horizontal=True)
        go_c = c3.button("이 사례 분석", type="primary", disabled=cid is None)
    if cid is not None:
        row = J[J["cot_id"] == cid]
        p = item_pass(row)
        s1 = summarize(p)
        jr = row.iloc[0]
        val = {"①": jr["기준① 법정 미확인"] or "법정 3항목 확인", "②": jr["기준② 판정"], "③": jr["기준③ 판정"],
               "이관": jr["이관 판정"], "④": jr["기준④ 판정"]}
        lines = ""
        for k, nm, w, law in ITEMS:
            v = p.iloc[0][k]
            chip = (("이행", "#DCFCE7", "#166534", ICON_OK) if v == 1 else ("미이행", "#FEE2E2", "#991B1B", ICON_X) if v == 0
                    else ("판정 불가", TRACK, BODY, ICON_WARN))
            pts = "제외" if pd.isna(v) else f"{w * v:g} / {w:g}"
            lines += (f'<div class="fc-row" style="grid-template-columns:minmax(160px,220px) 120px minmax(0,1fr) 90px">'
                      f'<div style="font-size:15px;font-weight:700">{escape(nm)}<div style="font-size:12px;color:{SUB};font-weight:400">'
                      f'{escape(law)}</div></div>'
                      f'<span class="fc-badge" style="background:{chip[1]};color:{chip[2]};font-size:13px">{chip[3]}{chip[0]}</span>'
                      f'<div style="font-size:13px;color:{BODY}">{escape(str(val[k]))}' + (f' · 원천 확인 필요' if k == "③" and jr["원천 확인 필요"] else "")
                      + f'</div><div style="font-size:13px;text-align:right;font-weight:700">{pts}</div></div>')
        html('<div class="fc-grid" style="grid-template-columns:minmax(260px,1fr) minmax(0,2.4fr)">'
             + score_card(f"cot_id {cid} · {jr['분할']}", s1, f"성향 {jr['성향_P5'] or '미확정'} · 처리 {jr['처리 전체'] or '없음'}")
             + f'<div class="fc-sec" style="margin:0;gap:0"><h2 class="fc-h2" style="padding-bottom:10px">항목별 이행 (저장 판정)</h2>{lines}</div>'
             + "</div>")
        if st.toggle("원문 보기", value=False, help="이 화면에만 보인다. 저장·전송하지 않는다"):
            c = base["ctx"]["cots"].loc[cid]
            with st.container(border=True):
                for fld in ["question", "cot1", "cot2", "cot3", "answer"]:
                    st.markdown(f"**{fld}**")
                    st.text(str(c[fld]))
        if go_c:
            with st.status("시작", expanded=True) as box:
                st.session_state["c"] = show(run_live(base, [int(cid)], live=(mode_c == "실시간")), box)
        if "c" in st.session_state and st.session_state["c"]["result"]["cot_id"].iloc[0] == cid:
            live_result(st.session_state["c"])

with tab_all:
    html(stat_cards([
        ("③ 범위 밖 제외", f"{int((J['기준③ 처리'] == '제외').sum()):,}", "CoT, 엄격판·교정판에서 뺌"),
        ("④ 기재 없음", f"{int((J['기준④ 판정'] == '기재 없음').sum()):,}", "CoT, 교정판에 등급 문장 추가"),
        ("이관 결함", f"{int((J['이관 판정'] == '결함').sum()):,}", "CoT, 태그로 표시"),
        ("②-a 근거 부족 확정", f"{int((J['기준② 판정'] == '②-a 근거 부족 확정').sum()):,}", "CoT, 태그로 표시"),
    ]))
    with st.container(border=True):
        st.markdown(f"**증권 CoT {len(J):,}건 전체 판정** (저장된 등급 사용, 기준일 {as_of})")
        cols = ["cot_id", "분할", "기준① 확인 수", "기준② 판정", "기준③ 판정", "기준③ 처리", "기준④ 판정", "이관 판정", "원천 확인 필요", "처리 전체"]
        st.dataframe(J[cols], hide_index=True, width="stretch", height=360)
    with st.container(border=True):
        st.markdown("**내려받기**")
        files = [("엄격판 Training", "outputs/corrected/엄격판_Training.jsonl"), ("엄격판 Validation", "outputs/corrected/엄격판_Validation.jsonl"),
                 ("교정판 Training", "outputs/corrected/교정판_Training.jsonl"), ("교정판 Validation", "outputs/corrected/교정판_Validation.jsonl"),
                 ("근거 카드", "outputs/evidence_cards.csv"), ("CoT별 품질 점수", "outputs/score_cot.csv"),
                 ("manifest", f"outputs/runs/{as_of}/manifest.json")]
        dc = st.columns(3)
        for i, (label, rel) in enumerate(files):
            fp = ROOT / rel
            if fp.exists():
                dc[i % 3].download_button(label, fp.read_bytes(), file_name=fp.name, key=rel, width="stretch")
            else:
                dc[i % 3].caption(f"{label}: 없음 ({rel}). python -m src.fitcheck run --as-of {as_of} 로 만든다")
