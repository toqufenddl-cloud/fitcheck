"""FitCheck 시연 앱 (이 컴퓨터 전용, D69). 실행: streamlit run app/fitcheck_app.py

- 분석 탭: 표본 n건에 대해 '기준 문서 검색 → 규칙 생성 → 상품 등급 검색 → 근거 조항 검색 → 규칙 적용'을 단계별로 보여준다.
- 사례 탭: cot_id 하나에 같은 5단계. '원문 보기'는 기본 꺼짐 (원문은 이 화면에만, 저장·전송하지 않음).
- 전체 탭: 7,201건 전체 판정(저장된 등급 사용)과 내려받기.
서버는 127.0.0.1에서만 열고 사용 통계 전송을 끈다 (.streamlit/config.toml).
파이프라인은 src/live.py (CLI python -m src.fitcheck live 와 같은 함수).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.live import load_base, pick_sample, run_live  # noqa: E402

STEPS = {0: "준비", 1: "1. 기준 문서 검색", 2: "2. 규칙 생성", 3: "3. 상품 등급 검색", 4: "4. 근거 조항 검색", 5: "5. 규칙 적용"}
DEMO = ROOT / "docs" / "demo_script.md"

st.set_page_config(page_title="FitCheck 시연", layout="wide")
st.title("FitCheck: 금융상품 추천 학습 데이터 품질 진단")
st.caption("데이터: AI Hub 「금융상품·서비스 및 소비자 특성 데이터」(한국지능정보사회진흥원) · AI 학습용 생성 데이터이며 실제 고객 행동이 아니다")

as_of = st.sidebar.date_input("기준일", value=date(2026, 11, 24)).isoformat()


@st.cache_resource(show_spinner="데이터 판정 준비 중 (약 30초)…")
def base_for(as_of: str) -> dict:
    return load_base(as_of)


base = base_for(as_of)
st.sidebar.write(f"증권 CoT {len(base['j']):,}건 판정 완료")


def show(events, box):
    """run_live 이벤트를 st.status 안에 차례로 그린다. 마지막 done 이벤트를 돌려준다."""
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
    box.update(label=f"완료 ({last['seconds']}초)", state="complete", expanded=False)
    return last


def summary(ev):
    st.subheader("결과 요약: 데이터 기준 vs 최신 문서 기준")
    s = ev["summary"]
    cols = st.columns(3)
    tot = s.set_index("지표")
    cols[0].metric("결함 CoT (데이터 기준)", tot.at["결함 CoT (하나라도)", "데이터 기준"])
    cols[1].metric("결함 CoT (최신 문서 기준)", tot.at["결함 CoT (하나라도)", "최신 문서 기준"])
    cols[2].metric("③ 판정이 바뀐 CoT", tot.at["③ 판정이 바뀐 CoT", "최신 문서 기준"])
    st.dataframe(s, hide_index=True, width="stretch")
    st.info(ev["rule"]["message"])
    ch = ev["result"][ev["result"]["③ 바뀜"]]
    if len(ch):
        st.caption("최신 등급으로 ③ 판정이 바뀐 CoT")
        st.dataframe(ch[["cot_id", "성향", "추천 상품(데이터→최신)", "③ 데이터 기준", "③ 최신 문서 기준"]], hide_index=True,
                     width="stretch")
    st.caption("근거 카드 (문서·조항·시행일·검색 시각)")
    st.dataframe(ev["cards"], hide_index=True, width="stretch")


tab_a, tab_c, tab_all = st.tabs(["분석", "사례", "전체"])

with tab_a:
    c1, c2, c3 = st.columns([1, 1, 2])
    n = c1.number_input("분석 건수", min_value=3, max_value=200, value=30, step=1,
                        help="최신 기준 부적합 사례 min(n/3, 16)건과 공개 기준 범위 밖 최소 3건을 꼭 넣는다")
    mode = c2.radio("등급·문서", ["실시간", "저장값"], horizontal=True)
    if c3.button("분석 실행", type="primary"):
        ids = pick_sample(base, int(n))
        st.write(f"표본 cot_id: {', '.join(map(str, ids))}")
        with st.status("시작", expanded=True) as box:
            st.session_state["a"] = show(run_live(base, ids, live=(mode == "실시간")), box)
    if "a" in st.session_state:
        summary(st.session_state["a"])

with tab_c:
    j = base["j"]
    c1, c2, c3 = st.columns([1, 1, 2])
    cid = c1.selectbox("cot_id", j["cot_id"].astype(int).tolist(), index=None, placeholder="cot_id 선택 또는 입력")
    mode_c = c2.radio("등급·문서 ", ["실시간", "저장값"], horizontal=True)
    if cid is not None:
        row = j.set_index("cot_id").loc[cid]
        st.write(f"저장 판정: ③ {row['기준③ 판정']} · ④ {row['기준④ 판정']} · 이관 {row['이관 판정']} · ② {row['기준② 판정']} · 처리 {row['처리 전체'] or '-'}")
        if st.toggle("원문 보기", value=False, help="이 화면에만 보인다. 저장·전송하지 않는다"):
            c = base["ctx"]["cots"].loc[cid]
            for f in ["question", "cot1", "cot2", "cot3", "answer"]:
                st.markdown(f"**{f}**")
                st.text(str(c[f]))
        if c3.button("이 사례 분석", type="primary"):
            with st.status("시작", expanded=True) as box:
                st.session_state["c"] = show(run_live(base, [int(cid)], live=(mode_c == "실시간")), box)
        if "c" in st.session_state and st.session_state["c"]["result"]["cot_id"].iloc[0] == cid:
            summary(st.session_state["c"])

with tab_all:
    j = base["j"]
    st.write(f"증권 CoT {len(j):,}건 전체 판정 (저장된 등급 사용, 기준일 {as_of})")
    cols = ["cot_id", "분할", "기준① 확인 수", "기준② 판정", "기준③ 판정", "기준③ 처리", "기준④ 판정", "이관 판정", "원천 확인 필요", "처리 전체"]
    k = st.columns(4)
    k[0].metric("③ 범위 밖 제외", int((j["기준③ 처리"] == "제외").sum()))
    k[1].metric("④ 기재 없음", int((j["기준④ 판정"] == "기재 없음").sum()))
    k[2].metric("이관 결함", int((j["이관 판정"] == "결함").sum()))
    k[3].metric("②-a 근거 부족 확정", int((j["기준② 판정"] == "②-a 근거 부족 확정").sum()))
    st.dataframe(j[cols], hide_index=True, width="stretch", height=360)
    st.markdown("**내려받기**")
    files = [("엄격판 Training", "outputs/corrected/엄격판_Training.jsonl"), ("엄격판 Validation", "outputs/corrected/엄격판_Validation.jsonl"),
             ("교정판 Training", "outputs/corrected/교정판_Training.jsonl"), ("교정판 Validation", "outputs/corrected/교정판_Validation.jsonl"),
             ("근거 카드", "outputs/evidence_cards.csv"), ("manifest", f"outputs/runs/{as_of}/manifest.json")]
    dc = st.columns(3)
    for i, (label, rel) in enumerate(files):
        p = ROOT / rel
        if p.exists():
            dc[i % 3].download_button(label, p.read_bytes(), file_name=p.name, key=rel)
        else:
            dc[i % 3].caption(f"{label}: 없음 ({rel}). python -m src.fitcheck run --as-of {as_of} 로 만든다")
