"""근거 카드(마크다운). 사례 값·판정·인용 조항만 적고 CoT 원문 문장은 넣지 않는다 (D60)."""
from __future__ import annotations

from pathlib import Path

from src.rag import ATTRIBUTION

ROOT = Path(__file__).resolve().parents[2]
CARD_DIR = ROOT / "outputs" / "llm" / "evidence_cards"


def write_card(case: dict, res: dict, evidence: list, as_of: str, company: str | None) -> Path:
    cited = set(res.get("인용 조항", "").split("|"))
    lines = [f"# 근거 카드 — cot_id {case['cot_id']} ({case['유형']}, {case['분할']})", "",
             f"- 기준일(as_of): {as_of} / 회사 기준: {company or '공개기준'}",
             f"- 성향(D21): {case['성향_D21']} / cot2 허용 범위: {case['cot2 허용 범위']} / 기준 범위: {case['기준 범위']}",
             f"- 추천 등급: {case['추천 등급']}",
             f"- 규칙 판정: {case['규칙']}",
             f"- LLM 판정: 결함={res.get('결함')}, 처리={res.get('처리')}, 사유 코드={res.get('사유 코드')}, "
             f"이관 분류={res.get('이관 분류')}, 확신도={res.get('확신도')}, 상태={res['상태']}",
             "- LLM 근거 요약: 고객 문장을 인용할 수 있어 카드에는 넣지 않는다 (data/llm_cache, git 제외)",
             "", "## 검색된 조항 (★ = 인용)", ""]
    for cid, score, m, body in evidence:
        mark = "★ " if cid in cited else ""
        lines += [f"### {mark}{m['문서명']} {m['조항']} {m['조항 제목']}",
                  f"- 층위: {m['층위']} / 회사: {m['회사명'] or '-'} / 시행: {m['시행일'] or '미상'} / 출처: {m['출처 URL'] or '-'}",
                  f"- 거리 점수: {score:.3f}", "", body[:1200], ""]
    lines += ["---", ATTRIBUTION, ""]
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    p = CARD_DIR / f"{case['cot_id']}_{case['유형']}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
