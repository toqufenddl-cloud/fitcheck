"""D61 2차 시도: LLM 역할 축소. 로컬 LLM(exaone3.5)은 두 가지만 판단한다.

(a) 고객의 등급 언급 문장 분류: 희망 / 고려 중 / 경험 / 회피 / 비교 표현 / 일반 표현
(b) answer에 범위 초과를 알리고 확인 절차로 넘기는 이관 표현이 있는지 (D56 정의)

범위 안/밖, 사유 코드, 원천 등급 비교는 규칙 결과(judge)를 그대로 쓰고 LLM에 묻지 않는다.
결함 여부는 combine()이 "규칙의 범위 판단 + LLM 분류"로 결합한다.
프롬프트에 규칙 판정을 넣지 않는다. 예시(few-shot)는 검증 표본 밖 CoT에서만 고른다 (src/rag/fewshot.csv).
원문은 이 컴퓨터 안의 LLM에만 넣고, 원응답은 data/llm_cache/(git 제외)에만 저장한다 (D60).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import jsonschema
import pandas as pd

from src.extract import _mask_ranges
from src.rag import OLLAMA_URL, assert_local
from src.transfer import sentence_mentions

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "llm_cache" / "classify_raw.jsonl"
LLM_MODEL = "exaone3.5:7.8b"
CLASSES = ["희망", "고려 중", "경험", "회피", "비교 표현", "일반 표현"]
_SPLIT = re.compile(r"(?<=[.!?])\s+")

SCHEMA = {
    "type": "object",
    "properties": {
        "언급 분류": {"type": "string", "enum": CLASSES},
        "이관 표현": {"type": "boolean"},
        "인용 조항": {"type": "array", "items": {"type": "string"}},
        "확신도": {"type": "string", "enum": ["높음", "중간", "낮음"]},
    },
    "required": ["언급 분류", "이관 표현", "인용 조항", "확신도"],
}

SYSTEM = """너는 금융상품 상담 시나리오(AI 학습용 데이터)를 읽고 두 가지만 판단한다.

(a) 고객이 질문에서 위험등급을 언급한 문장의 성격을 하나 고른다.
- 희망: 그 등급(또는 등급 명칭)의 상품을 원하거나 추천·비교해 달라고 요청한다.
- 고려 중: 그 등급의 위험을 감수·수용·감내할 의향이 있다거나, 투자를 고민·검토 중이라고 말한다.
- 경험: 그 등급 상품을 과거에 보유·투자한 경험만 말한다. 경험과 함께 그 등급을 요청하면 희망 또는 고려 중으로 고른다.
- 회피: 그 등급(또는 '너무 높은 위험')을 피하고 싶거나 부담스럽다고 말한다.
- 비교 표현: '예금(적금)보다 다소 높은 위험'처럼 다른 상품과 비교해 위험 수준을 말한다.
- 일반 표현: 등급 명칭이 특정 등급을 원하는 뜻 없이 일반적인 설명으로 쓰였다.
(b) 추천 답변(answer)에, 고객이 원하는 등급이 고객 투자성향의 허용 범위를 넘는다는 사실을 알리고 상담사 확인·서면 확인 같은 확인 절차로 넘기는 표현이 있으면 true.
- "필요 시 전문가와 상담하세요" 같은 일반 안내는 이관이 아니다 (false).
- 범위 초과를 알리는 내용과 확인 절차가 함께 있어야 true다.

[근거 조항]은 판단의 배경이다. "인용 조항"에는 근거 조항 id(예: R1)만 적고, 관련 조항이 없으면 빈 목록으로 둔다.
반드시 지정한 JSON 형식으로만 답한다."""


def mention_sentences(question: str, max_n: int = 3, max_len: int = 300) -> list:
    out = []
    for s in _SPLIT.split(str(question)):
        if sentence_mentions(_mask_ranges(s)):
            out.append(s.strip()[:max_len])
    return out[:max_n]


def fewshot_messages(cot: pd.DataFrame, path: Path = Path(__file__).resolve().parent / "fewshot.csv") -> list:
    """검증 표본 밖 예시를 (human, ai) 메시지 쌍으로 만든다. 예시는 (a) 분류만 보여 주고 이관 표현은 false로 둔다
    (데이터에 이관 표현이 있는 CoT가 0건이라 실제 예시가 없다, D56)."""
    fs = pd.read_csv(path, encoding="utf-8", dtype={"cot_id": int})
    q = cot.set_index("cot_id")["question"]
    msgs = []
    for _, r in fs.iterrows():
        msgs.append(("human", "[예시] 질문 속 등급 언급 문장: " + json.dumps(mention_sentences(q[r["cot_id"]]), ensure_ascii=False)
                     + "\n[예시] answer: (생략)"))
        msgs.append(("ai", json.dumps({"언급 분류": r["언급 분류"], "이관 표현": False, "인용 조항": [], "확신도": "높음"},
                                      ensure_ascii=False)))
    return msgs


def query_for(propensity: str, grades: list) -> str:
    """사례 값으로 만든 검색 질의 (사례마다 다름)."""
    g = ", ".join(f"{x}등급" for x in grades) if grades else "높은 위험등급"
    return f"{propensity} 투자자가 {g} 상품을 원하는 경우 투자성향에 적합하지 않은 상품 권유 금지와 확인 절차"


def build_prompt(case: dict, evidence: list) -> str:
    ev = "\n".join(f"[R{i + 1}] {m['문서명']} {m['조항']} ({m['층위']}): {body[:600]}"
                   for i, (_, _, m, body) in enumerate(evidence))
    return (f"[고객 투자성향] {case['성향']} (허용 위험등급: {case['허용 등급']})\n"
            f"[질문 속 등급 언급 문장] {json.dumps(case['문장'], ensure_ascii=False)}\n"
            f"[answer] {case['answer'][:1500]}\n\n[근거 조항]\n{ev}\n\n"
            "[질문] (a) 등급 언급 문장의 성격과 (b) answer의 이관 표현 여부를 판단하라.")


def _llm(num_ctx: int = 6144):
    assert_local()
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=num_ctx, format=SCHEMA)


def classify(case: dict, evidence: list, fewshot: list, llm=None) -> dict:
    llm = llm or _llm()
    prompt = build_prompt(case, evidence)
    ids = {f"R{i + 1}": cid for i, (cid, _, _, _) in enumerate(evidence)}
    msgs = [("system", SYSTEM)] + fewshot + [("human", prompt)]
    t0, out, raw, err = time.time(), None, [], ""
    for _ in range(2):
        resp = llm.invoke(msgs).content
        raw.append(resp)
        try:
            out = json.loads(resp); jsonschema.validate(out, SCHEMA); break
        except Exception as e:
            err, out = str(e)[:300], None
            msgs = msgs + [("ai", resp), ("human", f"형식 오류: {err}\n지정한 JSON 형식으로만 다시 답하라.")]
    res = {"cot_id": case["cot_id"], "상태": "정상" if out else "형식 실패", "재시도": len(raw) - 1,
           "소요(초)": round(time.time() - t0, 1), "검색 top3": "|".join(list(ids.values())[:3])}
    if out:
        res.update({"LLM 분류": out["언급 분류"], "LLM 이관 표현": out["이관 표현"], "확신도": out["확신도"],
                    "인용 조항": "|".join(ids.get(c, f"?{c}") for c in out["인용 조항"]),
                    "인용 오류": any(c not in ids for c in out["인용 조항"])})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cot_id": case["cot_id"], "prompt_hash": hashlib.sha1(prompt.encode()).hexdigest(),
                            "prompt": prompt, "raw": raw, "error": err}, ensure_ascii=False) + "\n")
    return res


def combine(llm_class: str, llm_transfer: bool, outside_grades: set, literal_outside: set, rec_grades: set) -> str:
    """결함 여부를 코드로 결합한다 (규칙의 범위 판단 + LLM 분류).

    - outside_grades: 규칙이 본 '성향 밖' 언급 등급(해석 등급), literal_outside: R4 문자 그대로의 성향 밖 등급
    - 결함: LLM 분류가 희망·고려 중·경험이고, 성향 밖 언급 등급을 실제 추천했고, LLM이 이관 표현 없음으로 봄
      비교 표현이면 문자 그대로의 성향 밖 등급(예: 3등급)을 추천했을 때 결함 (D50 R4)
    - 결함 아님: 회피·일반 표현, 성향 밖 언급이 없음, 언급 등급을 추천하지 않음, 이관 표현 있음
    애매한 경우 처리: 형식 실패면 '판정 불가'.
    """
    if llm_class is None:
        return "판정 불가"
    if llm_transfer:
        return "결함 아님"
    if llm_class in ("희망", "고려 중", "경험") and (outside_grades & rec_grades):
        return "결함"
    if llm_class == "비교 표현" and ((literal_outside | outside_grades) & rec_grades):
        return "결함"
    return "결함 아님"
