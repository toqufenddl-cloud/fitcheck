"""로컬 LLM(exaone3.5, Ollama) 진단. 검색된 조항만 근거로 쓰고, JSON 스키마로 출력을 강제한다.

- 출력 검증 실패 시 오류 메시지를 붙여 1회 재시도, 다시 실패하면 '형식 실패'.
- 인용 조항 id가 실제로 넣어 준 근거에 없으면 '인용 오류'로 표시한다.
- 프롬프트·원응답은 data/llm_cache/(git 제외)에만 저장한다. 원문 문장이 들어 있기 때문이다 (D60).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jsonschema

from src.rag import OLLAMA_URL, assert_local

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "llm_cache" / "diagnose_raw.jsonl"
LLM_MODEL = "exaone3.5:7.8b"

SCHEMA = {
    "type": "object",
    "properties": {
        "결함": {"type": "string", "enum": ["결함", "결함 아님", "근거 부족"]},
        "처리": {"type": "string", "enum": ["제외", "교정", "태그", "해당 없음"]},
        "사유 코드": {"type": "string", "enum": ["1", "2", "3", "4", "원천 오류", "해당 없음"]},
        "이관 분류": {"type": "string", "enum": ["R1 명확한 희망", "R2 고려 중", "R4 비교 표현", "경험", "회피", "일반 표현", "해당 없음"]},
        "근거": {"type": "string"},
        "인용 조항": {"type": "array", "items": {"type": "string"}},
        "확신도": {"type": "string", "enum": ["높음", "중간", "낮음"]},
    },
    "required": ["결함", "처리", "사유 코드", "이관 분류", "근거", "인용 조항", "확신도"],
}

SYSTEM = """너는 금융상품 추천 상담 데이터(AI 학습용 시나리오)의 품질을 진단하는 검토자다.
규칙
1. 아래 [근거 조항]에 있는 내용만 근거로 쓴다. 근거 조항에 없는 규정이나 조항 번호를 지어내지 않는다.
2. 근거 조항으로 판단할 수 없으면 "결함"을 "근거 부족"으로 둔다.
3. "인용 조항"에는 [근거 조항]의 id(예: R1)만 적는다.
4. 고객 성향의 허용 범위 밖 상품을 추천했거나, 고객이 성향 밖 등급을 원하는데 범위 초과를 알리고 확인 절차로 넘기지 않았으면 결함이다.
5. "필요 시 전문가 상담" 같은 일반 안내 문구는 이관(범위 초과 고지 + 확인 절차)으로 보지 않는다.
6. 성향보다 높은 등급의 과거 경험은 현재 원하는 등급을 대신하지 않는다.
7. 사유 코드: 1=고객 요청·경험을 이유로 범위 밖 추천, 2=예외 설명 없이 범위 밖 추천, 3=성향 판단 오류(성향 복수 등), 4=성향과 허용 범위 짝 불일치.
8. 반드시 지정한 JSON 형식으로만 답한다."""

TASK = {
    "범위": "추천 상품 위험등급이 고객 성향에 맞는지 판단하라. 결함, 처리, 사유 코드를 정하고 이관 분류는 '해당 없음'으로 둔다.",
    "이관": "질문 속 등급 언급이 어떤 성격(R1 명확한 희망 / R2 고려 중 / R4 비교 표현 / 경험 / 회피 / 일반 표현)인지 분류하고, 성향 밖 요청인데 이관 없이 그 등급을 추천했는지로 결함 여부를 정하라.",
    "원천": "상품의 등록 등급과 설명 속 등급이 다를 때, 원천 데이터 확인이 필요한지 판단하라. 확인이 필요하면 결함='결함', 처리='태그', 사유 코드='원천 오류'.",
}


def build_prompt(case: dict, evidence: list) -> str:
    ev = "\n".join(f"[R{i + 1}] {m['문서명']} {m['조항']} ({m['층위']}, 시행 {m['시행일'] or '미상'}): {body[:900]}"
                   for i, (_, _, m, body) in enumerate(evidence))
    facts = {k: case[k] for k in ["cot_id", "성향_cot2", "성향_D21", "cot2 허용 범위", "기준 범위", "추천 등급"]}
    rule = case["규칙"]
    return (f"[사례] {json.dumps(facts, ensure_ascii=False)}\n"
            f"[규칙 판정(참고)] {json.dumps(rule, ensure_ascii=False)}\n"
            f"[필요한 문장] {json.dumps(case['문장'], ensure_ascii=False)}\n"
            + (f"[이관 표현 있음] {case['이관 표현']}\n" if "이관 표현" in case else "")
            + f"\n[근거 조항]\n{ev}\n\n[질문] {TASK[case['유형']]}")


def _llm(num_ctx: int = 4096, keep_alive: str | int = "5m"):
    assert_local()
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=num_ctx,
                      format=SCHEMA, keep_alive=keep_alive)


def diagnose(case: dict, evidence: list, llm=None) -> dict:
    """사례 하나 진단. 반환: 진단 dict + 상태('정상'/'형식 실패'), 인용 오류, 소요 시간."""
    llm = llm or _llm()
    prompt = build_prompt(case, evidence)
    ids = {f"R{i + 1}": cid for i, (cid, _, _, _) in enumerate(evidence)}
    msgs = [("system", SYSTEM), ("human", prompt)]
    t0, out, raw, err = time.time(), None, [], ""
    for attempt in range(2):
        resp = llm.invoke(msgs).content
        raw.append(resp)
        try:
            out = json.loads(resp)
            jsonschema.validate(out, SCHEMA)
            break
        except Exception as e:  # 형식 오류: 1회 재시도
            err = str(e)[:300]
            out = None
            msgs = msgs + [("ai", resp), ("human", f"형식 오류: {err}\n지정한 JSON 형식으로만 다시 답하라.")]
    res = {"case_id": case["case_id"], "cot_id": case["cot_id"], "유형": case["유형"], "분할": case["분할"],
           "상태": "정상" if out else "형식 실패", "재시도": len(raw) - 1, "소요(초)": round(time.time() - t0, 1),
           "검색 조항": "|".join(ids.values())}
    if out:
        cited = out["인용 조항"]
        res.update({k: out[k] for k in ["결함", "처리", "사유 코드", "이관 분류", "확신도"]})
        res["인용 조항"] = "|".join(ids.get(c, f"?{c}") for c in cited)
        res["인용 오류"] = any(c not in ids for c in cited)
        res["근거 요약"] = out["근거"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"case_id": case["case_id"], "prompt_hash": hashlib.sha1(prompt.encode()).hexdigest(),
                            "prompt": prompt, "raw": raw, "error": err}, ensure_ascii=False) + "\n")
    return res


def needs_human(res: dict, rule_defect: bool | None) -> str:
    """사람 확인으로 보낼 이유. 없으면 ''.
    - 형식 실패, 인용 오류, 근거 부족, 확신도 낮음, 규칙과 LLM의 결함 판정이 다름"""
    if res["상태"] != "정상":
        return "형식 실패"
    if res.get("인용 오류"):
        return "인용 오류"
    if res["결함"] == "근거 부족":
        return "근거 부족"
    if res["확신도"] == "낮음":
        return "확신도 낮음"
    if rule_defect is not None and (res["결함"] == "결함") != rule_defect:
        return "규칙과 다름"
    return ""
