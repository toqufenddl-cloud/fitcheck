"""근거 검색(B안) + 로컬 LLM 진단(C안). 모든 처리는 이 컴퓨터 안에서만 한다 (D60).

데이터: AI Hub(한국지능정보사회진흥원) 사업 결과
"""
import os
from urllib.parse import urlparse

ATTRIBUTION = "데이터: AI Hub(한국지능정보사회진흥원) 사업 결과"
OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def assert_local():
    """D60: 원문이 밖으로 나가지 않는지 확인한다. 하나라도 어긋나면 멈춘다.
    - Ollama 주소가 이 컴퓨터(localhost)인지
    - LangSmith 추적(LANGCHAIN_TRACING_V2 / LANGSMITH_TRACING)이 꺼져 있는지
    """
    host = urlparse(OLLAMA_URL).hostname
    if host not in LOCAL_HOSTS:
        raise RuntimeError(f"D60: Ollama 주소가 로컬이 아님: {OLLAMA_URL}")
    for k in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        if os.environ.get(k, "").lower() in ("1", "true", "yes"):
            raise RuntimeError(f"D60: {k}가 켜져 있어 추적 데이터가 외부로 나갈 수 있음")
    return True
