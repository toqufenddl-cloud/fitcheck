"""조항 임베딩(bge-m3, 로컬 Ollama)과 FAISS 검색. 기준일(as_of)·회사 기준으로 거른다.

FAISS를 쓰는 이유: Chroma는 기본으로 익명 사용 통계를 보내므로, 네트워크로 아무것도 보내지 않는 FAISS를 쓴다 (D60).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rag import OLLAMA_URL, assert_local

ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = ROOT / "data" / "llm_cache" / "faiss"   # data/는 git 제외
EMBED_MODEL = "bge-m3"


def _embeddings():
    assert_local()
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)


def build(chunks: pd.DataFrame, path: Path = STORE_DIR):
    """조항 표로 FAISS 색인을 만들어 저장한다. 본문 앞에 문서명·조항을 붙여 임베딩한다."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    docs = [Document(page_content=f"[{r['문서명']} {r['조항']} {r['조항 제목']}]\n{r['본문']}",
                     metadata={k: r[k] for k in chunks.columns if k != "본문"})
            for _, r in chunks.iterrows()]
    store = FAISS.from_documents(docs, _embeddings())
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return store


def load(path: Path = STORE_DIR):
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local(str(path), _embeddings(), allow_dangerous_deserialization=True)  # 직접 만든 로컬 파일만 읽는다


def active(meta: dict, as_of: str, company: str | None) -> bool:
    """기준일에 시행 중이고, 회사 기준 층위라면 지정한 회사 것인지.
    애매한 경우 처리: 시행일이 비어 있으면 시행 중으로 본다. 회사 기준이 아닌 층위(법·당국·상품)는 회사로 거르지 않는다."""
    if meta.get("시행일") and meta["시행일"] > as_of:
        return False
    if meta.get("종료일") and meta["종료일"] < as_of:
        return False
    if company and meta.get("층위") == "회사 기준" and meta.get("회사명") != company:
        return False
    return True


def search(store, query: str, as_of: str, company: str | None = None, k: int = 5, fetch: int = 40) -> list:
    """반환: [(chunk_id, 점수, metadata, 본문)] 상위 k개 (기준일·회사로 거른 뒤)."""
    hits = store.similarity_search_with_score(query, k=fetch)
    out = []
    for doc, score in hits:
        if active(doc.metadata, as_of, company):
            out.append((doc.metadata["chunk_id"], float(score), doc.metadata, doc.page_content))
        if len(out) >= k:
            break
    return out
