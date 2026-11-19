"""규정 문서를 조항 단위로 나눈다 (B안 근거 검색용).

docs/regulations/regulations_index.csv 한 행 = 문서 하나.
열: doc_id, 문서명, 층위(법/당국/회사 기준/상품), 회사명, 시행일, 종료일, 출처 URL, 파일, 비고
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG_DIR = ROOT / "docs" / "regulations"
INDEX_PATH = REG_DIR / "regulations_index.csv"

# '제6조(투자권유 …)', '제6조의2', '제 6 조' 등. 줄 앞(또는 문서 처음)에 나오는 것만 조항 머리로 본다
_ARTICLE = re.compile(r"(?m)^\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(\([^)\n]{0,60}\))?")


def load_index(path: Path = INDEX_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def read_text(path: Path) -> str:
    """PDF는 pypdf로 페이지 텍스트를 잇고, 그 밖(md, txt)은 그대로 읽는다."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def split_articles(text: str, max_chars: int = 1500) -> list:
    """조항 단위로 나눈다. 반환: [(조항 번호, 조항 제목, 본문)]

    애매한 경우 처리
    - 조항 머리('제N조')가 하나도 없으면(상품 설명서 등) 빈 줄 기준 문단을 max_chars 안에서 묶어 '본문-1, 본문-2…'로 둔다.
    - 본문 중간의 '제N조에 따라' 같은 인용은 줄 앞이 아니면 머리로 보지 않는다. 줄 앞에 인용이 오면 잘못 나뉠 수 있다.
    - 한 조항이 max_chars보다 길면 항(①②…) 경계에서 나누고 조항 번호 뒤에 '-1, -2'를 붙인다.
    """
    heads = list(_ARTICLE.finditer(text))
    if not heads:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        out, buf = [], ""
        for p in paras:
            if buf and len(buf) + len(p) > max_chars:
                out.append(buf); buf = ""
            buf = (buf + "\n" + p).strip()
        if buf:
            out.append(buf)
        return [(f"본문-{i + 1}", "", t) for i, t in enumerate(out)]
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        no = re.sub(r"\s+", "", m.group(1))
        title = (m.group(2) or "").strip("() ")
        body = text[m.start():end].strip()
        if len(body) <= max_chars:
            out.append((no, title, body))
            continue
        parts = re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", body)
        buf, n = "", 0
        for p in parts:
            if buf and len(buf) + len(p) > max_chars:
                n += 1; out.append((f"{no}-{n}", title, buf.strip())); buf = ""
            buf += p
        if buf.strip():
            n += 1; out.append((f"{no}-{n}" if n > 1 else no, title, buf.strip()))
    return out


def load_chunks(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """색인의 문서를 모두 읽어 조항 단위 표로 만든다. 파일이 없는 행은 건너뛰고 '파일 없음'으로 알린다."""
    index = load_index() if index is None else index
    rows, missing = [], []
    for _, d in index.iterrows():
        p = REG_DIR / d["파일"]
        if not d["파일"] or not p.exists():
            missing.append(d["doc_id"]); continue
        for no, title, body in split_articles(read_text(p)):
            rows.append({"chunk_id": f"{d['doc_id']}:{no}", "doc_id": d["doc_id"], "문서명": d["문서명"],
                         "조항": no, "조항 제목": title, "층위": d["층위"], "회사명": d["회사명"],
                         "시행일": d["시행일"], "종료일": d["종료일"], "출처 URL": d["출처 URL"], "본문": body})
    if missing:
        print("파일 없음(건너뜀):", missing)
    return pd.DataFrame(rows)
