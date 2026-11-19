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
    - 본문 중간의 '제N조에 따라' 같은 인용은 줄 앞이 아니면 머리로 보지 않는다. 줄 앞 인용을 거르려고 제목 괄호가 있는 머리만 쓴다.
    - '부칙' 줄 뒤에 나오는 '제N조(…)'는 '부칙제N조'로 둔다 (본칙 조항과 섞이지 않게).
    - 장 제목('제1장 총칙')은 조항으로 나누지 않고 앞 조항 본문에 붙는다.
    - 한 조항이 max_chars보다 길면 항(①②…) 경계에서 나누고 조항 번호 뒤에 '-1, -2'를 붙인다.
    """
    heads = list(_ARTICLE.finditer(text))
    # 제목 괄호가 있는 머리만 쓴다 (줄 앞의 인용 '제 21조에 따라' 등을 거르기 위해). 제목 있는 머리가 하나도 없으면 전부 쓴다
    titled = [m for m in heads if m.group(2)]
    heads = titled or heads
    # 부칙 뒤의 '제N조(…)' 언급은 본칙 조항과 구분해 '부칙 제N조'로 둔다
    bm = re.search(r"(?m)^\s*부\s*칙", text)
    bstart = bm.start() if bm else len(text) + 1
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
        if m.start() >= bstart:
            no = "부칙" + no
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
        splitter = SPLITTERS.get(d["doc_id"], split_articles)
        for no, title, body in splitter(read_text(p)):
            rows.append({"chunk_id": f"{d['doc_id']}:{no}", "doc_id": d["doc_id"], "문서명": d["문서명"],
                         "조항": no, "조항 제목": title, "층위": d["층위"], "회사명": d["회사명"],
                         "시행일": d["시행일"], "종료일": d["종료일"], "출처 URL": d["출처 URL"], "본문": body})
    if missing:
        print("파일 없음(건너뜀):", missing)
    df = pd.DataFrame(rows)
    # 같은 id가 여러 번이면(개정마다 부칙이 따로 있음) 두 번째부터 '#2, #3…'을 붙인다
    n = df.groupby("chunk_id").cumcount()
    df.loc[n > 0, "chunk_id"] = df.loc[n > 0, "chunk_id"] + "#" + (n[n > 0] + 1).astype(str)
    return df


# ---------------------------------------------------------------------------
# 문서별 분할 (doc_id로 고른다)
# ---------------------------------------------------------------------------

_PARA = re.compile(r"(?m)^\s*([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")


def split_law(text: str, max_chars: int = 1500) -> list:
    """법률 txt: '제N조(제목)' 단위로 나눈 뒤 그 안을 항(①②③) 단위로 나눈다. 반환 [(조항 id, 제목, 본문)]

    - 조항 id: '제17조③'. 항이 없는 조항은 '제17조'.
    - 부칙 줄 뒤의 조항은 '부칙제1조'.
    애매한 경우 처리
    - 항 첫 줄 앞에 조 머리 문장(예: '제17조(적합성원칙)')이 붙으므로, 각 항 조각 앞에 '제N조(제목)'을 붙여 검색 문맥을 남긴다.
    - 항 하나가 max_chars보다 길면(예: 제19조①의 각 호 목록) 그대로 둔다 (호·목 단위로는 나누지 않는다).
    """
    out = []
    for no, title, body in split_articles(text, max_chars=10 ** 6):
        paras = list(_PARA.finditer(body))
        if not paras:
            out.append((no, title, body)); continue
        head = f"{no}({title})" if title else no
        pre = body[:paras[0].start()].strip()
        # ①은 보통 조 머리와 같은 줄에 있어 줄 앞 항으로 잡히지 않는다 → 머리 부분을 ① 조각으로 둔다
        if pre:
            out.append((f"{no}①" if "①" in pre else no, title, pre))
        for i, m in enumerate(paras):
            end = paras[i + 1].start() if i + 1 < len(paras) else len(body)
            out.append((f"{no}{m.group(1)}", title, f"{head} {body[m.start():end].strip()}"))
    return out


# 절 제목: '10. 투자권유 절차', '17-①. 계약서류의 교부'. 제목 안 '·'도 허용 (목차 점선 줄은 미리 지운다)
_KOFIA_SEC = re.compile(r"(?m)^\s*(\d{1,2})(-[①-⑳])?\.\s+([^\n]{1,40}?)\s*$")
_KOFIA_ITEM = re.compile(r"(?m)^\s*(\d{1,2})\)\s")
_KOFIA_REF = re.compile(r"(?m)^\s*※\s*회사참고사항\s*(\d{1,2}-\d{1,2})")


def split_kofia(text: str, max_chars: int = 1500) -> list:
    """금융투자협회 표준투자권유준칙 PDF: 번호 절('10. 투자권유 절차') → 항목('4)')·'※ 회사참고사항 10-4' 블록 단위.

    - 조각 id: '10.4)', '회사참고사항 10-4', 절 머리말만 있는 부분은 '10.'
    애매한 경우 처리
    - 목차는 점선('···')이 있는 줄과 첫 본문 쪽 표시('- 1 -') 앞부분을 지워서 뺀다. 쪽 번호 줄('- 9 -')도 지운다.
    - 절 번호는 앞 절보다 같거나 1 큰 경우만 절 머리로 본다 (본문 속 '1. …' 목록을 절로 잘못 보지 않게).
    - 회사참고사항은 '※'로 시작하는 줄만 블록 머리로 본다. 본문 속 '회사참고사항 10-4의 …' 언급은 머리가 아니다.
    - 같은 id가 다시 나오면 load_chunks에서 '#2'를 붙인다. max_chars를 넘는 조각은 그대로 둔다.
    """
    first = re.search(r"(?m)^\s*-\s*1\s*-\s*$", text)
    body = text[first.end():] if first else text
    body = "\n".join(l for l in body.splitlines() if "···" not in l and not re.fullmatch(r"\s*-\s*\d+\s*-\s*", l))
    heads, last = [], 0
    for m in _KOFIA_SEC.finditer(body):
        n = int(m.group(1))
        if last <= n <= last + 1:
            heads.append(("sec", m.start(), f"{n}{m.group(2) or ''}", m.group(3).strip())); last = n
    out = []
    for i, (_, s, sec, title) in enumerate(heads):
        e = heads[i + 1][1] if i + 1 < len(heads) else len(body)
        sb = body[s:e]
        marks = sorted([(m.start(), f"{sec}.{m.group(1)})") for m in _KOFIA_ITEM.finditer(sb)] +
                       [(m.start(), f"회사참고사항 {m.group(1)}") for m in _KOFIA_REF.finditer(sb)])
        if not marks or marks[0][0] > 0:
            pre = sb[: marks[0][0] if marks else len(sb)].strip()
            if pre:
                out.append((f"{sec}.", title, pre))
        for k, (ms, cid) in enumerate(marks):
            me = marks[k + 1][0] if k + 1 < len(marks) else len(sb)
            out.append((cid, title, f"{sec}. {title} — {sb[ms:me].strip()}"))
    return out


SPLITTERS = {"LAW-FCPA": split_law, "LAW-FCPA-ENF": split_law, "REG-KOFIA": split_kofia}
