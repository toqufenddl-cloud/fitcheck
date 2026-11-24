"""보고용 숫자 모음: outputs/scorecard.csv (기계용)와 outputs/report_numbers.md (사람용).

각 노트북은 자기 '구분'의 행만 다시 쓴다 (다른 구분 행은 그대로 둔다). report_numbers.md는 scorecard.csv에서 다시 만든다.
숫자만 적는다. 해석 문장은 넣지 않는다 (CLAUDE.md: 해석은 사용자가 쓴다).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
COLS = ["구분", "지표", "Training", "Validation", "전체", "단위", "출처", "근거 결정", "기준일"]
ATTR = "데이터: AI Hub(한국지능정보사회진흥원) 사업 결과"


def update_scorecard(rows: list, section: str, path: Path = OUT / "scorecard.csv") -> pd.DataFrame:
    """section 구분의 행을 rows로 바꾼다.
    애매한 경우 처리: 파일이 없으면 새로 만든다. rows에 없는 열은 빈칸으로 둔다."""
    new = pd.DataFrame(rows).reindex(columns=COLS)
    new["구분"] = section
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        old = old[old["구분"] != section]
        new = pd.concat([old, new.astype(object).where(new.notna(), "").astype(str)], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    write_report_numbers(new)
    return new


TABLE_DIR = OUT / "report_tables"   # 행·열이 지표 표 형식에 안 맞는 집계 표 (예: 06 연령대별 분포)


def save_table(df: pd.DataFrame, name: str, title: str, source: str, basis: str, as_of: str) -> Path:
    """집계 표를 outputs/report_tables/<name>.csv로 저장하고 report_numbers.md를 다시 쓴다.
    애매한 경우 처리: 제목·출처·근거·기준일은 표 아래 첫 줄 메타(#로 시작)로 같이 저장한다. 원문 문장은 넣지 않는다(집계값만)."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    p = TABLE_DIR / f"{name}.csv"
    meta = f"# {title} | 출처 {source} | 근거 {basis} | 기준일 {as_of}\n"
    p.write_text(meta + df.to_csv(index=False), encoding="utf-8-sig")
    sc = OUT / "scorecard.csv"
    if sc.exists():
        write_report_numbers(pd.read_csv(sc, encoding="utf-8-sig", dtype=str, keep_default_na=False))
    return p


def _tables_md() -> list:
    lines = []
    for p in sorted(TABLE_DIR.glob("*.csv")) if TABLE_DIR.exists() else []:
        raw = p.read_text(encoding="utf-8-sig").splitlines()
        meta = raw[0].lstrip("# ").split(" | ") if raw and raw[0].startswith("#") else [p.stem]
        df = pd.read_csv(p, encoding="utf-8-sig", comment=None, skiprows=1 if raw[0].startswith("#") else 0, dtype=str,
                         keep_default_na=False)
        lines += [f"## {meta[0]}", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(r) + " |" for r in df.astype(str).values.tolist()]
        lines += ["", " / ".join(meta[1:]), ""]
    return lines


def write_report_numbers(sc: pd.DataFrame, path: Path = OUT / "report_numbers.md") -> None:
    lines = ["# 보고용 숫자", "", "`outputs/scorecard.csv`와 `outputs/report_tables/`에서 자동 생성. 해석 없이 숫자만 둔다.", ""]
    for sec, g in sc.groupby("구분", sort=False):
        lines += [f"## {sec}", "", "| 지표 | Training | Validation | 전체 | 단위 | 출처 | 근거 |", "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['지표']} | {r['Training']} | {r['Validation']} | {r['전체']} | {r['단위']} | {r['출처']} | {r['근거 결정']} |")
        lines += ["", f"기준일: {g['기준일'].iloc[0]}", ""]
    lines += _tables_md()
    lines += ["---", ATTR, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
