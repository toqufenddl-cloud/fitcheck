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
        new = pd.concat([old, new.astype(str).replace("nan", "")], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    write_report_numbers(new)
    return new


def write_report_numbers(sc: pd.DataFrame, path: Path = OUT / "report_numbers.md") -> None:
    lines = ["# 보고용 숫자", "", "`outputs/scorecard.csv`에서 자동 생성. 해석 없이 숫자만 둔다.", ""]
    for sec, g in sc.groupby("구분", sort=False):
        lines += [f"## {sec}", "", "| 지표 | Training | Validation | 전체 | 단위 | 출처 | 근거 |", "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['지표']} | {r['Training']} | {r['Validation']} | {r['전체']} | {r['단위']} | {r['출처']} | {r['근거 결정']} |")
        lines += ["", f"기준일: {g['기준일'].iloc[0]}", ""]
    lines += ["---", ATTR, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
