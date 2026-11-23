"""사람 판정용 작성 파일(xlsx)을 만들고, 저장된 판정을 검수표 CSV에 합친다.

검수표 CSV는 엑셀로 열지 않는다(날짜·인코딩 변환 위험). 사람은 xlsx에만 적고, CSV는 이 코드로만 고친다.
작성 파일에는 규칙 결과·LLM 결과·표본 설계 그룹을 넣지 않고, 행 순서를 섞는다 (판정 편향 방지).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_CHOICES = {
    "내 분류": ["희망", "고려 중", "경험", "회피", "비교 표현", "일반 표현"],
    "내 이관 표현": ["있음", "없음"],
    "내 판정": ["결함", "결함 아님"],
}
XLSX_COLS = ["cot_id", "질문", "답변", "내 분류", "내 이관 표현", "내 판정", "메모"]


def make_label_xlsx(cot_ids, cots: pd.DataFrame, path: Path, seed: int = 1) -> bool:
    """cot_ids의 질문·답변 원문으로 작성 파일을 만든다. 순서는 seed로 섞는다.
    cots: cot_id를 인덱스로 하고 question, answer 열이 있는 표.
    애매한 경우 처리: 파일이 이미 있으면 덮어쓰지 않고 False를 돌려준다 (사람이 적은 내용 보호)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    path = Path(path)
    if path.exists():
        return False
    ids = pd.Series(list(cot_ids)).sample(frac=1, random_state=seed).tolist()
    wb = Workbook(); ws = wb.active; ws.title = "판정"
    ws.append(XLSX_COLS)
    for cid in ids:
        x = cots.loc[cid]
        ws.append([int(cid), str(x["question"]), str(x["answer"]), None, None, None, None])
    n = len(ids) + 1
    for col, name in zip("DEF", ["내 분류", "내 이관 표현", "내 판정"]):
        dv = DataValidation(type="list", formula1='"' + ",".join(LABEL_CHOICES[name]) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="허용되지 않는 값",
                            error="목록에서 고르세요: " + ", ".join(LABEL_CHOICES[name]))
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
    for col, w in zip("ABCDEFG", [9, 60, 80, 12, 12, 12, 30]):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "B2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return True


def merge_label_xlsx(xlsx: Path, csv: Path) -> pd.DataFrame:
    """작성 파일의 내 분류·내 이관 표현·내 판정·메모를 cot_id로 검수표 CSV에 합쳐 저장하고, 합친 표를 돌려준다.
    애매한 경우 처리:
    - 작성 파일이 기준이다. 작성 파일에서 비운 칸은 CSV에서도 비운다 (CSV는 코드로만 고치므로 충돌하지 않는다).
    - 허용 목록 밖 값(복사·붙여넣기로 드롭다운을 우회한 경우)이나 cot_id 불일치가 있으면 아무것도 쓰지 않고 멈춘다.
    - 앞뒤 공백은 지운다. 그 밖의 표기 차이(예: '결함아님')는 고치지 않고 오류로 본다.
    - 엑셀이 파일을 열고 있어 읽을 수 없으면 PermissionError를 그대로 낸다 (엑셀을 닫고 다시 실행)."""
    lab = pd.read_excel(xlsx, dtype=str, keep_default_na=False)
    lab = lab.apply(lambda s: s.str.strip())
    base = pd.read_csv(csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if set(lab["cot_id"]) != set(base["cot_id"]) or lab["cot_id"].duplicated().any():
        raise ValueError(f"cot_id 불일치: 작성 파일 {len(lab)}행, CSV {len(base)}행")
    bad = [(r["cot_id"], k, r[k]) for _, r in lab.iterrows() for k, v in LABEL_CHOICES.items() if r[k] and r[k] not in v]
    if bad:
        raise ValueError(f"허용 목록 밖 값 (아무것도 저장하지 않음): {bad}")
    cols = list(LABEL_CHOICES) + ["메모"]
    out = base.drop(columns=[c for c in cols if c in base.columns]).merge(lab[["cot_id"] + cols], on="cot_id", how="left")
    out = out[[c for c in base.columns if c not in cols] + cols]
    tmp = Path(csv).with_suffix(".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(csv)
    return out
