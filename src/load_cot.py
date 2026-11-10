"""CoT(상담 시나리오)와 상품 원천데이터를 zip에서 바로 읽어 데이터프레임으로 만드는 함수.

- 원본 zip은 압축을 풀지 않는다.
- 판정은 하지 않는다. 읽기와 형식 정리만 한다 (판정 함수는 03 단계에서 따로 만든다).
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

# 프로젝트 루트 = 이 파일(src/load_cot.py)의 한 단계 위
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# zip 안의 한글 파일명은 cp949로 저장되어 있다
_ZIP_ENCODING = "cp949"


def _open_zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path, metadata_encoding=_ZIP_ENCODING)


def _group_label(zip_path: Path, prefix: str) -> str:
    """zip 파일명에서 분류 이름만 떼어 낸다.
    예: 'TL_1.CoT_1.증권_01.고객특성 강조형.zip' → '증권_01.고객특성 강조형'
    """
    stem = zip_path.stem
    return stem.split(prefix, 1)[1] if prefix in stem else stem


def _parse_product_names(value):
    """product_names를 리스트로 바꾼다.

    애매한 경우 처리:
    - 이미 리스트이면 그대로 쓴다 (대부분의 파일).
    - 문자열이면 JSON 리스트로 읽어 본다 (데이터 문서 예시 형식).
    - JSON으로 읽히지 않는 문자열(따옴표가 깨진 경우)은 고쳐서 추측하지 않는다.
      빈 리스트를 돌려주고 parse_ok=False로 표시한다. 원래 값은 product_names_raw에 남는다.
    """
    if isinstance(value, list):
        return value, True
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed, True
        except json.JSONDecodeError:
            pass
    return [], False


def load_cot(raw_dir: Path = RAW_DIR, splits=("Training", "Validation")) -> pd.DataFrame:
    """CoT JSON을 모두 읽어 한 행 = CoT 1건인 데이터프레임을 돌려준다.

    추가 컬럼
    - split: Training / Validation
    - query_group: 폴더(zip) 기준 질문유형 (예: '보험_02.투자성 상품 추천형')
    - file_name: zip 안의 파일명
    - product_names_raw: 원래 값 그대로
    - product_names: 리스트로 정리한 값
    - product_names_ok: 리스트로 정리되었는지 여부
    """
    rows = []
    for split in splits:
        prefix = "TL_1.CoT_" if split == "Training" else "VL_1.CoT_"
        label_dir = Path(raw_dir) / split / "02.라벨링데이터"
        for zp in sorted(label_dir.glob(f"{prefix}*.zip")):
            group = _group_label(zp, "CoT_1." if "CoT_1." in zp.name else "CoT_2.")
            with _open_zip(zp) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    rec = json.loads(z.read(info).decode("utf-8-sig"))
                    rec["split"] = split
                    rec["query_group"] = group
                    rec["file_name"] = info.filename
                    rec["product_names_raw"] = rec.get("product_names")
                    rec["product_names"], rec["product_names_ok"] = _parse_product_names(
                        rec.get("product_names"))
                    rows.append(rec)
    df = pd.DataFrame(rows)
    front = ["split", "query_group", "file_name", "cot_id"]
    return df[front + [c for c in df.columns if c not in front]]


def load_products(raw_dir: Path = RAW_DIR, split: str = "Training") -> pd.DataFrame:
    """상품 원천데이터 JSON을 모두 읽어 한 행 = 상품 JSON 1개인 데이터프레임을 돌려준다.

    증권 상품과 보험 상품은 키 구성이 다르다. 없는 키는 NaN으로 남긴다.
    추가 컬럼
    - domain: 증권 / 보험
    - product_group: 원천데이터 분류 (예: 주식형, 변액)
    - file_name: zip 안의 파일명
    """
    prefix = "TS_1.상품_" if split == "Training" else "VS_1.상품_"
    src_dir = Path(raw_dir) / split / "01.원천데이터"
    rows = []
    for zp in sorted(src_dir.glob(f"{prefix}*.zip")):
        # 예: 'TS_1.상품_2.보험_07.변액' → domain='보험', product_group='변액'
        m = re.search(r"상품_\d\.(증권|보험)_\d+\.(.+)$", zp.stem)
        domain, group = (m.group(1), m.group(2)) if m else ("?", zp.stem)
        with _open_zip(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                rec = json.loads(z.read(info).decode("utf-8-sig"))
                rec["domain"] = domain
                rec["product_group"] = group
                rec["file_name"] = info.filename
                rows.append(rec)
    df = pd.DataFrame(rows)
    front = ["domain", "product_group", "file_name"]
    return df[front + [c for c in df.columns if c not in front]]
