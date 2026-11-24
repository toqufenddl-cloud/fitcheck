"""07 결함 요인 모델 (D66). 판정이 아니라 요인 분석용.

설계·피처는 docs/reference/결함요인모델_참고구현.py를 그대로 따른다 (경로만 프로젝트에 맞춤).
- 라벨 (judgments.csv): A = 기준③ 처리 '제외', B = 기준④ 판정 '기재 없음'.
- 규칙 누수 방지: 라벨을 만드는 규칙의 입력(고객 성향, 추천 상품 등급, cot2 허용 범위)은 피처에 넣지 않는다.
- 학습 Training / 평가 Validation. 로지스틱 회귀(표준화, class_weight balanced) + 의사결정나무(깊이 3, min_samples_leaf 50).
애매한 경우 처리:
- 상품 유형은 추천 상품명(앞뒤 공백 제거)이 원천 상품명과 같을 때만 쓴다. 같은 이름이 여러 개면 마지막 것 (참고 구현과 같음).
- 연령은 메타 age를 숫자로 바꾼다 ('50대 이상'→5, '80대 이상'→8). 못 바꾸면 그 행을 뺀다 (참고 구현의 dropna).
- 경험·이해도 언급과 고객 진술 성향은 기준 ①② 함수(judge12)의 결과를 쓴다 (③④ 라벨과 입력이 겹치지 않음).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

AGE = {"20대": 2, "30대": 3, "40대": 4, "50대": 5, "50대 이상": 5, "60대": 6, "60대 이상": 6, "70대": 7, "80대 이상": 8}
TYPES = ["주식형", "채권형", "재간접", "단기금융", "파생형", "퇴직연금"]
LABELS = {"y_범위밖제외": "A 범위 밖 제외 (기준③)", "y_등급설명없음": "B 위험등급 기재 없음 (기준④)"}


def build_features(cots: pd.DataFrame, prod: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:
    """cots: judge()의 ctx['cots'] (cot_id 색인, question·answer·product_names·query_type·age·gender·①·고객 진술 성향).
    prod: load_products(), judgments: outputs/judgments.csv. 반환: 피처 + 라벨 (원문 없음)."""
    ptype = {str(n).strip(): t for n, t in zip(prod["product_name"], prod["product_type"])}
    j = judgments.set_index("cot_id")
    rows = []
    for cid, d in cots.iterrows():
        q = str(d["question"]); pn = d["product_names"] if isinstance(d["product_names"], list) else []
        types = {ptype[n.strip()] for n in pn if n.strip() in ptype}
        x = dict(cot_id=cid, split=j.at[cid, "분할"],
                 유형_조건기반=int(d["query_type"] == "투자성향 및 조건기반형"), 유형_상품비교=int(d["query_type"] == "상품 비교 추천형"),
                 연령=AGE.get(d["age"], np.nan), 여성=int(d["gender"] == "여"), 추천수=len(pn),
                 질문_숫자등급=int(bool(re.search(r"[1-6]\s*등급", q))),
                 질문_높은위험명칭=int(bool(re.search(r"(매우\s*높은|다소\s*높은|높은)\s*위험", q))),
                 질문_고려표현=int(bool(re.search(r"감수|수용|감내|고민|고려|거부감", q))),
                 질문_회피표현=int(bool(re.search(r"부담|원하지\s*않|원치\s*않|피하|꺼리|지양|너무\s*높은|지나치게", q))),
                 질문_경험언급=int(d["①취득 경험"] == "확인"), 질문_이해도언급=int(d["①상품이해도"] == "확인"),
                 고객진술성향=int(bool(d["고객 진술 성향"])), 질문길이=len(q) / 100, 답변길이=len(str(d["answer"])) / 100)
        for t in TYPES:
            x["상품_" + t] = int(t in types)
        x["y_범위밖제외"] = int(j.at[cid, "기준③ 처리"] == "제외")
        x["y_등급설명없음"] = int(j.at[cid, "기준④ 판정"] == "기재 없음")
        rows.append(x)
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    df["재간접또는퇴직연금"] = ((df["상품_재간접"] + df["상품_퇴직연금"]) > 0).astype(int)
    df["질문유형"] = np.where(df["유형_조건기반"] == 1, "조건기반", np.where(df["유형_상품비교"] == 1, "상품비교", "고객특성"))
    return df


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("cot_id", "split", "재간접또는퇴직연금", "질문유형") and not c.startswith("y_")]


def fit_models(df: pd.DataFrame):
    """반환: (metrics DataFrame, 로지스틱 계수 DataFrame, {라벨: 나무 모델}, 피처 목록)."""
    F = feature_cols(df)
    tr, te = df[df["split"] == "Training"], df[df["split"] == "Validation"]
    mu, sd = tr[F].mean(), tr[F].std().replace(0, 1)
    mets, coefs, trees = [], [], {}
    for y in LABELS:
        r = {"라벨": y, "설명": LABELS[y], "Training 행": len(tr), "Validation 행": len(te),
             "기준선 Training": round(tr[y].mean(), 3), "기준선 Validation": round(te[y].mean(), 3)}
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit((tr[F] - mu) / sd, tr[y])
        p = lr.predict_proba((te[F] - mu) / sd)[:, 1]
        pr = (p >= 0.5).astype(int)
        r.update({"LR AUC": round(roc_auc_score(te[y], p), 3), "LR 평균 정밀도": round(average_precision_score(te[y], p), 3),
                  "LR 재현율(0.5)": round(recall_score(te[y], pr), 3), "LR 정밀도(0.5)": round(precision_score(te[y], pr), 3)})
        cv = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"), (tr[F] - mu) / sd, tr[y],
                             cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
        r.update({"LR Training 5겹 CV AUC": round(cv.mean(), 3), "CV AUC 표준편차": round(cv.std(), 3)})
        dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", min_samples_leaf=50, random_state=0).fit(tr[F], tr[y])
        q = dt.predict_proba(te[F])[:, 1]
        r.update({"DT AUC": round(roc_auc_score(te[y], q), 3), "DT 평균 정밀도": round(average_precision_score(te[y], q), 3)})
        mets.append(r); trees[y] = dt
        for f, c, imp in zip(F, lr.coef_[0], dt.feature_importances_):
            coefs.append({"라벨": y, "피처": f, "LR 계수(표준화)": round(c, 3), "DT 중요도": round(imp, 3)})
    coef = pd.DataFrame(coefs)
    coef = coef.assign(_a=coef["LR 계수(표준화)"].abs()).sort_values(["라벨", "_a"], ascending=[True, False]).drop(columns="_a")
    return pd.DataFrame(mets), coef.reset_index(drop=True), trees, F


def factor_rates(df: pd.DataFrame) -> pd.DataFrame:
    """요인별 결함 비율 (전체 행 기준, %)."""
    out = []
    for y in LABELS:
        out.append({"라벨": y, "요인": "전체", "값": "", "비율(%)": round(df[y].mean() * 100, 1), "CoT": len(df)})
        for f in ("질문유형", "재간접또는퇴직연금", "상품_단기금융", "질문_높은위험명칭", "질문_숫자등급"):
            for v, g in df.groupby(f):
                out.append({"라벨": y, "요인": f, "값": str(v), "비율(%)": round(g[y].mean() * 100, 1), "CoT": len(g)})
        for (a, b), g in df.groupby(["질문유형", "재간접또는퇴직연금"]):
            out.append({"라벨": y, "요인": "질문유형×재간접또는퇴직연금", "값": f"{a}×{b}", "비율(%)": round(g[y].mean() * 100, 1), "CoT": len(g)})
    return pd.DataFrame(out)
