"""FitCheck 실행 명령 (판정기준서 v1.3, D67).

    python -m src.fitcheck run --as-of 2026-11-24

순서: 판정 → 근거 카드 → 점수표 → 엄격판·교정판 → report_numbers. 모두 기존 src 함수를 부른다.
기록: outputs/runs/<기준일>/manifest.json (기준일, 규칙 버전 = rules.csv 해시, 기준 문서와 시행일, 판정기준서 버전, git 커밋).
- G 최신 등급은 웹을 다시 조회하지 않고 outputs/latest_grade_check.csv(05b 조회 결과)를 읽어 G4만 다시 판정한다.
- 원본 데이터는 data/raw (git 제외). 원문이 들어 있는 엄격판·교정판은 outputs/corrected/ (git 제외)에 쓴다.
애매한 경우 처리: 교정 기록(corrections_log)이 판정의 '교정' 표시에 쓰이므로, 데이터셋을 만든 뒤 판정을 한 번 더 돌려 최종 판정을 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _read(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", **kw)


def _git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def manifest(as_of: str, counts: dict, dirty_before: bool) -> dict:
    rules = ROOT / "src" / "rules"
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    idx = _read(ROOT / "docs" / "regulations" / "regulations_index.csv", dtype=str, keep_default_na=False)
    gold = set(_read(ROOT / "src" / "rag" / "gold_articles.csv", dtype=str, keep_default_na=False)["doc_id"])
    head = (ROOT / "docs" / "판정기준서.md").read_text(encoding="utf-8").splitlines()[0]
    ver = re.search(r"v\d+(?:\.\d+)*", head)
    return {
        "기준일": as_of,
        "실행 시각": datetime.now().isoformat(timespec="seconds"),
        "규칙 버전": {"rules.csv sha256[:16]": h(rules / "rules.csv"), "d43_range.csv sha256[:16]": h(rules / "d43_range.csv"),
                    "transfer_terms.csv sha256[:16]": h(rules / "transfer_terms.csv")},
        "판정기준서 버전": ver.group(0) if ver else head,
        "기준 문서": [{"doc_id": r["doc_id"], "문서명": r["문서명"], "시행일": r["시행일"], "근거 카드 연결": r["doc_id"] in gold}
                    for _, r in idx.iterrows()],
        "git 커밋": _git("rev-parse", "HEAD"),
        "실행 전 작업 트리 변경 있음": dirty_before,   # 실행이 쓰는 outputs는 빼고 실행 전에 본다
        "결과 수": counts,
    }


def run(as_of: str) -> dict:
    from src.dataset import build, dataset_rows, save
    from src.evidence_cards import build_cards
    from src.judge import judge
    from src.latest_grade import g4_rejudge
    from src.load_cot import load_cot, load_products
    from src.report import update_scorecard, write_report_numbers
    from src.scorecard import scorecard_rows

    dirty_before = bool(_git("status", "--porcelain", "--untracked-files=no"))
    cot, prod = load_cot(), load_products()
    # 1. 판정 (+ G4 재판정은 저장된 조회 결과로)
    j, ctx = judge(cot, prod, as_of=as_of)
    lg = _read(OUT / "latest_grade_check.csv", keep_default_na=False, na_values=[""])
    g4 = g4_rejudge(ctx["recs"], ctx["cots"], ctx["range"], lg)
    g4["분할"] = g4["cot_id"].map(ctx["cots"]["split"])
    g4_ids = set(g4.loc[g4["G4 태그"] != "", "cot_id"].astype(int))
    # 엄격판·교정판 (교정 기록이 판정에 쓰이므로 먼저 만들고 판정을 다시 돌린다)
    d4 = _read(OUT / "d4_cause_by_product.csv", keep_default_na=False)
    strict, corr, newlog = build(cot, j, d4, lg, g4_ids, ctx["recs"])
    save(strict, corr, newlog)
    j, ctx = judge(cot, prod, as_of=as_of)
    j.to_csv(OUT / "judgments.csv", index=False, encoding="utf-8-sig")
    g4.to_csv(OUT / "latest_grade_rejudge.csv", index=False, encoding="utf-8-sig")
    # 2. 근거 카드
    cards = build_cards(j, ctx, lg, g4)
    cards.to_csv(OUT / "evidence_cards.csv", index=False, encoding="utf-8-sig")
    # 3. 점수표
    update_scorecard(scorecard_rows(j, ctx, g4, _read(OUT / "latest_grade_check.csv", keep_default_na=False), as_of,
                                    src="fitcheck run"), "품질 점수표 (판정기준서 v1.3)")
    # 4. 데이터셋 구역
    update_scorecard(dataset_rows(strict, corr, newlog, d4, as_of, src="fitcheck run"), "정제 데이터셋 (08)")
    # 5. report_numbers
    write_report_numbers(_read(OUT / "scorecard.csv", dtype=str, keep_default_na=False))
    counts = {"증권 CoT": len(j), "제외 CoT": int((j["기준③ 처리"] == "제외").sum()),
              "이관 결함 CoT": int((j["이관 판정"] == "결함").sum()), "이관 사람 확인 CoT": int((j["이관 판정"] == "사람 확인").sum()),
              "근거 카드 행": len(cards), "근거 카드 CoT": int(cards["cot_id"].nunique()),
              "엄격판 CoT": len(strict), "교정판 CoT": len(corr), "G4 최신 기준 부적합 추천": int((g4["G4 태그"] != "").sum())}
    m = manifest(as_of, counts, dirty_before)
    rd = OUT / "runs" / as_of; rd.mkdir(parents=True, exist_ok=True)
    (rd / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.fitcheck", description="FitCheck 판정·근거 카드·점수표·데이터셋 실행")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="기준일로 전체 실행")
    r.add_argument("--as-of", required=True, help="기준일 YYYY-MM-DD (규칙·범위표·문서 시행일 판단)")
    lv = sub.add_parser("live", help="시연: 문서 검색 → 규칙 생성 → 상품 등급 검색 → 근거 검색 → 적용 (src/live.py)")
    lv.add_argument("--n", type=int, default=30, help="분석 건수 (최신 기준 부적합 min(n/3, 16), 범위 밖 최소 3 포함)")
    lv.add_argument("--as-of", default="2026-11-24", help="기준일 YYYY-MM-DD")
    lv.add_argument("--saved", action="store_true", help="웹 조회 없이 저장값으로")
    lv.add_argument("--ids", default="", help="cot_id 목록 (쉼표). 주면 --n 대신")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        datetime.strptime(a.as_of, "%Y-%m-%d")
        m = run(a.as_of)
        print(json.dumps(m["결과 수"], ensure_ascii=False))
        print("manifest:", OUT / "runs" / a.as_of / "manifest.json")
    elif a.cmd == "live":
        live_cli(a)


def live_cli(a):
    """진행 메시지·표를 터미널에 차례로 찍는다. 표는 앞 몇 행만 (원문 없음)."""
    import time
    from src.live import load_base, pick_sample, run_live, save_run
    datetime.strptime(a.as_of, "%Y-%m-%d")
    t0 = time.time()
    print(f"[준비] 데이터 판정 (기준일 {a.as_of}) ...", flush=True)
    base = load_base(a.as_of)
    ids = [int(x) for x in a.ids.split(",") if x.strip()] or pick_sample(base, a.n)
    print(f"[준비] {round(time.time() - t0, 1)}초, cot_id {ids}", flush=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
    last = None
    for ev in run_live(base, ids, live=not a.saved):
        k = ev["kind"]
        if k == "progress":
            if ev["i"] == ev["n"] or ev["i"] % 10 == 0:
                print(f"  [{ev['step']}] 조회 {ev['i']}/{ev['n']}", flush=True)
        elif k in ("msg", "warn"):
            print(f"[{ev['step']}] {'주의: ' if k == 'warn' else ''}{ev['text']}", flush=True)
        elif k == "table":
            print(f"[{ev['step']}] {ev['title']} ({len(ev['data'])}행)\n{ev['data'].head(8).to_string(index=False)}", flush=True)
        elif k == "done":
            last = ev
    print("\n[결과 요약]\n" + last["summary"].to_string(index=False))
    d = save_run(last, a.as_of, len(ids), not a.saved)
    print(f"\n파이프라인 {last['seconds']}초, 전체 {round(time.time() - t0, 1)}초. 저장: {d}")


if __name__ == "__main__":
    main()
