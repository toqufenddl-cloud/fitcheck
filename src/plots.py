"""그래프 공통 설정. 집계값만 받아 그린다 (원문·개별 행은 넣지 않는다)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOURCE = "데이터 출처: AI Hub 「금융상품·서비스 및 소비자 특성 데이터」(한국지능정보사회진흥원), 본 프로젝트 집계"
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def setup():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save(fig, name: str, title: str, source: str = SOURCE) -> Path:
    """제목과 출처 한 줄을 붙여 outputs/figures/<name>.png로 저장한다."""
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.01, source, fontsize=8, color="#666666", ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def bar_labels(ax, bars, fmt="{:,.0f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")
