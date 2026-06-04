#!/usr/bin/env python3
"""Generate deterministic concept figures for the thesis.

These figures are text-sensitive thesis diagrams, so they are generated with
matplotlib rather than an image model. GPT-generated images can be used as
style references, but all final labels and numbers are fixed here.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def setup_font() -> None:
    candidates = [
        Path.home() / "Library/Fonts/KoPubWorld-Batang-Medium.otf",
        Path("/Library/Fonts/KoPubWorld-Batang-Medium.otf"),
        Path.home() / "Library/Fonts/KoPubWorld-Dotum-Medium.otf",
        Path("/Library/Fonts/KoPubWorld-Dotum-Medium.otf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in [
        "KoPubWorldDotum_Pro",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "KoPubWorldBatang_Pro",
        "AppleMyungjo",
    ]:
        if name in installed:
            plt.rcParams["font.family"] = name
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.size"] = 10


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGS / filename, bbox_inches="tight", pad_inches=0.12)
    if filename == "fig1_research_flow.png":
        fig.savefig(FIGS / "fig1_research_flow.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"saved figures/{filename} ({os.path.getsize(FIGS / filename) // 1024}KB)")


def add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    fill: str = "#F8FAFC",
    edge: str = "#334155",
    title_color: str = "#111827",
    body_color: str = "#374151",
    lw: float = 1.5,
    title_size: float = 11,
    body_size: float = 8.3,
    radius: float = 0.12,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    title_y = y + h * (0.5 if not body else 0.72)
    ax.text(
        x + w / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        linespacing=1.25,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.34,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=body_color,
            linespacing=1.45,
        )


def add_centered_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[tuple[str, float, str, str]],
    fill: str = "#F8FAFC",
    edge: str = "#64748B",
    lw: float = 1.0,
    radius: float = 0.08,
    gap: float | None = None,
) -> None:
    """Draw a thesis-style text box with explicit Korean line centering."""
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)

    n = len(lines)
    if gap is None:
        gap = h * (0.27 if n >= 3 else 0.24)
    y0 = y + h / 2 + gap * (n - 1) / 2
    for idx, (text, size, weight, color) in enumerate(lines):
        ax.text(
            x + w / 2,
            y0 - gap * idx,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
        )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.6, shrinkA=3, shrinkB=3),
    )


def make_fig1() -> None:
    fig, ax = plt.subplots(figsize=(5.55, 6.35))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 12.85)
    ax.axis("off")

    edge = "#4B5563"
    light = "#F3F4F6"
    lighter = "#F8FAFC"
    text = "#111827"

    def square_box(x, y, w, h, fill="white", lw=0.9):
        box = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge, facecolor=fill)
        ax.add_patch(box)
        return box

    def put(x, y, s, size=9.0, weight="normal", ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=text, ha=ha, va=va, linespacing=1.38)

    left_x = 0.35
    left_w = 1.65
    body_x = 2.18
    body_w = 7.25

    def chapter_row(y, h, chapter, title, lines, font_size=8.6, two_cols=False, title_size=8.2):
        square_box(left_x, y, left_w, h, fill=light)
        square_box(body_x, y, body_w, h, fill="white")
        put(left_x + left_w / 2, y + h / 2 + 0.18, chapter, size=10.2, weight="bold", ha="center")
        put(left_x + left_w / 2, y + h / 2 - 0.27, title, size=title_size, weight="bold", ha="center")
        if two_cols:
            mid = body_x + body_w / 2
            square_box(mid, y + 0.22, 0.001, h - 0.44, fill=edge, lw=0.25)
            left_lines, right_lines = lines
            start_y = y + h - 0.56
            gap = (h - 0.92) / max(len(left_lines), len(right_lines), 1)
            for idx, line in enumerate(left_lines):
                put(body_x + 0.25, start_y - gap * idx, line, size=font_size)
            for idx, line in enumerate(right_lines):
                put(mid + 0.25, start_y - gap * idx, line, size=font_size)
        else:
            start_y = y + h - 0.40
            gap = min(0.43, (h - 0.70) / max(len(lines) - 1, 1))
            for idx, line in enumerate(lines):
                put(body_x + 0.28, start_y - gap * idx, line, size=font_size)

    chapter_row(
        11.20,
        1.45,
        "제1장",
        "서론",
        ["제1절 연구의 배경 및 목적", "제2절 연구의 범위", "제3절 연구 방법 및 과정"],
        font_size=8.7,
    )
    chapter_row(
        9.20,
        1.75,
        "제2장",
        "이론적 배경\n및 선행연구",
        [
            "제1절 헤도닉 가격모형",
            "제2절 머신러닝 기반 부동산 가격 예측",
            "제3절 설명가능한 인공지능(XAI)",
            "제4절 선행연구 검토 및 차별성",
        ],
        font_size=7.8,
        title_size=7.3,
    )
    chapter_row(
        7.62,
        1.43,
        "제3장",
        "분석의 틀",
        ["제1절 연구 방법", "제2절 변수의 정의 및 구축", "제3절 연구 모형"],
        font_size=8.7,
    )
    chapter_row(
        2.45,
        4.90,
        "제4장",
        "실증 분석",
        (
            [
                "제1절 기술통계 및 변수 분포",
                "제2절 모형별 예측 성능 비교",
                "제3절 소거분석(Ablation)",
                "제4절 전체 SHAP 분석",
            ],
            [
                "제5절 권역별 SHAP 분석",
                "제6절 연도별 SHAP 분석",
                "제7절 연도×권역 교차 SHAP 분석",
            ],
        ),
        font_size=7.7,
        two_cols=True,
    )
    chapter_row(
        0.65,
        1.48,
        "제5장",
        "결론",
        ["제1절 연구 결과 요약", "제2절 연구의 한계 및 향후 과제"],
        font_size=8.9,
    )

    square_box(0.24, 0.36, 9.38, 12.34, fill="none", lw=0.75)
    save(fig, "fig1_research_flow.png")


def make_fig2() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 2.72))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.35)
    ax.axis("off")

    add_centered_box(
        ax,
        0.35,
        1.72,
        2.25,
        1.28,
        [
            ("입력 변수 X", 8.9, "bold", "#111827"),
            ("주택 특성 · 입지 접근성", 6.8, "normal", "#374151"),
            ("권역 · 거시경제", 6.8, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.1,
        radius=0.08,
    )

    add_centered_box(
        ax,
        3.25,
        1.72,
        3.45,
        1.28,
        [
            ("순차적 트리 앙상블", 8.9, "bold", "#111827"),
            ("Tree 1 → Tree 2 → ··· → Tree K", 6.8, "normal", "#374151"),
            ("이전 예측 오차를 반복 보정", 6.8, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.1,
        radius=0.08,
    )
    arrow(ax, (2.6, 2.36), (3.25, 2.36))

    add_centered_box(
        ax,
        7.35,
        1.72,
        2.3,
        1.28,
        [
            ("예측값", 8.9, "bold", "#111827"),
            ("y_hat = Σ η f_k(x)", 6.9, "normal", "#374151"),
            ("log(㎡당 가격)", 6.9, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.1,
        radius=0.08,
    )
    arrow(ax, (6.7, 2.36), (7.35, 2.36))

    add_centered_box(
        ax,
        2.05,
        3.26,
        5.9,
        0.66,
        [
            ("목적함수", 7.6, "bold", "#111827"),
            ("손실함수 + 정규화 항: 예측 오차와 모형 복잡도를 함께 제어", 5.9, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#475569",
        lw=0.9,
        radius=0.06,
        gap=0.22,
    )

    features = [
        ("정규화", "과적합 억제", 0.65),
        ("축소", "학습률 0.1", 2.95),
        ("서브샘플링", "행·열 일부 사용", 5.25),
        ("검증 설계", "무작위 · 시간순 · 단지", 7.55),
    ]
    for title, body, x in features:
        add_centered_box(
            ax,
            x,
            0.38,
            1.8,
            0.76,
            [
                (title, 6.7, "bold", "#111827"),
                (body, 5.8, "normal", "#475569"),
            ],
            fill="#F1F5F9",
            edge="#94A3B8",
            lw=0.8,
            radius=0.06,
            gap=0.20,
        )
    save(fig, "fig2_xgboost_concept.png")


def make_fig3() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.35)
    ax.axis("off")

    add_centered_box(
        ax,
        0.35,
        4.9,
        2.55,
        0.96,
        [
            ("예측 모형", 8.7, "bold", "#111827"),
            ("XGBoost 통합 모형", 6.6, "normal", "#374151"),
            ("테스트 표본", 6.6, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.0,
        radius=0.08,
    )
    add_centered_box(
        ax,
        3.72,
        4.9,
        2.55,
        0.96,
        [
            ("TreeSHAP", 8.7, "bold", "#111827"),
            ("예측값을 변수별", 6.6, "normal", "#374151"),
            ("기여도 phi_j로 분해", 6.6, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.0,
        radius=0.08,
    )
    add_centered_box(
        ax,
        7.1,
        4.9,
        2.55,
        0.96,
        [
            ("해석 자료", 8.7, "bold", "#111827"),
            ("SHAP값 행렬", 6.6, "normal", "#374151"),
            ("표본 × 변수", 6.6, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.0,
        radius=0.08,
    )
    arrow(ax, (2.9, 5.38), (3.72, 5.38))
    arrow(ax, (6.27, 5.38), (7.1, 5.38))

    ax.text(
        5.0,
        4.27,
        "Shapley value 원리에 따라 각 변수의 평균적 예측 기여를 집계",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#475569",
    )
    arrow(ax, (5.0, 4.9), (5.0, 4.5), "#64748B")

    output_boxes = [
        (0.35, "전역 중요도", "평균 |SHAP|\n상위 변수 식별"),
        (3.72, "방향성·의존성", "변수값 변화에 따른\n기여 방향 확인"),
        (7.1, "권역·연도 분해", "강남3구·비강남\n연도별 구조 비교"),
    ]
    for x, title, body in output_boxes:
        add_centered_box(
            ax,
            x,
            2.72,
            2.55,
            0.96,
            [(title, 7.7, "bold", "#111827")]
            + [(line, 6.3, "normal", "#374151") for line in body.split("\n")],
            fill="#F1F5F9",
            edge="#94A3B8",
            lw=0.9,
            radius=0.08,
        )
        arrow(ax, (5.0, 4.1), (x + 1.28, 3.68), "#64748B")

    add_centered_box(
        ax,
        2.05,
        0.95,
        5.9,
        0.82,
        [
            ("논문 내 산출물", 7.9, "bold", "#111827"),
            ("SHAP Bar · Summary · Dependence · 권역×연도 비교", 6.3, "normal", "#374151"),
        ],
        fill="#F8FAFC",
        edge="#64748B",
        lw=1.0,
        radius=0.08,
        gap=0.22,
    )
    for x in (1.62, 5.0, 8.38):
        arrow(ax, (x, 2.72), (5, 1.77), "#64748B")

    save(fig, "fig3_shap_framework.png")


def main() -> None:
    setup_font()
    make_fig1()
    make_fig2()
    make_fig3()


if __name__ == "__main__":
    main()
