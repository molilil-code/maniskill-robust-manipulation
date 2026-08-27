# -*- coding: utf-8 -*-
"""生成项目全部结果图(发布级)。

产物:
  results/figures/ood_success_comparison.png  图1 主结果:分组柱状图(6 条件 × 2 策略)
  results/figures/ood_improvement.png         图2 OOD 提升量 ΔSR(含负值 Goal Near)
  results/figures/seed_stability.png          图3 训练 seed 稳定性 dot plot
  results/figures/failure_analysis.png        图4 失败分析三栏图(a/b/c)

配色采用已验证的 dataviz 默认 palette(light mode):
  蓝 #2a78d6 = Baseline / 正增量
  橙 #eb6834 = + Episode DR
  红 #e34948 = 负增量
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 调色板与排版(light mode)
# ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axes(ax):
    """统一坐标轴风格:去上右边框、横向网格、无刻度短线。"""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved:", path)
    plt.close(fig)


# ============================================================
# 图1 主结果:Grouped Bar Chart
# ============================================================
def fig1_main_bar():
    conditions = [
        "Normal", "Cube OOD", "Goal Far",
        "Qpos Shift", "Episode Combined", "Full Combined",
    ]
    baseline_mean = np.array([61.7, 29.7, 8.0, 64.3, 9.7, 15.3])
    baseline_std = np.array([20.2, 7.2, 7.2, 12.6, 5.5, 9.1])
    dr_mean = np.array([70.3, 35.7, 20.3, 77.0, 16.0, 19.7])
    dr_std = np.array([3.1, 4.0, 12.9, 3.6, 2.6, 1.2])

    x = np.arange(len(conditions))
    width = 0.36
    gap = 0.02          # 组内两柱之间的 surface gap
    off = width / 2 + gap / 2

    fig, ax = plt.subplots(figsize=(10, 5.2))
    style_axes(ax)

    ax.bar(x - off, baseline_mean, width, color=BLUE,
           label="Depth+Goal", zorder=3)
    ax.bar(x + off, dr_mean, width, color=ORANGE,
           label="Depth+Goal + Episode DR", zorder=3)

    ax.errorbar(x - off, baseline_mean, yerr=baseline_std,
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
    ax.errorbar(x + off, dr_mean, yerr=dr_std,
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)

    for xi, m in zip(x - off, baseline_mean):
        ax.text(xi, m + 1.2, f"{m:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK2)
    for xi, m in zip(x + off, dr_mean):
        ax.text(xi, m + 1.2, f"{m:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK2)

    # 高亮 Normal 组的 seed 方差下降
    ax.annotate(
        "Normal: seed std 20.2 → 3.1",
        xy=(-off, baseline_mean[0] + baseline_std[0]),
        xytext=(0.30, 97.5),
        ha="center", fontsize=9.5, color=INK2,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
    )

    ax.set_xticks(x, conditions, fontsize=10.5)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=10, handlelength=1.0)

    save(fig, "ood_success_comparison.png")


# ============================================================
# 图2 OOD Improvement:ΔSR = SR_DR - SR_Baseline
# ============================================================
def fig2_delta():
    conditions = [
        "Cube OOD", "Goal Far", "Qpos Shift",
        "Episode Combined", "Full Combined", "Goal Near",
    ]
    deltas = [6.0, 12.3, 12.7, 6.3, 4.3, -2.7]
    colors = [BLUE if d >= 0 else RED for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    style_axes(ax)

    # 正/负区域背景与零线
    ax.axhspan(0, 16, color="#0ca30c", alpha=0.045, zorder=0)
    ax.axhspan(-6, 0, color="#d03b3b", alpha=0.045, zorder=0)
    ax.axhline(0, color=INK2, linewidth=1.0, zorder=2)

    x = np.arange(len(conditions))
    ax.bar(x, deltas, 0.55, color=colors, zorder=3)

    for xi, d in zip(x, deltas):
        sign = "+" if d >= 0 else "−"
        ax.text(
            xi, d + (0.5 if d >= 0 else -0.5),
            f"{sign}{abs(d):.1f}",
            ha="center", va="bottom" if d >= 0 else "top",
            fontsize=10, color=INK2,
        )

    ax.set_xticks(x, conditions, fontsize=10.5)
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylim(-6, 16)
    ax.set_ylabel("Δ Success Rate (pp)", fontsize=11)
    ax.text(0.99, 0.03,
            "ΔSR = SR$_{DR}$ − SR$_{Baseline}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=MUTED)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(fc=BLUE, label="improvement"),
            Patch(fc=RED, label="degradation"),
        ],
        frameon=False, loc="upper right", fontsize=9.5,
    )

    save(fig, "ood_improvement.png")


# ============================================================
# 图3 训练 seed 稳定性(Normal 条件)
# ============================================================
def fig3_seeds():
    baseline_seeds = [50, 50, 85]
    dr_seeds = [67, 73, 71]
    means = [61.7, 70.3]
    stds = [20.2, 3.1]
    colors = [BLUE, ORANGE]
    groups = ["Depth+Goal", "Depth+Goal\n+ Episode DR"]

    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax)

    xpos = [0.0, 1.5]
    jitter = [-0.1, 0.0, 0.1]

    for xc, seeds, mean, std, c in zip(
        xpos, (baseline_seeds, dr_seeds), means, stds, colors
    ):
        # mean ± seed std 阴影带
        ax.fill_between(
            [xc - 0.26, xc + 0.26], mean - std, mean + std,
            color=c, alpha=0.14, lw=0, zorder=1,
        )
        # 均值线
        ax.plot([xc - 0.26, xc + 0.26], [mean, mean],
                color=c, lw=2.6, zorder=3)
        # 三个 seed 的点
        for j, v in zip(jitter, seeds):
            ax.scatter(xc + j, v, s=130, color=c,
                       edgecolor="white", lw=1.4, zorder=4)
            ax.text(xc + j, v + 4.5, str(v), ha="center",
                    fontsize=9, color=INK2)

    # 均值标注(左对齐 / 右对齐,避免出界)
    ax.text(0.30, means[0] + 3, f"mean {means[0]:.1f} ± {stds[0]:.1f}",
            va="center", fontsize=9.5, color=INK2)
    ax.text(1.20, means[1] + 3, f"mean {means[1]:.1f} ± {stds[1]:.1f}",
            va="center", ha="right", fontsize=9.5, color=INK2)

    ax.annotate(
        "seed std 20.2 → 3.1 (−85%)",
        xy=(1.5, means[1] + stds[1]),
        xytext=(0.72, 99),
        ha="center", fontsize=9.5, color=INK2,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
    )

    from matplotlib.lines import Line2D
    ax.set_xticks(xpos, groups, fontsize=10.5)
    ax.set_xlim(-0.55, 2.05)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Normal condition · 3 training seeds",
                 loc="left", fontsize=11.5, pad=10)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=BLUE,
                   markersize=8, label="Depth+Goal"),
            Line2D([], [], marker="o", ls="", color=ORANGE,
                   markersize=8, label="Depth+Goal + Episode DR"),
        ],
        frameon=False, loc="upper left", fontsize=9.5,
    )

    save(fig, "seed_stability.png")


# ============================================================
# 图4 Failure Analysis 三栏图
#   (a) RGB / Depth 观测对比
#   (b) Goal Far 失败 rollout 帧
#   (c) TCP-Cube / Cube-Goal 距离曲线
# ============================================================
def fig4_failure():
    import imageio.v2 as imageio

    # --------------------------------------------------------
    # 显式 rect 布局(单位:英寸,左下角为原点)。
    # (a) 居中且收窄;(b)/(c) 之间留出明确间隔;
    # 每个面板标题占据独立的一行 cell,与内容互不重叠。
    # --------------------------------------------------------
    fig_w = 11.5
    img_w, img_h = 2400, 1000          # observation_rgb_depth.png 原始尺寸
    img_area_w = 0.72 * fig_w          # (a) 图片区宽度(居中)
    img_area_h = img_area_w * img_h / img_w   # 按 2.4:1 等比计算高度
    x_a = (fig_w - img_area_w) / 2     # (a) 居中偏移

    cap_h = 0.44      # 标题行高度
    gap_small = 0.18  # 标题与内容间距
    gap_rows = 0.45   # (a) 与 (b)/(c) 之间的间距
    margin = 0.15
    gap_cols = 0.55   # (b) 与 (c) 之间的间隔

    content_h = 2.5   # 底部内容行高度(帧 + 曲线)
    col_w = (fig_w - 2 * margin - gap_cols) / 2   # 每列宽度
    frame_w = (col_w - 0.24) / 4                  # 4 帧 + 3 个间隙

    fig_h = (margin + cap_h + gap_small + img_area_h
             + gap_rows + cap_h + gap_small + content_h + margin)
    fig = plt.figure(figsize=(fig_w, fig_h))

    def rect(x, y, w, h):
        """英寸 -> 图形分数坐标(x, y 为左下角)。"""
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    def caption(x, w, y, letter, text):
        ax = fig.add_axes(rect(x, y, w, cap_h))
        ax.axis("off")
        ax.text(0, 0.5, r"$\bf{(" + letter + r")}$ " + text,
                fontsize=10.5, color=INK, va="center", ha="left",
                wrap=True)
        return ax

    x_left = margin
    x_right = x_left + col_w + gap_cols          # 右列起点

    y_content = margin                                    # 底部内容行
    y_cap_row = y_content + content_h + gap_small         # (b)/(c) 标题行
    y_img = y_cap_row + cap_h + gap_rows                  # (a) 图片
    y_cap_a = y_img + img_area_h + gap_small              # (a) 标题行

    # ---- (a) 观测对比(居中、收窄) ----
    caption(x_a, img_area_w, y_cap_a, "a",
            "Observation design — goal region visible in RGB, "
            "nearly invisible in depth → Depth+Goal input")
    ax_a = fig.add_axes(rect(x_a, y_img, img_area_w, img_area_h))
    ax_a.imshow(plt.imread(OUT_DIR / "observation_rgb_depth.png"))
    ax_a.axis("off")

    # ---- (b) Goal Far 失败 rollout 帧 ----
    caption(x_left, col_w, y_cap_row, "b",
            "Goal Far failure rollout — baseline Depth+Goal policy")

    reader = imageio.get_reader("videos/depth_goal/goal_far/9.mp4")
    ts = [1, 17, 33, 49]
    frames = [reader.get_data(i) for i in ts]
    reader.close()

    frame_gap = 0.08
    for k, (t, fr) in enumerate(zip(ts, frames)):
        ax_b = fig.add_axes(rect(
            x_left + k * (frame_w + frame_gap),
            y_content + content_h - frame_w - 0.24,   # 顶部对齐,下方留给 t 标注
            frame_w, frame_w + 0.24,
        ))
        ax_b.imshow(fr)
        ax_b.axis("off")
        ax_b.set_title(f"t = {t}", fontsize=8.5, color=MUTED, pad=2)

    # ---- (c) 距离曲线(10 个失败 episode 的均值 ± std) ----
    caption(x_right, col_w, y_cap_row, "c",
            "Distance diagnostics on Goal Far (baseline policy)")

    ax_c = fig.add_axes(rect(x_right, y_content, col_w, content_h))
    style_axes(ax_c)

    df = pd.read_csv("videos/depth_goal/goal_far/distances.csv")
    g = df.groupby("step")[["d_tcp_cube", "d_cube_goal"]]
    mu, sd = g.mean(), g.std()
    steps = mu.index.to_numpy()

    ax_c.plot(steps, mu["d_tcp_cube"], color=BLUE, lw=2.2,
              label="TCP–cube distance", zorder=3)
    ax_c.fill_between(
        steps,
        mu["d_tcp_cube"] - sd["d_tcp_cube"],
        mu["d_tcp_cube"] + sd["d_tcp_cube"],
        color=BLUE, alpha=0.13, lw=0, zorder=1,
    )
    ax_c.plot(steps, mu["d_cube_goal"], color=ORANGE, lw=2.2,
              label="cube–goal distance", zorder=3)
    ax_c.fill_between(
        steps,
        mu["d_cube_goal"] - sd["d_cube_goal"],
        mu["d_cube_goal"] + sd["d_cube_goal"],
        color=ORANGE, alpha=0.13, lw=0, zorder=1,
    )

    # 成功阈值(goal radius = 0.1 m)
    ax_c.axhline(0.1, color=AXIS, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax_c.text(50, 0.106, "success threshold (0.1 m)",
              ha="right", va="bottom", fontsize=8.5, color=MUTED)

    # 结尾注释
    ax_c.annotate(
        "TCP reaches the cube",
        xy=(50, mu["d_tcp_cube"].iloc[-1]),
        xytext=(12, 0.04),
        fontsize=9, color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9),
    )
    ax_c.annotate(
        "cube stops 0.25 m from goal",
        xy=(50, mu["d_cube_goal"].iloc[-1]),
        xytext=(33, 0.345),
        ha="center",
        fontsize=9, color=ORANGE,
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9),
    )

    ax_c.set_xlim(1, 50)
    ax_c.set_ylim(0, 0.40)
    ax_c.set_xlabel("Timestep", fontsize=10.5)
    ax_c.set_ylabel("Distance (m)", fontsize=10.5)
    ax_c.legend(frameon=False, loc="upper right", fontsize=9)
    ax_c.text(0.02, 0.02, "mean ± std over 10 failed episodes",
              transform=ax_c.transAxes, fontsize=8.5, color=MUTED,
              va="bottom")

    save(fig, "failure_analysis.png")


# ============================================================
if __name__ == "__main__":
    fig1_main_bar()
    fig2_delta()
    fig3_seeds()
    fig4_failure()
    print("all figures done")
