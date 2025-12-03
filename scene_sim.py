"""
scene_sim.py
单独针对“题目要求场景”的动态仿真与可视化：
- 只展示真实运动轨迹（不包含跟踪与评估指标）
- 标注三部雷达、交汇点 F、干扰区、杂波区域
- 在图上清晰标出全局直角坐标系（按 config 中的设置）
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import numpy as np

from config import (
    KM,
    RADARS,
    F_POS,
    JAM_REGION_CENTER,
    JAM_REGION_RADIUS,
    CLUTTER_REGION_X,
    CLUTTER_REGION_Y,
    TOTAL_TIME,
    N_STEPS,
    DT,
)
from data_gen import generate_target_trajectories, in_jam_region

# 和 dynamic_sim 一致，优先使用 TkAgg，防止某些环境下阻塞
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

# 使用本地中文字体（SimHei.ttf），确保中文正常显示
try:
    font_manager.fontManager.addfont("fonts/SimHei.ttf")
    rcParams["font.sans-serif"] = ["SimHei"]
    rcParams["axes.unicode_minus"] = False
except Exception:
    # 字体加载失败时继续运行，只是中文可能显示为方块
    pass


def main(save_gif: bool = True, gif_name: str = None):
    """
    场景还原仿真入口：
    只演示 A/B 的真值轨迹与场景元素，用于说明：
    - 雷达布局
    - 坐标系定义
    - 目标初始位置、交汇点和机动形状
    """
    print("[scene_sim] 生成真值轨迹...")
    times, trajA, trajB = generate_target_trajectories()

    # 以 km 为单位进行可视化
    trajA_km = trajA[:, :2] / KM
    trajB_km = trajB[:, :2] / KM

    x_min_km = CLUTTER_REGION_X[0] / KM
    x_max_km = CLUTTER_REGION_X[1] / KM
    y_min_km = CLUTTER_REGION_Y[0] / KM
    y_max_km = CLUTTER_REGION_Y[1] / KM

    # 画布与坐标轴
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")

    # 按配置覆盖整个杂波区域（与 config 中 CLUTTER_REGION_X/Y 对齐）
    ax.set_xlim(x_min_km, x_max_km)
    ax.set_ylim(y_min_km, y_max_km)

    # ---- 坐标系标注（全局直角坐标系，与 config 一致）----
    ax.set_xlabel("X (km)", fontsize=14)
    ax.set_ylabel("Y (km)", fontsize=14)
    ax.set_title("Scenario Simulation with Global Cartesian Coordinates", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 原点与坐标轴方向箭头（坐标轴覆盖整个可视区域）
    origin = np.array([0.0, 0.0])

    # X 轴：从左到右贯穿整个图，并在右端加箭头
    ax.annotate(
        "",
        xy=(x_max_km, 0.0),
        xytext=(x_min_km, 0.0),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
    )
    ax.text(
        x_max_km,
        0.0 + 1.0,
        "X 轴",
        ha="right",
        va="bottom",
        fontsize=14,
    )

    # Y 轴：从下到上贯穿整个图，并在顶端加箭头
    ax.annotate(
        "",
        xy=(0.0, y_max_km),
        xytext=(0.0, y_min_km),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
    )
    ax.text(
        0.0 + 1.0,
        y_max_km,
        "Y 轴",
        ha="left",
        va="top",
        fontsize=14,
    )

    # 原点标注
    ax.plot(origin[0], origin[1], "ko", markersize=4)
    ax.text(
        origin[0] + 2.0,
        origin[1] + 2.0,
        "O (0, 0)",
        fontsize=14,
    )

    # ---- 场景静态元素：雷达、交汇点 F、干扰区、杂波区域 ----

    # 雷达位置与覆盖圈
    for rid, pos in RADARS.items():
        pos_km = pos / KM
        ax.plot(pos_km[0], pos_km[1], "r^", markersize=8)
        ax.text(
            pos_km[0],
            pos_km[1] - 3.0,
            f"{rid}\n({pos_km[0]:.1f}, {pos_km[1]:.1f}) km",
            color="red",
            fontsize=14,
            ha="center",
        )

        # 示意性的 60km 探测圈（与 dynamic_sim 一致）
        circle = Circle(
            pos_km,
            60.0,
            edgecolor="red",
            facecolor="none",
            linestyle=":",
            linewidth=1.5,
            alpha=0.5,
        )
        ax.add_patch(circle)

    # 用虚线连接 R1 和 R2，并标注长度 50km
    if "R1" in RADARS and "R2" in RADARS:
        r1_km = RADARS["R1"] / KM
        r2_km = RADARS["R2"] / KM
        ax.plot(
            [r1_km[0], r2_km[0]],
            [r1_km[1], r2_km[1]],
            linestyle="--",
            color="black",
            linewidth=2.0,
        )
        mid = 0.5 * (r1_km + r2_km)
        ax.text(
            mid[0],
            mid[1] + 2.0,
            "50 km",
            color="black",
            fontsize=14,
            ha="center",
            va="bottom",
        )

    # 交汇点 F
    F_km = F_POS / KM
    ax.plot(F_km[0], F_km[1], "bo", markersize=6)
    ax.text(
        F_km[0] + 2.0,
        F_km[1] + 2.0,
        "F 点",
        color="blue",
        fontsize=14,
    )

    # 干扰区域（按照 config：中心 JAM_REGION_CENTER，半边长 JAM_REGION_RADIUS 的正方形）
    jam_left = JAM_REGION_CENTER[0] - JAM_REGION_RADIUS
    jam_bottom = JAM_REGION_CENTER[1] - JAM_REGION_RADIUS
    jam_size = 2 * JAM_REGION_RADIUS

    jam_left_km = jam_left / KM
    jam_bottom_km = jam_bottom / KM
    jam_size_km = jam_size / KM

    jam_rect = Rectangle(
        (jam_left_km, jam_bottom_km),
        jam_size_km,
        jam_size_km,
        edgecolor="orange",
        facecolor="orange",
        alpha=0.25,
        linewidth=2.0,
    )
    ax.add_patch(jam_rect)
    ax.text(
        JAM_REGION_CENTER[0] / KM,
        JAM_REGION_CENTER[1] / KM,
        "干扰区\n(Jamming)",
        color="black",
        fontsize=14,
        ha="center",
        va="center",
    )

    # 杂波生成区域（矩形边框）
    clutter_rect = Rectangle(
        (CLUTTER_REGION_X[0] / KM, CLUTTER_REGION_Y[0] / KM),
        (CLUTTER_REGION_X[1] - CLUTTER_REGION_X[0]) / KM,
        (CLUTTER_REGION_Y[1] - CLUTTER_REGION_Y[0]) / KM,
        edgecolor="gray",
        facecolor="none",
        linestyle="--",
        linewidth=1.0,
    )
    ax.add_patch(clutter_rect)
    ax.text(
        CLUTTER_REGION_X[0] / KM + 5.0,
        CLUTTER_REGION_Y[1] / KM - 5.0,
        "杂波区域",
        color="gray",
        fontsize=14,
    )

    # A/B 起点标注（A_start / B_start）
    # A 起点在 trajA[0]
    if not np.isnan(trajA[0, 0]):
        A_start_km = trajA_km[0]
        ax.plot(A_start_km[0], A_start_km[1], marker="*", color="c", markersize=12)
        ax.text(
            A_start_km[0] + 2.0,
            A_start_km[1] + 2.0,
            f"A_start\n({A_start_km[0]:.1f}, {A_start_km[1]:.1f}) km",
            color="c",
            fontsize=14,
        )

    # B 起点为 trajB 第一帧非 NaN 位置
    valid_B_idx = np.where(~np.isnan(trajB[:, 0]))[0]
    if len(valid_B_idx) > 0:
        b0 = valid_B_idx[0]
        B_start_km = trajB_km[b0]
        ax.plot(B_start_km[0], B_start_km[1], marker="*", color="orange", markersize=12)
        ax.text(
            B_start_km[0] + 2.0,
            B_start_km[1] + 2.0,
            f"B_start\n({B_start_km[0]:.1f}, {B_start_km[1]:.1f}) km",
            color="orange",
            fontsize=14,
        )

    # ---- 动态元素：A/B 真值轨迹 + 当前点 ----
    line_A, = ax.plot([], [], "c--", linewidth=2.0, label="目标 A 轨迹")
    line_B, = ax.plot([], [], color="orange", linestyle="--", linewidth=2.5, label="目标 B 轨迹")

    point_A, = ax.plot([], [], "co", markersize=10, label="A 当前")
    point_B, = ax.plot([], [], marker="o", color="orange", markersize=10, label="B 当前")

    # 额外标注：当 A 在干扰区时，高亮提示
    jam_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="red",
        fontsize=16,
        fontweight="bold",
    )

    ax.legend(loc="upper right", fontsize=12)

    def update(frame: int):
        t = times[frame]

        # 当前之前的轨迹（含当前帧）
        trajA_hist = trajA_km[: frame + 1]
        trajB_hist = trajB_km[: frame + 1]

        valid_A = ~np.isnan(trajA_hist[:, 0])
        valid_B = ~np.isnan(trajB_hist[:, 0])

        line_A.set_data(trajA_hist[valid_A, 0], trajA_hist[valid_A, 1])
        line_B.set_data(trajB_hist[valid_B, 0], trajB_hist[valid_B, 1])

        # 当前点
        if not np.isnan(trajA[frame, 0]):
            posA = trajA_km[frame]
            point_A.set_data(posA[0], posA[1])
        else:
            point_A.set_data([], [])

        if not np.isnan(trajB[frame, 0]):
            posB = trajB_km[frame]
            point_B.set_data(posB[0], posB[1])
        else:
            point_B.set_data([], [])

        # 干扰区提示
        if not np.isnan(trajA[frame, 0]) and in_jam_region(trajA[frame, :2], t):
            jam_text.set_text(f"t = {t:.1f}s：A 目标处于干扰区内")
        else:
            jam_text.set_text(f"t = {t:.1f}s")

        return line_A, line_B, point_A, point_B, jam_text

    ani = FuncAnimation(fig, update, frames=range(N_STEPS), interval=50, blit=False)

    if save_gif:
        # 默认保存到 results/ 目录下
        if gif_name is None:
            os.makedirs("results", exist_ok=True)
            gif_path = os.path.join("results", "scene.gif")
        else:
            # 如果用户传了完整路径，就按传入的来；否则仍然放到 results 下
            if os.path.dirname(gif_name):
                gif_path = gif_name
            else:
                os.makedirs("results", exist_ok=True)
                gif_path = os.path.join("results", gif_name)

        print("[scene_sim] 正在生成场景 GIF，请稍等...")
        ani.save(gif_path, writer="pillow", fps=10)
        print(f"[scene_sim] GIF 已保存为 {gif_path}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os

    os.system("clear")
    main()
