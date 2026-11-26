"""
仿真步骤 1：Abaqus 健康仿真数据体检、四通道对比与预处理检查
产出：
  1. analysis/result/data/仿真步骤1_健康四通道基础体检表.csv
  2. analysis/result/graphs/仿真步骤1_图1_四通道全时段波形与启动段剔除.png
  3. analysis/result/graphs/仿真步骤1_图2_时间步长检查与啮合周期细节.png
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from 数据读取模块 import (
    RESULT_DATA_DIR,
    RESULT_GRAPH_DIR,
    load_simulation_csv,
)

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

CHANNELS = [
    ("RP1_A1", "输入轴 RP1 · X向 (径向/中心距方向)", "#1f77b4"),
    ("RP1_A2", "输入轴 RP1 · Y向 (切向/啮合受力方向)", "#d62728"),
    ("RP2_A1", "中间轴 RP2 · X向 (径向/中心距方向)", "#2ca02c"),
    ("RP2_A2", "中间轴 RP2 · Y向 (切向/啮合受力方向)", "#9467bd"),
]

TRIM_START_S = 0.05  # 剔除前 0.05 s 升速启动瞬态


def main() -> None:
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # 读取全部 4 个通道（不截断，保留 0~0.5 s 完整数据以便体检）
    all_data = {}
    rows = []
    ymax = 0.0

    for col, desc, color in CHANNELS:
        t, acc, fs_est, meta = load_simulation_csv(column=col, trim_start_s=0.0)
        all_data[col] = (t, acc, desc, color, meta)
        ymax = max(ymax, float(np.max(np.abs(acc))))

        steady = t >= TRIM_START_S
        acc_st = acc[steady]
        rows.append({
            "通道": col,
            "物理位置与方向": desc,
            "总点数": len(t),
            "稳态点数": int(np.sum(steady)),
            "稳态均值_m_s2": round(float(np.mean(acc_st)), 2),
            "稳态RMS_m_s2": round(float(np.sqrt(np.mean(acc_st ** 2))), 2),
            "稳态峰值_m_s2": round(float(np.max(np.abs(acc_st))), 2),
            "稳态峰峰值_m_s2": round(float(np.max(acc_st) - np.min(acc_st)), 2),
        })

    df = pd.DataFrame(rows)
    csv_path = RESULT_DATA_DIR / "仿真步骤1_健康四通道基础体检表.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 统一纵轴范围，向上取整到整百，确保 4 通道同分度值对比
    ylim = float(np.ceil(ymax / 200.0) * 200.0)
    yticks = np.linspace(-ylim, ylim, 5)

    # === 图 1：四通道 0~0.5 s 全时段波形（同横纵轴分度） ===
    fig, axes = plt.subplots(4, 1, figsize=(12, 9.6), sharex=True, sharey=True)
    for ax, (col, desc, color) in zip(axes, CHANNELS):
        t, acc, _, _, _ = all_data[col]
        st_rms = float(np.sqrt(np.mean(acc[t >= TRIM_START_S] ** 2)))
        ax.plot(t, acc, color=color, linewidth=0.65)
        ax.axvspan(0.0, TRIM_START_S, color="#ffcccc", alpha=0.45, label="前 0.05 s 启动瞬态（后续剔除）")
        ax.axvline(TRIM_START_S, color="#d62728", linestyle="--", linewidth=0.9)
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(r"加速度 ($\mathrm{m/s^2}$)", fontsize=9)
        ax.set_title(rf"{col} — {desc}    [稳态 RMS = {st_rms:.1f} $\mathrm{{m/s^2}}$]", fontsize=10.2, fontweight="bold")
        ax.grid(True, alpha=0.25)
        if col == "RP1_A1":
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlim(0.0, 0.5)
    axes[-1].set_xticks(np.arange(0.0, 0.501, 0.05))
    axes[-1].set_xlabel("仿真时间 (s)（四通道统一横纵轴分度值）", fontsize=9.5)
    fig.suptitle("健康仿真四通道全时段波形对照（30 Hz，Abaqus 导出）", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    g1 = RESULT_GRAPH_DIR / "仿真步骤1_图1_四通道全时段波形与启动段剔除.png"
    fig.savefig(g1, bbox_inches="tight")
    plt.close(fig)

    # === 图 2：时间步长均匀性检查 + 0.01 s 极局部啮合周期波形 ===
    t0, acc_rp2_y, _, _, _ = all_data["RP2_A2"]
    _, acc_rp2_x, _, _, _ = all_data["RP2_A1"]
    dt_us = np.diff(t0) * 1e6
    ideal_dt_us = 1e6 / 12800.0  # 78.125 us

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6))

    # 上图：时间步长抖动检查
    axes[0].plot(t0[1:], dt_us, color="#1f77b4", linewidth=0.7, label="Abaqus 实际输出时间步长")
    axes[0].axhline(ideal_dt_us, color="#d62728", linestyle="--", linewidth=1.1, label=f"理想等间隔步长 ({ideal_dt_us:.3f} μs = 1/12800 s)")
    axes[0].set_title(
        f"预处理检查 1：时间步长均匀性（范围 {dt_us.min():.3f} ~ {dt_us.max():.3f} μs）——做 FFT 前重采样为严格等间隔",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("仿真时间 (s)", fontsize=9)
    axes[0].set_ylabel("相邻采样步长 (μs)", fontsize=9)
    axes[0].set_xlim(0.0, 0.5)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 下图：0.20 ~ 0.21 s 局部 0.01 s 放大（约 8.7 个一级啮合周期，T_m1 = 1/870 s = 1.149 ms）
    mask_zoom = (t0 >= 0.20) & (t0 <= 0.21)
    t_ms = (t0[mask_zoom] - 0.20) * 1000.0
    axes[1].plot(t_ms, acc_rp2_y[mask_zoom], color="#9467bd", linewidth=1.2, label="RP2_A2（中间轴 Y向·切向）")
    axes[1].plot(t_ms, acc_rp2_x[mask_zoom], color="#2ca02c", linewidth=1.0, alpha=0.85, label="RP2_A1（中间轴 X向·径向）")

    # 标出理论一级啮合周期 Tm1 = 1000 / 870 ms ≈ 1.1494 ms
    tm1_ms = 1000.0 / 870.0
    for k in range(1, 9):
        axes[1].axvline(k * tm1_ms, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
    axes[1].set_title(
        rf"预处理检查 2：稳态局部 10 ms 放大（竖虚线间距 = 1 个啮合周期 $T_{{m1}} = 1/870\ \mathrm{{s}} \approx 1.15\ \mathrm{{ms}}$）",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].set_xlabel("相对时间 (ms)（窗口 0.20 ~ 0.21 s）", fontsize=9)
    axes[1].set_ylabel(r"加速度 ($\mathrm{m/s^2}$)", fontsize=9)
    axes[1].set_xlim(0.0, 10.0)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    g2 = RESULT_GRAPH_DIR / "仿真步骤1_图2_时间步长检查与啮合周期细节.png"
    fig.savefig(g2, bbox_inches="tight")
    plt.close(fig)

    print("体检表已保存:", csv_path)
    print(df.to_string(index=False))
    print(f"\n时间步长统计: min={dt_us.min():.4f} us, median={np.median(dt_us):.4f} us, max={dt_us.max():.4f} us")
    print("已生成图表:")
    print(" -", g1)
    print(" -", g2)


if __name__ == "__main__":
    main()
