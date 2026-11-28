"""
仿真步骤 2：健康仿真时域特征指标提取（有量纲 + 无量纲）
产出：
  1. analysis/result/data/仿真步骤2_健康时域特征表.csv
  2. analysis/result/graphs/仿真步骤2_图1_四通道有量纲与无量纲指标对比.png
  3. analysis/result/graphs/仿真步骤2_图2_加速度概率分布与峭度直觉.png
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
from scipy.stats import kurtosis, norm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from 数据读取模块 import (
    SIM_HEALTHY_DATA_DIR as RESULT_DATA_DIR,
    SIM_HEALTHY_GRAPH_DIR as RESULT_GRAPH_DIR,
    load_simulation_csv,
)

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

CHANNELS = [
    ("RP1_A1", "RP1_X\n(输入径向)", "#1f77b4"),
    ("RP1_A2", "RP1_Y\n(输入切向)", "#d62728"),
    ("RP2_A1", "RP2_X\n(中间径向)", "#2ca02c"),
    ("RP2_A2", "RP2_Y\n(中间切向)", "#9467bd"),
]

TRIM_START_S = 0.05
FS_TARGET = 12800.0


def load_steady_uniform(col: str) -> tuple[np.ndarray, np.ndarray]:
    """剔除前 0.05 s 启动瞬态，重采样到严格等间隔 12800 Hz，并去均值。"""
    t, acc, _, _ = load_simulation_csv(column=col, trim_start_s=TRIM_START_S)
    t_u = np.arange(t[0], t[-1], 1.0 / FS_TARGET)
    a_u = np.interp(t_u, t, acc)
    a_u = a_u - np.mean(a_u)
    return t_u, a_u


def extract_time_features(x: np.ndarray) -> dict:
    """提取 6 个最经典的机械振动时域指标。"""
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    p2p = float(np.max(x) - np.min(x))
    mean_abs = float(np.mean(np.abs(x)))
    kurt = float(kurtosis(x, fisher=False))  # 正态分布基准 = 3.0
    crest = peak / rms if rms > 0 else 0.0
    impulse = peak / mean_abs if mean_abs > 0 else 0.0
    return {
        "有效值RMS_m_s2": round(rms, 2),
        "峰值_m_s2": round(peak, 2),
        "峰峰值_m_s2": round(p2p, 2),
        "峭度": round(kurt, 3),
        "峰值因子": round(crest, 3),
        "脉冲因子": round(impulse, 3),
    }


def main() -> None:
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    signals = {}
    rows = []
    for col, short_label, color in CHANNELS:
        _, x = load_steady_uniform(col)
        signals[col] = x
        feat = extract_time_features(x)
        rows.append({"通道": col, "标签": short_label.replace("\n", " "), **feat})

    df = pd.DataFrame(rows)
    csv_path = RESULT_DATA_DIR / "仿真步骤2_健康时域特征表.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # === 图 1：左图（有量纲）+ 右图（无量纲）对比 ===
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x_pos = np.arange(len(CHANNELS))
    width = 0.25
    x_labels = [c[1] for c in CHANNELS]

    # 左图：有量纲指标
    ax = axes[0]
    b1 = ax.bar(x_pos - width, df["有效值RMS_m_s2"], width, label="有效值 RMS", color="#1f77b4")
    b2 = ax.bar(x_pos, df["峰值_m_s2"], width, label="峰值 Peak", color="#ff7f0e")
    b3 = ax.bar(x_pos + width, df["峰峰值_m_s2"], width, label="峰峰值 Peak-to-Peak", color="#2ca02c")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.set_ylabel(r"加速度 ($\mathrm{m/s^2}$)", fontsize=10)
    ax.set_title("有量纲指标（反映振动“有多大”）", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 120, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, df["峰峰值_m_s2"].max() * 1.15)

    # 右图：无量纲指标
    ax = axes[1]
    b4 = ax.bar(x_pos - width, df["峭度"], width, label="峭度 Kurtosis", color="#9467bd")
    b5 = ax.bar(x_pos, df["峰值因子"], width, label="峰值因子 Crest", color="#d62728")
    b6 = ax.bar(x_pos + width, df["脉冲因子"], width, label="脉冲因子 Impulse", color="#17becf")
    ax.axhline(3.0, color="#9467bd", linestyle="--", linewidth=1.1, alpha=0.75, label="标准正态峭度基准 = 3.0")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.set_ylabel("无量纲比值", fontsize=10)
    ax.set_title("无量纲指标（反映波形“尖不尖”，不受幅值缩放影响）", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b4, b5, b6):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(df["脉冲因子"].max(), df["峭度"].max()) * 1.25)

    fig.suptitle("健康仿真（30 Hz 稳态段）时域特征基准", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    g1 = RESULT_GRAPH_DIR / "仿真步骤2_图1_四通道有量纲与无量纲指标对比.png"
    fig.savefig(g1, bbox_inches="tight")
    plt.close(fig)

    # === 图 2：标准化加速度分布直方图 vs 标准正态曲线（直观理解峭度） ===
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True, sharey=True)
    focus = [
        ("RP2_A2", "主力监测通道：RP2_A2（中间轴 Y向）", "#9467bd"),
        ("RP1_A2", "对照通道：RP1_A2（输入轴 Y向）", "#d62728"),
    ]
    z_grid = np.linspace(-5, 5, 400)
    normal_pdf = norm.pdf(z_grid, 0, 1)

    for ax, (col, title, color) in zip(axes, focus):
        x = signals[col]
        z = x / np.std(x)  # 标准化为均值0、标准差1，便于同分度值对比
        k_val = float(kurtosis(x, fisher=False))
        ax.hist(z, bins=70, density=True, color=color, alpha=0.55, edgecolor="white", linewidth=0.3, label=f"健康仿真分布 (峭度={k_val:.2f})")
        ax.plot(z_grid, normal_pdf, color="black", linestyle="--", linewidth=1.4, label="标准正态分布 (峭度=3.00)")
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        ax.set_xlabel(r"标准化加速度 ($x / \sigma$，单位：倍标准差)", fontsize=9.5)
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(0, 0.50)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("概率密度", fontsize=9.5)

    fig.suptitle("直觉理解“峭度”：健康仿真分布与标准正态钟形曲线几乎贴合", fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout()
    g2 = RESULT_GRAPH_DIR / "仿真步骤2_图2_加速度概率分布与峭度直觉.png"
    fig.savefig(g2, bbox_inches="tight")
    plt.close(fig)

    print("时域特征表已保存:", csv_path)
    print(df.to_string(index=False))
    print("\n已生成图表:")
    print(" -", g1)
    print(" -", g2)


if __name__ == "__main__":
    main()
