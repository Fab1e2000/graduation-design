"""
仿真步骤 4：健康仿真希尔伯特包络解调基线分析
流程（最经典的共振解调三步法）：
  1. 带通滤波：提取围绕 2fm1 (1740 Hz) 的高能量啮合频带（1500 ~ 2000 Hz）
  2. 希尔伯特变换：求出高频信号的瞬时包络外轮廓（剥掉高频载波，只留低频起伏）
  3. 包络谱 FFT：观察 0 ~ 60 Hz 低频段，检查中间轴转频 f_mid = 9.16 Hz 及其倍频的基线水平

产出：
  1. analysis/result/data/仿真步骤4_健康包络特征基线表.csv
  2. analysis/result/graphs/仿真步骤4_图1_包络解调过程演示_RP2_A2.png
  3. analysis/result/graphs/仿真步骤4_图2_四通道健康包络谱基线_0至60Hz.png
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
from scipy.signal import butter, filtfilt, hilbert

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

TRIM_START_S = 0.05
FS_TARGET = 12800.0
BAND_LOW = 1500.0            # 带通滤波下限 (Hz)
BAND_HIGH = 2000.0           # 带通滤波上限 (Hz)，覆盖 2fm1 = 1740 Hz 及两侧主要边频
F_MID = 30.0 * 29.0 / 95.0   # 中间轴转频 ≈ 9.1579 Hz（未来缺齿的理论冲击基频）
F_IN = 30.0                  # 输入轴转频 = 30.0 Hz

CHANNELS = [
    ("RP1_A1", "输入轴 RP1 · X向 (径向)", "#1f77b4"),
    ("RP1_A2", "输入轴 RP1 · Y向 (切向)", "#d62728"),
    ("RP2_A1", "中间轴 RP2 · X向 (径向)", "#2ca02c"),
    ("RP2_A2", "中间轴 RP2 · Y向 (切向)", "#9467bd"),
]


def load_steady_uniform(col: str) -> tuple[np.ndarray, np.ndarray]:
    """剔除前 0.05 s 启动瞬态 -> 重采样 12800 Hz -> 去均值。"""
    t, acc, _, _ = load_simulation_csv(column=col, trim_start_s=TRIM_START_S)
    t_u = np.arange(t[0], t[-1], 1.0 / FS_TARGET)
    x = np.interp(t_u, t, acc)
    x = x - np.mean(x)
    return t_u, x


def bandpass_and_envelope(x: np.ndarray, fs: float, low_hz: float, high_hz: float):
    """4阶巴特沃斯零相位带通滤波 + 希尔伯特包络提取 + 包络谱计算。"""
    nyq = 0.5 * fs
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    x_filt = filtfilt(b, a, x)

    # 希尔伯特解析信号求包络
    analytic = hilbert(x_filt)
    env = np.abs(analytic)

    # 去均值后加汉宁窗做包络谱 FFT
    env0 = env - np.mean(env)
    n = len(env0)
    win = np.hanning(n)
    env_amp = np.abs(np.fft.rfft(env0 * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return x_filt, env, freqs, env_amp


def peak_near(freqs: np.ndarray, amp: np.ndarray, target_hz: float, tol_hz: float = 2.5) -> float:
    mask = np.abs(freqs - target_hz) <= tol_hz
    return float(np.max(amp[mask])) if np.any(mask) else 0.0


def main() -> None:
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    rows = []

    for col, desc, color in CHANNELS:
        t, x = load_steady_uniform(col)
        x_filt, env, f_env, a_env = bandpass_and_envelope(x, FS_TARGET, BAND_LOW, BAND_HIGH)
        results[col] = (t, x, x_filt, env, f_env, a_env, desc, color)

        m60 = (f_env >= 3.0) & (f_env <= 60.0)
        mean_floor = float(np.mean(a_env[m60]))
        a_1x = peak_near(f_env, a_env, 1 * F_MID)
        a_2x = peak_near(f_env, a_env, 2 * F_MID)
        a_3x = peak_near(f_env, a_env, 3 * F_MID)
        a_in = peak_near(f_env, a_env, F_IN)

        rows.append({
            "通道": col,
            "方向": desc,
            "解调频带_Hz": f"{int(BAND_LOW)}-{int(BAND_HIGH)}",
            "1x中间轴_9.16Hz": round(a_1x, 2),
            "2x中间轴_18.32Hz": round(a_2x, 2),
            "3x中间轴_27.47Hz": round(a_3x, 2),
            "输入轴_30Hz": round(a_in, 2),
            "低频平均底噪": round(mean_floor, 2),
            "9.16Hz峰噪比": round(a_1x / mean_floor, 2) if mean_floor > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    csv_path = RESULT_DATA_DIR / "仿真步骤4_健康包络特征基线表.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # === 图 1：用主力通道 RP2_A2 直观演示“带通滤波 -> 剥出红线包络 -> 包络谱”三步 ===
    t, x, x_filt, env, f_env, a_env, _, _ = results["RP2_A2"]
    mask_t = (t >= 0.10) & (t <= 0.35)  # 0.25 s 窗口（约 2.3 圈中间轴）

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6))

    # 上图：带通滤波后的高频波 + 红色包络外轮廓
    axes[0].plot(t[mask_t], x_filt[mask_t], color="#9467bd", alpha=0.55, linewidth=0.7, label="① 带通滤波后高频波 (1500~2000 Hz)")
    axes[0].plot(t[mask_t], env[mask_t], color="#d62728", linewidth=1.6, label="② 希尔伯特提取的包络外轮廓（把高频壳子剥掉）")
    axes[0].set_title("主力通道 RP2_A2：带通滤波 + 希尔伯特包络提取（0.10 ~ 0.35 s）", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("仿真时间 (s)", fontsize=9.5)
    axes[0].set_ylabel(r"加速度 ($\mathrm{m/s^2}$)", fontsize=9.5)
    axes[0].set_xlim(0.10, 0.35)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.25)

    # 下图：对红色包络线做 FFT -> 0~60 Hz 包络谱
    m60 = f_env <= 60.0
    axes[1].plot(f_env[m60], a_env[m60], color="#d62728", linewidth=1.3, marker="o", markersize=3)
    ymax1 = float(np.max(a_env[m60])) * 1.25
    axes[1].set_ylim(0, ymax1)
    for k in (1, 2, 3):
        fx = k * F_MID
        lbl = r"未来缺齿理论冲击线 $n \times f_{\mathrm{mid}}$ (9.16, 18.32, 27.47 Hz)" if k == 1 else None
        axes[1].axvline(fx, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.8, label=lbl)
        axes[1].text(fx + 0.5, ymax1 * 0.88, f"{k}x={fx:.1f}Hz", color="#1f77b4", fontsize=8.5, va="top")
    axes[1].axvline(F_IN, color="#2ca02c", linestyle=":", linewidth=1.1, alpha=0.8, label="输入轴转频 30.0 Hz")
    axes[1].set_title("③ 对红色包络线做 FFT 得到包络谱（0 ~ 60 Hz）——健康状态下 9.16 Hz 处无显著冲击峰", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("包络频率 (Hz)", fontsize=9.5)
    axes[1].set_ylabel(r"包络幅值 ($\mathrm{m/s^2}$)", fontsize=9.5)
    axes[1].set_xlim(0, 60.0)
    axes[1].set_xticks(np.arange(0, 61, 5))
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    g1 = RESULT_GRAPH_DIR / "仿真步骤4_图1_包络解调过程演示_RP2_A2.png"
    fig.savefig(g1, bbox_inches="tight")
    plt.close(fig)

    # === 图 2：四通道健康包络谱基线对照（0 ~ 60 Hz，同横轴；RP1组同纵轴、RP2组同纵轴） ===
    rp1_max = max(
        float(np.max(results["RP1_A1"][5][results["RP1_A1"][4] <= 60.0])),
        float(np.max(results["RP1_A2"][5][results["RP1_A2"][4] <= 60.0])),
    )
    rp2_max = max(
        float(np.max(results["RP2_A1"][5][results["RP2_A1"][4] <= 60.0])),
        float(np.max(results["RP2_A2"][5][results["RP2_A2"][4] <= 60.0])),
    )
    rp1_ylim = float(np.ceil(rp1_max * 1.25 / 10.0) * 10.0)
    rp2_ylim = float(np.ceil(rp2_max * 1.25 / 5.0) * 5.0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9.4), sharex=True)
    for idx, (col, desc, color) in enumerate(CHANNELS):
        ax = axes[idx]
        _, _, _, _, f_env, a_env, _, _ = results[col]
        m = f_env <= 60.0
        ax.plot(f_env[m], a_env[m], color=color, linewidth=1.2, marker="o", markersize=2.5)
        ylim = rp1_ylim if col.startswith("RP1") else rp2_ylim
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.linspace(0, ylim, 5))
        ax.set_ylabel(r"包络幅值 ($\mathrm{m/s^2}$)", fontsize=9)
        ax.set_title(f"{col} — {desc}（解调频带 1500~2000 Hz）", fontsize=10.2, fontweight="bold")
        ax.grid(True, alpha=0.25)

        for k in (1, 2, 3):
            fx = k * F_MID
            ax.axvline(fx, color="#1f77b4", linestyle="--", linewidth=0.9, alpha=0.7)
            if idx == 0:
                ax.text(fx + 0.5, ylim * 0.88, f"{k}f_mid", color="#1f77b4", fontsize=8.2, va="top")
        ax.axvline(F_IN, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.7)
        if idx == 0:
            ax.text(F_IN + 0.5, ylim * 0.88, "f_in=30Hz", color="#2ca02c", fontsize=8.2, va="top")

    axes[-1].set_xlim(0, 60.0)
    axes[-1].set_xticks(np.arange(0, 61, 5))
    axes[-1].set_xlabel("包络频率 (Hz)（四通道统一横轴；RP1 两图同纵轴、RP2 两图同纵轴）", fontsize=9.5)
    fig.suptitle("健康仿真四通道包络谱基线（0 ~ 60 Hz，监控缺齿冲击基频 9.16 Hz）", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    g2 = RESULT_GRAPH_DIR / "仿真步骤4_图2_四通道健康包络谱基线_0至60Hz.png"
    fig.savefig(g2, bbox_inches="tight")
    plt.close(fig)

    print("包络基线表已保存:", csv_path)
    print(df.to_string(index=False))
    print("\n已生成图表:")
    print(" -", g1)
    print(" -", g2)


if __name__ == "__main__":
    main()
