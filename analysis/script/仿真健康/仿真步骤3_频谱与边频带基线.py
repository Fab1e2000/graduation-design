"""
仿真步骤 3：健康仿真频谱分析（0~4000 Hz 全频谱）与啮合频率两侧边频带基线
产出：
  1. analysis/result/data/仿真步骤3_健康频域特征基线表.csv
  2. analysis/result/graphs/仿真步骤3_图1_四通道全频谱_0至4000Hz.png
  3. analysis/result/graphs/仿真步骤3_图2_啮合频率边频带基线放大.png
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
FM1 = 870.0                  # 一级啮合频率 (30 Hz * 29 齿)
F_MID = 30.0 * 29.0 / 95.0   # 中间轴转频 ≈ 9.1579 Hz（未来缺齿的理论调制频率）

CHANNELS = [
    ("RP1_A1", "输入轴 RP1 · X向 (径向)", "#1f77b4"),
    ("RP1_A2", "输入轴 RP1 · Y向 (切向)", "#d62728"),
    ("RP2_A1", "中间轴 RP2 · X向 (径向)", "#2ca02c"),
    ("RP2_A2", "中间轴 RP2 · Y向 (切向)", "#9467bd"),
]


def load_steady_fft(col: str) -> tuple[np.ndarray, np.ndarray]:
    """剔除前 0.05 s -> 重采样 12800 Hz -> 去均值 -> 汉宁窗单边幅值谱。"""
    t, acc, _, _ = load_simulation_csv(column=col, trim_start_s=TRIM_START_S)
    t_u = np.arange(t[0], t[-1], 1.0 / FS_TARGET)
    x = np.interp(t_u, t, acc)
    x = x - np.mean(x)

    n = len(x)
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(x * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS_TARGET)
    return freqs, amp


def peak_near(freqs: np.ndarray, amp: np.ndarray, target_hz: float, tol_hz: float = 3.5) -> float:
    """读取目标频率 ±tol_hz 窗口内的最大幅值（考虑 0.45 s 信号 df≈2.22 Hz）。"""
    mask = np.abs(freqs - target_hz) <= tol_hz
    return float(np.max(amp[mask])) if np.any(mask) else 0.0


def sideband_mean(freqs: np.ndarray, amp: np.ndarray, center_hz: float, f_mod: float = F_MID, orders: int = 3) -> float:
    """计算 center_hz 两侧 ±1..orders 阶理论边频带的平均幅值。"""
    vals = []
    for k in range(1, orders + 1):
        vals.append(peak_near(freqs, amp, center_hz - k * f_mod, tol_hz=2.5))
        vals.append(peak_near(freqs, amp, center_hz + k * f_mod, tol_hz=2.5))
    return float(np.mean(vals))


def main() -> None:
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    spectra = {}
    rows = []
    for col, desc, color in CHANNELS:
        f, a = load_steady_fft(col)
        spectra[col] = (f, a, desc, color)

        a1 = peak_near(f, a, 1 * FM1)
        a2 = peak_near(f, a, 2 * FM1)
        a3 = peak_near(f, a, 3 * FM1)
        a4 = peak_near(f, a, 4 * FM1)
        sb1 = sideband_mean(f, a, 1 * FM1)
        sb2 = sideband_mean(f, a, 2 * FM1)

        rows.append({
            "通道": col,
            "方向": desc,
            "1fm1_870Hz幅值": round(a1, 2),
            "2fm1_1740Hz幅值": round(a2, 2),
            "3fm1_2610Hz幅值": round(a3, 2),
            "4fm1_3480Hz幅值": round(a4, 2),
            "1fm1两侧边频均值": round(sb1, 2),
            "2fm1两侧边频均值": round(sb2, 2),
            "2fm1边频比_%": round(sb2 / a2 * 100.0, 2) if a2 > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    csv_path = RESULT_DATA_DIR / "仿真步骤3_健康频域特征基线表.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # === 图 1：四通道 0 ~ 4000 Hz 全频谱（同横轴；RP1组同纵轴、RP2组同纵轴） ===
    fmax = 4000.0
    rp1_ymax = max(
        float(np.max(spectra["RP1_A1"][1][spectra["RP1_A1"][0] <= fmax])),
        float(np.max(spectra["RP1_A2"][1][spectra["RP1_A2"][0] <= fmax])),
    )
    rp2_ymax = max(
        float(np.max(spectra["RP2_A1"][1][spectra["RP2_A1"][0] <= fmax])),
        float(np.max(spectra["RP2_A2"][1][spectra["RP2_A2"][0] <= fmax])),
    )
    rp1_ylim = float(np.ceil(rp1_ymax * 1.18 / 50.0) * 50.0)
    rp2_ylim = float(np.ceil(rp2_ymax * 1.18 / 50.0) * 50.0)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 9.8), sharex=True)
    harmonics = [(k * FM1, rf"${k}f_{{m1}}$={int(k*FM1)}Hz") for k in range(1, 5)]

    for idx, (col, desc, color) in enumerate(CHANNELS):
        ax = axes[idx]
        f, a, _, _ = spectra[col]
        m = f <= fmax
        ax.plot(f[m], a[m], color=color, linewidth=0.9)
        ylim = rp1_ylim if col.startswith("RP1") else rp2_ylim
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.linspace(0, ylim, 5))
        ax.set_ylabel(r"幅值 ($\mathrm{m/s^2}$)", fontsize=9)
        ax.set_title(f"{col} — {desc}", fontsize=10.5, fontweight="bold")
        ax.grid(True, alpha=0.25)

        for hf, hlabel in harmonics:
            ax.axvline(hf, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.6)
            if idx == 0:
                ax.text(hf + 25, ylim * 0.88, hlabel, color="#d62728", fontsize=8.5, va="top")

    axes[-1].set_xlim(0, fmax)
    axes[-1].set_xticks(np.arange(0, fmax + 1, 500))
    axes[-1].set_xlabel("频率 (Hz)（四通道统一横轴；RP1 两图同纵轴、RP2 两图同纵轴）", fontsize=9.5)
    fig.suptitle("健康仿真（30 Hz 稳态段）四通道全频谱 0 ~ 4000 Hz", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    g1 = RESULT_GRAPH_DIR / "仿真步骤3_图1_四通道全频谱_0至4000Hz.png"
    fig.savefig(g1, bbox_inches="tight")
    plt.close(fig)

    # === 图 2：1fm1 (870 Hz) 与 2fm1 (1740 Hz) 两侧边频带局部放大（RP2_A2 & RP1_A2） ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    focus_rows = [
        ("RP2_A2", "主力监测通道：RP2_A2（中间轴 Y向·缺齿所在轴）", "#9467bd"),
        ("RP1_A2", "对照通道：RP1_A2（输入轴 Y向）", "#d62728"),
    ]
    centers = [
        (FM1, r"一级啮合基频 $f_{m1}=870\ \mathrm{Hz}$ 两侧（±50 Hz）"),
        (2 * FM1, r"二倍啮合频率 $2f_{m1}=1740\ \mathrm{Hz}$ 两侧（±50 Hz）"),
    ]

    for r_idx, (col, r_title, color) in enumerate(focus_rows):
        f, a, _, _ = spectra[col]
        for c_idx, (fc, c_title) in enumerate(centers):
            ax = axes[r_idx, c_idx]
            m = (f >= fc - 50.0) & (f <= fc + 50.0)
            ax.plot(f[m], a[m], color=color, linewidth=1.2, marker="o", markersize=2.5)
            ymax_local = float(np.max(a[m])) * 1.22
            ax.set_ylim(0, ymax_local)

            # 标出中心啮合频率
            ax.axvline(fc, color="#d62728", linestyle="--", linewidth=1.2, label=f"中心 {fc:.0f} Hz")
            # 标出未来缺齿理论边频带 ±1, ±2, ±3 * f_mid (9.16 Hz)
            for k in (-3, -2, -1, 1, 2, 3):
                fsb = fc + k * F_MID
                lbl = r"理论缺齿边频 $\pm n f_{\mathrm{mid}}$ ($\pm 9.16\ \mathrm{Hz}$)" if k == 1 else None
                ax.axvline(fsb, color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.75, label=lbl)

            ax.set_xlim(fc - 50.0, fc + 50.0)
            ax.set_xticks(np.arange(fc - 50.0, fc + 51.0, 10.0))
            ax.set_title(f"{r_title}\n{c_title}", fontsize=9.8, fontweight="bold")
            ax.grid(True, alpha=0.25)
            if r_idx == 1:
                ax.set_xlabel("频率 (Hz)", fontsize=9.5)
            if c_idx == 0:
                ax.set_ylabel(r"幅值 ($\mathrm{m/s^2}$)", fontsize=9.5)
            if r_idx == 0 and c_idx == 1:
                ax.legend(loc="upper right", fontsize=8.2)

    fig.suptitle(
        r"健康仿真边频带基线检查：蓝点线为未来缺齿理论边频位置（$f_c \pm n \times 9.16\ \mathrm{Hz}$）",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    g2 = RESULT_GRAPH_DIR / "仿真步骤3_图2_啮合频率边频带基线放大.png"
    fig.savefig(g2, bbox_inches="tight")
    plt.close(fig)

    print("频域基线表已保存:", csv_path)
    print(df.to_string(index=False))
    print("\n已生成图表:")
    print(" -", g1)
    print(" -", g2)


if __name__ == "__main__":
    main()
