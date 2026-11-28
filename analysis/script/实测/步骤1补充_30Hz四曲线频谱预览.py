"""
快速频谱感性预览：将图 3 中的 4 条 30 Hz 曲线（实测一/二/三级 + Abaqus 健康仿真）
做单边幅值谱（FFT），统一横轴频率分度，方便建立“时域波形 -> 频域峰值”的直觉认知。
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from 数据读取模块 import (
    MEASURED_GRAPH_DIR as RESULT_GRAPH_DIR,
    load_health_case,
    load_simulation_csv,
)

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def compute_fft_amp(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """加汉宁窗计算单边幅值谱（带窗函数幅值恢复系数 2.0）。"""
    x0 = x - np.mean(x)
    n = len(x0)
    win = np.hanning(n)
    # 单边幅值谱，除以 sum(win)/2 保证正弦波峰值准确还原
    spec = np.abs(np.fft.rfft(x0 * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def load_sim_uniform(column: str = "RP2_A2", trim_start_s: float = 0.05, fs_target: float = 12800.0):
    """读取仿真稳态段，并线性重采样到严格等间隔 12800 Hz。"""
    t, acc, _, _ = load_simulation_csv(column=column, trim_start_s=trim_start_s)
    t_uni = np.arange(t[0], t[-1], 1.0 / fs_target)
    acc_uni = np.interp(t_uni, t, acc)
    return t_uni, acc_uni, fs_target


def top_peaks(freqs: np.ndarray, amp: np.ndarray, fmin: float = 5.0, fmax: float = 2500.0, k: int = 5, sep_hz: float = 15.0):
    """找出指定频段内幅值最高的前 k 个主峰频率。"""
    mask = (freqs >= fmin) & (freqs <= fmax)
    f_sub = freqs[mask]
    a_sub = amp[mask].copy()
    peaks = []
    for _ in range(k):
        if len(a_sub) == 0 or np.max(a_sub) <= 0:
            break
        idx = int(np.argmax(a_sub))
        fp = float(f_sub[idx])
        ap = float(a_sub[idx])
        peaks.append((fp, ap))
        a_sub[np.abs(f_sub - fp) < sep_hz] = 0.0
    return peaks


def plot_four_spectra(fmax: float, xticks_step: float, out_path: Path, title_suffix: str) -> None:
    cases = [
        ("第一级-30", "#1f77b4", "实测 · 第一级-30（输入电机侧）"),
        ("第二级-30", "#2ca02c", "实测 · 第二级-30（定轴箱输出级）"),
        ("第三级-30", "#ff7f0e", "实测 · 第三级-30（行星级）"),
    ]

    # 先计算 3 条实测频谱，确定统一纵轴上限
    meas_results = []
    ymax_meas = 0.0
    for cname, color, label in cases:
        _, acc, fs, _ = load_health_case(cname, remove_dc=True)
        f, a = compute_fft_amp(acc, fs)
        m = f <= fmax
        ymax_meas = max(ymax_meas, float(np.max(a[m])))
        meas_results.append((f[m], a[m], color, label, cname))

    # 向上取整到一个清爽的纵轴刻度上限
    ymax_meas_nice = np.ceil(ymax_meas * 1.15 * 1000) / 1000.0

    # 计算仿真稳态频谱
    _, acc_sim, fs_sim = load_sim_uniform(column="RP2_A2", trim_start_s=0.05)
    f_sim, a_sim = compute_fft_amp(acc_sim, fs_sim)
    m_sim = f_sim <= fmax

    # 理论特征频率（30 Hz 工况）
    ref_lines = [
        (30.0, "输入转频 30Hz", "#888888"),
        (329.7, r"二级啮合 $f_{m2}$=329.7Hz", "#2ca02c"),
        (870.0, r"一级啮合 $f_{m1}$=870Hz", "#d62728"),
        (1740.0, r"$2f_{m1}$=1740Hz", "#d62728"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9.8), sharex=True)

    for i, (f, a, color, label, cname) in enumerate(meas_results):
        ax = axes[i]
        ax.plot(f, a, color=color, linewidth=0.8)
        ax.set_ylim(0, ymax_meas_nice)
        ax.set_ylabel(r"幅值 ($\mathrm{m/s^2}$)", fontsize=9)
        pks = top_peaks(f, a, fmin=5.0, fmax=fmax, k=3)
        pk_str = " | ".join([f"{fp:.1f} Hz" for fp, _ in pks])
        ax.set_title(f"{label}    [前3主峰: {pk_str}]", fontsize=10.2, fontweight="bold")
        ax.grid(True, alpha=0.25)

    # 第 4 行：仿真频谱
    ax_s = axes[3]
    ax_s.plot(f_sim[m_sim], a_sim[m_sim], color="#9467bd", linewidth=0.9)
    ax_s.set_ylim(0, float(np.max(a_sim[m_sim])) * 1.18)
    ax_s.set_ylabel(r"幅值 ($\mathrm{m/s^2}$)", fontsize=9)
    pks_sim = top_peaks(f_sim[m_sim], a_sim[m_sim], fmin=5.0, fmax=fmax, k=3)
    pk_sim_str = " | ".join([f"{fp:.1f} Hz" for fp, _ in pks_sim])
    ax_s.set_title(
        f"仿真 · Abaqus 健康基准（30 Hz，RP2_A2 稳态段 0.05~0.50 s）    [前3主峰: {pk_sim_str}]",
        fontsize=10.2,
        fontweight="bold",
    )
    ax_s.grid(True, alpha=0.25)
    ax_s.set_xlabel("频率 (Hz)（四张图统一频率分度；上面 3 张实测图统一纵轴分度）", fontsize=9.5)
    ax_s.set_xlim(0, fmax)
    ax_s.set_xticks(np.arange(0, fmax + 1, xticks_step))

    # 在 4 张图上统一画理论频率参考竖线
    for ax_idx, ax in enumerate(axes):
        y_top = ax.get_ylim()[1]
        for rf, rlabel, rcol in ref_lines:
            if rf <= fmax:
                ax.axvline(rf, color=rcol, linestyle="--", linewidth=0.9, alpha=0.55)
                if ax_idx == 0:
                    ax.text(
                        rf + fmax * 0.006,
                        y_top * 0.88,
                        rlabel,
                        color=rcol,
                        fontsize=8.2,
                        va="top",
                    )

    fig.suptitle(f"30 Hz 工况四条曲线频谱对照（{title_suffix}）", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # 控制台打印前5主峰方便核对
    print(f"\n=== 频段 0 ~ {fmax:.0f} Hz 前5主峰 ===")
    for f, a, _, label, _ in meas_results:
        pks = top_peaks(f, a, fmin=5.0, fmax=fmax, k=5)
        print(f"  {label}: " + ", ".join([f"{fp:.1f}Hz({ap:.4f})" for fp, ap in pks]))
    pks_s = top_peaks(f_sim[m_sim], a_sim[m_sim], fmin=5.0, fmax=fmax, k=5)
    print("  仿真 RP2_A2: " + ", ".join([f"{fp:.1f}Hz({ap:.1f})" for fp, ap in pks_s]))


def main() -> None:
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    g_wide = RESULT_GRAPH_DIR / "步骤1_图4_30Hz四曲线频谱_0至2500Hz.png"
    g_zoom = RESULT_GRAPH_DIR / "步骤1_图5_30Hz四曲线频谱_0至1000Hz放大.png"

    plot_four_spectra(2500.0, 250.0, g_wide, "0 ~ 2500 Hz 全景")
    plot_four_spectra(1000.0, 100.0, g_zoom, "0 ~ 1000 Hz 局部放大")
    print("\n已生成频谱图:")
    print(" -", g_wide)
    print(" -", g_zoom)


if __name__ == "__main__":
    main()
