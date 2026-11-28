"""
仿真步骤 5：健康仿真 STFT 短时傅里叶时频分析与平稳性验证
参数说明：
  - 采样率 fs = 12800 Hz（全时段 0~0.50 s 线性重采样为严格等间隔）
  - 窗长 nperseg = 256 点（20 ms，远小于缺齿冲击周期 109 ms，兼顾时间与频率分辨率）
  - 重叠 noverlap = 240 点（步进 1.25 ms，画面平滑细腻）

产出：
  1. analysis/result/data/仿真步骤5_健康时频平稳性基线表.csv
  2. analysis/result/graphs/仿真步骤5_图1_四通道STFT时频图_0至4000Hz.png
  3. analysis/result/graphs/仿真步骤5_图2_主力通道RP2_A2时频与能量演化.png
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
from scipy.signal import stft

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

FS_TARGET = 12800.0
TRIM_START_S = 0.05
FM1 = 870.0
NPERSEG = 256
NOVERLAP = 240
NFFT = 512  # 补零细化频率显示网格

CHANNELS = [
    ("RP1_A1", "输入轴 RP1 · X向 (径向)"),
    ("RP1_A2", "输入轴 RP1 · Y向 (切向)"),
    ("RP2_A1", "中间轴 RP2 · X向 (径向)"),
    ("RP2_A2", "中间轴 RP2 · Y向 (切向)"),
]


def load_full_uniform(col: str) -> tuple[np.ndarray, np.ndarray]:
    """读取 0~0.50 s 完整仿真信号（保留前 0.05 s 启动段以展示升速轨迹），重采样到严格 12800 Hz。"""
    t, acc, _, _ = load_simulation_csv(column=col, trim_start_s=0.0)
    t_u = np.arange(0.0, t[-1], 1.0 / FS_TARGET)
    x = np.interp(t_u, t, acc)
    x = x - np.mean(x[t_u >= TRIM_START_S])
    return t_u, x


def compute_stft_db(x: np.ndarray, fs: float):
    """计算单边 STFT 幅值与 dB 谱。"""
    f, t_seg, Zxx = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        boundary="zeros",
    )
    amp = np.abs(Zxx) * 2.0
    amp_db = 20.0 * np.log10(np.maximum(amp, 1e-6))
    return f, t_seg, amp, amp_db


def main() -> None:
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    stft_res = {}
    rows = []

    for col, desc in CHANNELS:
        t_u, x = load_full_uniform(col)
        f, t_seg, amp, amp_db = compute_stft_db(x, FS_TARGET)
        stft_res[col] = (f, t_seg, amp, amp_db, desc)

        # 稳态段（0.06 ~ 0.48 s，避开首尾窗口边缘）能量平稳性统计
        m_st = (t_seg >= 0.06) & (t_seg <= 0.48)
        m_band_2fm1 = (f >= 2 * FM1 - 100.0) & (f <= 2 * FM1 + 100.0)
        m_4k = f <= 4000.0

        frame_energy_all = np.sum(amp[m_4k][:, m_st] ** 2, axis=0)
        frame_energy_2fm1 = np.sum(amp[m_band_2fm1][:, m_st] ** 2, axis=0)
        band_ratio = float(np.mean(frame_energy_2fm1 / np.maximum(frame_energy_all, 1e-12)) * 100.0)

        rms_frame = np.sqrt(frame_energy_all)
        cv_pct = float(np.std(rms_frame) / np.mean(rms_frame) * 100.0)

        rows.append({
            "通道": col,
            "方向": desc,
            "稳态帧数": int(np.sum(m_st)),
            "稳态时频能量波动CV_%": round(cv_pct, 2),
            "2fm1频带能量占比_%": round(band_ratio, 2),
            "峰值分贝_dB": round(float(np.max(amp_db[m_4k][:, m_st])), 2),
        })

    df = pd.DataFrame(rows)
    csv_path = RESULT_DATA_DIR / "仿真步骤5_健康时频平稳性基线表.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # === 图 1：四通道 2x2 STFT 时频图（0~0.50 s，0~4000 Hz，统一横纵轴分度） ===
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True, sharey=True)
    for idx, (col, desc) in enumerate(CHANNELS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        f, t_seg, _, amp_db, _ = stft_res[col]
        m_f = f <= 4000.0

        # 每个通道锁定自身峰值往下 45 dB 的动态范围，保证亮带与底噪反差清晰
        vmax = float(np.percentile(amp_db[m_f], 99.8))
        vmin = vmax - 45.0

        pcm = ax.pcolormesh(
            t_seg,
            f[m_f],
            amp_db[m_f],
            shading="gouraud",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(pcm, ax=ax, pad=0.015)
        cbar.set_label("幅值 (dB)", fontsize=8.5)

        # 标出 0.05 s 启动/稳态分界线 & 啮合谐波参考线
        ax.axvline(TRIM_START_S, color="#00e5ff", linestyle="--", linewidth=1.1, alpha=0.85)
        for k in (1, 2, 3, 4):
            ax.axhline(k * FM1, color="white", linestyle=":", linewidth=0.75, alpha=0.55)
            if c == 1:
                ax.text(0.495, k * FM1 + 50, f"{k}fm1", color="white", fontsize=8, ha="right", alpha=0.85)

        ax.set_title(f"{col} — {desc}", fontsize=10.5, fontweight="bold")
        if r == 1:
            ax.set_xlabel("仿真时间 (s)（青虚线左侧为前 0.05 s 启动升速段）", fontsize=9)
        if c == 0:
            ax.set_ylabel("频率 (Hz)", fontsize=9.5)
        ax.set_xlim(0.0, 0.50)
        ax.set_ylim(0.0, 4000.0)
        ax.set_xticks(np.arange(0.0, 0.501, 0.05))
        ax.set_yticks(np.arange(0.0, 4001.0, 500.0))

    fig.suptitle("健康仿真四通道 STFT 时频图（0 ~ 4000 Hz，窗长 20 ms）——笔直水平亮带表明稳态无冲击", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    g1 = RESULT_GRAPH_DIR / "仿真步骤5_图1_四通道STFT时频图_0至4000Hz.png"
    fig.savefig(g1, bbox_inches="tight")
    plt.close(fig)

    # === 图 2：主力通道 RP2_A2 时频图 + 2fm1 频带瞬时能量演化曲线 ===
    f, t_seg, amp, amp_db, _ = stft_res["RP2_A2"]
    m_3k = f <= 3000.0
    m_2fm1 = (f >= 2 * FM1 - 100.0) & (f <= 2 * FM1 + 100.0)

    # 提取 2fm1 频带瞬时幅值（取该频带内最大幅值随时间变化）
    band_peak_t = np.max(amp[m_2fm1], axis=0)
    full_rms_t = np.sqrt(np.mean(amp[m_3k] ** 2, axis=0))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    vmax = float(np.percentile(amp_db[m_3k], 99.8))
    vmin = vmax - 42.0
    pcm = axes[0].pcolormesh(t_seg, f[m_3k], amp_db[m_3k], shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=axes[0], pad=0.015)
    cbar.set_label("幅值 (dB)", fontsize=9)
    axes[0].axvline(TRIM_START_S, color="#00e5ff", linestyle="--", linewidth=1.2, label="0.05 s 稳态分界")
    for k in (1, 2, 3):
        axes[0].axhline(k * FM1, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
        axes[0].text(0.492, k * FM1 + 45, rf"${k}f_{{m1}}={int(k*FM1)}\ \mathrm{{Hz}}$", color="white", fontsize=8.8, ha="right")
    axes[0].set_ylabel("频率 (Hz)", fontsize=9.5)
    axes[0].set_ylim(0, 3000)
    axes[0].set_yticks(np.arange(0, 3001, 500))
    axes[0].set_title("主力通道 RP2_A2（中间轴 Y向）STFT 时频图（0 ~ 3000 Hz）", fontsize=11, fontweight="bold")

    # 下图：2fm1 频带瞬时幅值随时间演化
    axes[1].plot(t_seg, band_peak_t, color="#9467bd", linewidth=1.5, label=r"$2f_{m1}$ 频带 (1740±100 Hz) 瞬时峰值")
    axes[1].axvspan(0.0, TRIM_START_S, color="#ffcccc", alpha=0.45, label="前 0.05 s 启动升速段")
    axes[1].axvline(TRIM_START_S, color="#d62728", linestyle="--", linewidth=1.0)
    # 标注稳态均值线
    st_mask = (t_seg >= 0.06) & (t_seg <= 0.48)
    st_mean = float(np.mean(band_peak_t[st_mask]))
    axes[1].axhline(st_mean, color="#2ca02c", linestyle="-.", linewidth=1.1, label=rf"稳态均值 = {st_mean:.1f} $\mathrm{{m/s^2}}$")
    axes[1].set_xlabel("仿真时间 (s)（未来缺齿将在此每隔 0.109 s 出现一次周期性竖向亮条与峰值突跳）", fontsize=9.5)
    axes[1].set_ylabel(r"瞬时幅值 ($\mathrm{m/s^2}$)", fontsize=9.5)
    axes[1].set_xlim(0.0, 0.50)
    axes[1].set_xticks(np.arange(0.0, 0.501, 0.05))
    axes[1].legend(loc="upper right", fontsize=8.8)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    g2 = RESULT_GRAPH_DIR / "仿真步骤5_图2_主力通道RP2_A2时频与能量演化.png"
    fig.savefig(g2, bbox_inches="tight")
    plt.close(fig)

    print("时频平稳性基线表已保存:", csv_path)
    print(df.to_string(index=False))
    print("\n已生成图表:")
    print(" -", g1)
    print(" -", g2)


if __name__ == "__main__":
    main()
