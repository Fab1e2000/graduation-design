"""
步骤 1：数据体检与时域波形可视化
产出：
  1. analysis/result/data/步骤1_健康实测数据体检表.csv
  2. analysis/result/graphs/步骤1_图1_去直流偏置前后对比.png
  3. analysis/result/graphs/步骤1_图2_9组健康实测全景时域波形.png
  4. analysis/result/graphs/步骤1_图3_30Hz局部波形与健康仿真对照.png
"""

from __future__ import annotations

import os
from pathlib import Path

# 避免 macOS 沙箱或权限下 matplotlib 缓存警告
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from 数据读取模块 import (
    HEALTH_CASES,
    RESULT_DATA_DIR,
    RESULT_GRAPH_DIR,
    STAGE_DESC,
    load_health_case,
    load_simulation_csv,
)

# 统一中文字体与科研绘图风格
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def segment_stats(acc_zero: np.ndarray, fs: float, win_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """按每秒切分，返回每秒的 RMS 和峭度（Fisher=False，正态分布为3）。"""
    step = int(round(fs * win_s))
    n_seg = len(acc_zero) // step
    rms_list, kurt_list = [], []
    for i in range(n_seg):
        seg = acc_zero[i * step : (i + 1) * step]
        rms_list.append(float(np.sqrt(np.mean(seg ** 2))))
        kurt_list.append(float(kurtosis(seg, fisher=False)))
    return np.array(rms_list), np.array(kurt_list)


def plot_dc_removal(out_path: Path) -> None:
    """图1：用第一级-30直观演示什么是直流偏置、为什么必须去均值。"""
    t, acc_raw, fs, meta = load_health_case("第一级-30", remove_dc=False)
    dc = meta["dc_offset_ms2"]
    acc_zero = acc_raw - dc

    # 取中间稳态 0.1 s（第 15.0 ~ 15.1 s）方便肉眼看清波形
    mask = (t >= 15.0) & (t <= 15.1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    axes[0].plot(t[mask], acc_raw[mask], color="#d62728", linewidth=0.9, label="原始加速度 CH2_m_s2")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label=r"零线 (0 $\mathrm{m/s^2}$)")
    axes[0].axhline(dc, color="#1f77b4", linestyle="-.", linewidth=1.2, label=rf"直流偏置均值 = {dc:.4f} $\mathrm{{m/s^2}}$")
    axes[0].set_title("预处理前：第一级-30（15.0 ~ 15.1 s 局部放大）——波形整体偏在零线下方", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"加速度 ($\mathrm{m/s^2}$)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[mask], acc_zero[mask], color="#1f77b4", linewidth=0.9, label="去均值后加速度")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label=r"零线 (0 $\mathrm{m/s^2}$)")
    axes[1].set_title("预处理后：减去均值后，波形对称围绕 0 摆动（物理真实振动）", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel(r"加速度 ($\mathrm{m/s^2}$)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all9_overview(out_path: Path) -> pd.DataFrame:
    """图2：9组健康实测整段时域波形全景（3x3），并返回体检指标表。"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharey=True)
    rows = []

    stage_colors = {"第一级": "#1f77b4", "第二级": "#2ca02c", "第三级": "#ff7f0e"}

    for idx, case_name in enumerate(HEALTH_CASES):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        t, acc, fs, meta = load_health_case(case_name, remove_dc=True)

        rms = float(np.sqrt(np.mean(acc ** 2)))
        peak = float(np.max(np.abs(acc)))
        kurt_all = float(kurtosis(acc, fisher=False))
        seg_rms, seg_kurt = segment_stats(acc, fs, win_s=1.0)
        max_k_sec = int(np.argmax(seg_kurt))

        rows.append({
            "工况": case_name,
            "测点位置": meta["stage_desc"],
            "输入频率_Hz": meta["input_freq_hz"],
            "实际采样率_Hz": round(fs, 2),
            "时长_s": round(meta["duration_s"], 2),
            "直流偏置_m_s2": round(meta["dc_offset_ms2"], 4),
            "整段RMS_m_s2": round(rms, 4),
            "整段峰值_m_s2": round(peak, 4),
            "整段峭度": round(kurt_all, 2),
            "每秒峭度中位数": round(float(np.median(seg_kurt)), 2),
            "每秒峭度最大值": round(float(np.max(seg_kurt)), 2),
            "最大峭度出现秒数": max_k_sec,
            "每秒RMS波动率_%": round(float(np.std(seg_rms) / np.mean(seg_rms) * 100), 2),
        })

        # 绘图降采样：每 20 点取 min/max 包络或直接稀疏绘制以保真又快速
        # 这里按每 40 点画极值，避免漏掉任何尖峰毛刺
        bin_sz = 40
        n_bins = len(acc) // bin_sz
        t_bin = t[: n_bins * bin_sz].reshape(n_bins, bin_sz).mean(axis=1)
        a_min = acc[: n_bins * bin_sz].reshape(n_bins, bin_sz).min(axis=1)
        a_max = acc[: n_bins * bin_sz].reshape(n_bins, bin_sz).max(axis=1)

        color = stage_colors[meta["stage"]]
        ax.fill_between(t_bin, a_min, a_max, color=color, alpha=0.85, linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(
            f"{case_name}（{meta['stage_desc']}）\n"
            f"RMS={rms:.3f} | 峰值={peak:.3f} | 整段峭度={kurt_all:.1f} (中位={np.median(seg_kurt):.1f})",
            fontsize=9.5,
        )
        ax.set_ylim(-0.85, 0.85)
        ax.grid(True, alpha=0.25)
        if r == 2:
            ax.set_xlabel("时间 (s)", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"去均值加速度 ($\mathrm{m/s^2}$)", fontsize=9)

    fig.suptitle(r"9 组实测健康数据整段时域波形全景（已去均值，统一纵轴 $\pm 0.85\ \mathrm{m/s^2}$）", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_zoom_and_sim(out_path: Path) -> None:
    """图3：30 Hz 工况下三测点局部 0.2 s（同横纵轴分度）+ 健康仿真稳态 0.2 s（同横轴分度）直观对比。"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)

    cases_30 = [
        ("第一级-30", "#1f77b4", "实测 · 第一级-30（输入电机侧，15.0~15.2 s）"),
        ("第二级-30", "#2ca02c", "实测 · 第二级-30（定轴箱输出级，15.0~15.2 s）"),
        ("第三级-30", "#ff7f0e", "实测 · 第三级-30（行星级，15.0~15.2 s）"),
    ]

    for i, (cname, color, title) in enumerate(cases_30):
        t, acc, fs, _ = load_health_case(cname, remove_dc=True)
        mask = (t >= 15.0) & (t <= 15.2)
        axes[i].plot(t[mask] - 15.0, acc[mask], color=color, linewidth=0.8)
        axes[i].set_title(title, fontsize=10.5, fontweight="bold")
        axes[i].set_ylabel(r"加速度 ($\mathrm{m/s^2}$)", fontsize=9)
        axes[i].set_ylim(-0.55, 0.55)
        axes[i].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        axes[i].grid(True, alpha=0.3)

    # 第4行：健康仿真数据 RP2_A2，取稳态段 0.10 ~ 0.30 s（与上面实测完全相同的 0.2 s 时间分度）
    t_sim, acc_sim, fs_sim, meta_sim = load_simulation_csv(column="RP2_A2", trim_start_s=0.0)
    sim_mask = (t_sim >= 0.10) & (t_sim <= 0.30)
    t_sim_win = t_sim[sim_mask] - 0.10
    acc_sim_win = acc_sim[sim_mask]

    axes[3].plot(t_sim_win, acc_sim_win, color="#9467bd", linewidth=0.75, label="仿真 RP2_A2（中间轴Y向，稳态段 0.10~0.30 s）")
    axes[3].set_title(
        r"仿真 · Abaqus 健康基准（30 Hz，RP2_A2，稳态 0.2 s 窗口）——与上方实测同时间分度，纵轴量级不同",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[3].set_xlabel("相对时间 (s)（四张图统一 0 ~ 0.2 s 窗口，输入轴转 6 圈）", fontsize=9.5)
    axes[3].set_ylabel(r"加速度 ($\mathrm{m/s^2}$)", fontsize=9)
    axes[3].set_xlim(0.0, 0.2)
    axes[3].set_xticks(np.arange(0.0, 0.201, 0.025))
    axes[3].set_ylim(-1200, 1200)
    axes[3].set_yticks([-1000, -500, 0, 500, 1000])
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    g1 = RESULT_GRAPH_DIR / "步骤1_图1_去直流偏置前后对比.png"
    g2 = RESULT_GRAPH_DIR / "步骤1_图2_9组健康实测全景时域波形.png"
    g3 = RESULT_GRAPH_DIR / "步骤1_图3_30Hz局部波形与健康仿真对照.png"
    csv_out = RESULT_DATA_DIR / "步骤1_健康实测数据体检表.csv"

    plot_dc_removal(g1)
    df = plot_all9_overview(g2)
    plot_zoom_and_sim(g3)
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("Step 1 体检表已保存:", csv_out)
    print(df.to_string(index=False))
    print("\n已生成 3 张图表:")
    for g in (g1, g2, g3):
        print(" -", g)


if __name__ == "__main__":
    main()
