"""
通用振动数据读取模块（面向实测 DT9837B 与 Abaqus 仿真数据）

统一约定：
- 加速度单位统一转换为 m/s^2
- 采样率统一从采集配置文件读取真实硬件采样率 actual_rate_hz
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 项目根目录：毕业设计/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_DATA_DIR = PROJECT_ROOT / "data" / "采集数据" / "健康"
SIM_DATA_DIR = PROJECT_ROOT / "data" / "仿真数据"
RESULT_DATA_DIR = PROJECT_ROOT / "analysis" / "result" / "data"
RESULT_GRAPH_DIR = PROJECT_ROOT / "analysis" / "result" / "graphs"

# 按数据来源分层的输出目录
MEASURED_DATA_DIR = RESULT_DATA_DIR / "实测"
MEASURED_GRAPH_DIR = RESULT_GRAPH_DIR / "实测"
SIM_HEALTHY_DATA_DIR = RESULT_DATA_DIR / "仿真健康"
SIM_HEALTHY_GRAPH_DIR = RESULT_GRAPH_DIR / "仿真健康"
SIM_MISSING_DATA_DIR = RESULT_DATA_DIR / "仿真缺齿"
SIM_MISSING_GRAPH_DIR = RESULT_GRAPH_DIR / "仿真缺齿"

# 9 组健康工况的标准顺序
HEALTH_CASES: List[str] = [
    "第一级-10", "第一级-20", "第一级-30",
    "第二级-10", "第二级-20", "第二级-30",
    "第三级-10", "第三级-20", "第三级-30",
]

# 测点物理含义说明
STAGE_DESC: Dict[str, str] = {
    "第一级": "输入电机侧",
    "第二级": "定轴箱输出级",
    "第三级": "行星级",
}

G_TO_MS2 = 9.80665


def _read_float32_wav(wav_path: Path) -> np.ndarray:
    """读取 IEEE Float32 单声道 WAV 文件中的原始电压数组（单位：V）。"""
    raw = wav_path.read_bytes()
    pos = 12  # 跳过 RIFF header
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        if chunk_id == b"data":
            data_bytes = raw[pos + 8 : pos + 8 + chunk_size]
            return np.frombuffer(data_bytes, dtype="<f4").astype(np.float64)
        pos += 8 + chunk_size + (chunk_size % 2)
    raise ValueError(f"在 {wav_path} 中未找到 data 数据块")


def load_health_case(case_name: str, remove_dc: bool = False) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    读取单个实测健康工况。

    参数：
        case_name: 如 '第一级-30'
        remove_dc: 是否去除直流偏置（减去均值）

    返回：
        t: 时间数组 (s)
        acc: 加速度数组 (m/s^2)
        fs: 实际硬件采样率 (Hz)
        meta: 元信息字典
    """
    case_dir = HEALTH_DATA_DIR / case_name
    run_dirs = sorted(case_dir.glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"未在 {case_dir} 下找到 run_* 目录")
    run_dir = run_dirs[0]

    cfg = json.loads((run_dir / "acquisition_config.json").read_text(encoding="utf-8"))
    fs = float(cfg["actual_rate_hz"])

    # CH2 灵敏度 (mV/g) -> 换算为 (m/s^2)/V
    ch_idx = int(cfg["channels"][0])
    sens_mv_per_g = float(cfg["sensitivities_mV_per_g"][ch_idx])
    v_to_ms2 = G_TO_MS2 / (sens_mv_per_g / 1000.0)

    volts = _read_float32_wav(run_dir / cfg["full_sample_wav"])
    acc = volts * v_to_ms2
    dc_offset = float(np.mean(acc))
    if remove_dc:
        acc = acc - dc_offset

    t = np.arange(len(acc), dtype=np.float64) / fs
    stage, freq_str = case_name.split("-")
    meta = {
        "case_name": case_name,
        "stage": stage,
        "stage_desc": STAGE_DESC.get(stage, ""),
        "input_freq_hz": int(freq_str),
        "requested_rate_hz": float(cfg["requested_rate_hz"]),
        "actual_rate_hz": fs,
        "duration_s": float(t[-1]),
        "num_samples": len(acc),
        "dc_offset_ms2": dc_offset,
        "iepe": bool(cfg.get("iepe", False)),
        "coupling": cfg.get("coupling", ""),
        "sensitivity_mv_per_g": sens_mv_per_g,
        "run_dir": str(run_dir),
    }
    return t, acc, fs, meta


def load_simulation_csv(
    filename: str = "sim_healthy_30Hz_12800Hz.csv",
    column: str = "RP2_A2",
    trim_start_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    读取 Abaqus 导出的空格对齐仿真数据，并自动将 mm/s^2 转换为 m/s^2。

    列对应关系：
        0: Time (s)
        1: RP1_A1 (输入轴 X)
        2: RP2_A1 (中间轴 X)
        3: RP1_A2 (输入轴 Y)
        4: RP2_A2 (中间轴 Y)
    """
    col_map = {
        "RP1_A1": 1,
        "RP2_A1": 2,
        "RP1_A2": 3,
        "RP2_A2": 4,
    }
    if column not in col_map:
        raise ValueError(f"column 必须是 {list(col_map.keys())} 之一")

    csv_path = SIM_DATA_DIR / filename
    data = np.loadtxt(csv_path, skiprows=5)
    t = data[:, 0]
    acc_ms2 = data[:, col_map[column]] / 1000.0  # mm/s^2 -> m/s^2

    dt = np.diff(t)
    fs_est = float(1.0 / np.median(dt))

    if trim_start_s > 0:
        mask = t >= trim_start_s
        t = t[mask]
        acc_ms2 = acc_ms2[mask]

    meta = {
        "filename": filename,
        "column": column,
        "fs_est_hz": fs_est,
        "dt_min_us": float(np.min(dt) * 1e6),
        "dt_max_us": float(np.max(dt) * 1e6),
        "num_samples": len(t),
        "duration_s": float(t[-1] - t[0]),
    }
    return t, acc_ms2, fs_est, meta
