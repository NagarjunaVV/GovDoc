"""VRAM monitoring utilities using NVML as mandated by rule 9.

Writes timestamped GPU memory readings with stage labels to results/raw/vram_trace.jsonl.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

try:
    import pynvml
except ImportError:
    import nvidia_ml_py as pynvml

from govdoc.config import load_config, resolve_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_gpu_memory_info(device_index: int = 0) -> Dict[str, Any]:
    """Return memory info for device index in megabytes (MB) measured with pynvml."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        pynvml.nvmlShutdown()
        return {
            "device_name": name,
            "total_mb": round(info.total / (1024**2), 2),
            "used_mb": round(info.used / (1024**2), 2),
            "free_mb": round(info.free / (1024**2), 2),
        }
    except Exception as e:
        return {
            "device_name": "None",
            "total_mb": 0.0,
            "used_mb": 0.0,
            "free_mb": 0.0,
            "error": str(e),
        }


def get_gpu_memory_used_mb(device_index: int = 0) -> float:
    """Return only used GPU memory in MB."""
    return float(get_gpu_memory_info(device_index).get("used_mb", 0.0))


def get_gpu_peak_memory_mb(device_index: int = 0) -> float:
    """Return peak GPU memory in MB from results/raw/vram_trace.jsonl or torch."""
    trace_path = resolve_path("results/raw/vram_trace.jsonl")
    if trace_path.exists():
        try:
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            readings = [json.loads(l).get("used_mb", 0.0) for l in lines if l.strip()]
            if readings:
                return float(max(readings))
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return float(round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 2))
    except Exception:
        pass
    return float(get_gpu_memory_used_mb(device_index))


def record_vram_reading(
    stage: str,
    device_index: int = 0,
    trace_file: Optional[Path | str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Measure GPU VRAM via NVML and append timestamped entry to results/raw/vram_trace.jsonl.
    """
    mem_info = get_gpu_memory_info(device_index)
    now_iso = datetime.now(timezone.utc).isoformat()

    if trace_file is None:
        cfg = load_config()
        raw_dir = resolve_path(cfg.get("paths", {}).get("results_raw", "results/raw"))
        trace_path = raw_dir / "vram_trace.jsonl"
    else:
        trace_path = resolve_path(trace_file)

    trace_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": now_iso,
        "stage": stage,
        "device_name": mem_info.get("device_name"),
        "used_mb": mem_info.get("used_mb"),
        "total_mb": mem_info.get("total_mb"),
        "free_mb": mem_info.get("free_mb"),
    }
    if extra_metadata:
        record["metadata"] = extra_metadata

    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
