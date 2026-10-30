"""Experiment and run logging utilities for govdoc-rag."""

from datetime import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from govdoc.config import get_config_hash, get_repo_root, resolve_path


def get_git_commit() -> str:
    """Return the current short git commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=get_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def append_log(
    message: str,
    module: str = "GENERAL",
    log_file: Optional[Path | str] = None,
) -> Path:
    """Append a log entry or block to results/LOG.md."""
    if log_file is None:
        target = resolve_path("results/LOG.md")
    else:
        target = resolve_path(log_file)

    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"\n### [{timestamp}] {module}\n{message.strip()}\n"
    with open(target, "a", encoding="utf-8") as f:
        f.write(entry)
    return target


def capture_nvidia_smi(log_file: Optional[Path | str] = None) -> str:
    """Run nvidia-smi and append its output to results/LOG.md."""
    try:
        res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
        smi_out = res.stdout.strip()
    except Exception as e:
        smi_out = f"Failed to capture nvidia-smi: {e}"

    formatted = f"```\n{smi_out}\n```"
    append_log(formatted, module="NVIDIA-SMI CAPTURE", log_file=log_file)
    return smi_out


def save_experiment_record(
    experiment_id: str,
    metrics: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    notes: str = "",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an experiment JSON record conforming to rule 6 in results/raw/."""
    raw_dir = resolve_path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "date": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_hash": get_config_hash(),
        "seed": seed,
        "n_samples": n_samples,
        "metrics": metrics,
        "notes": notes,
    }
    if extra_fields:
        record.update(extra_fields)

    out_file = raw_dir / f"{experiment_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # Also log to LOG.md
    summary_line = f"Experiment `{experiment_id}`: samples={n_samples}, seed={seed}, metrics={metrics}"
    append_log(summary_line, module=f"EXPERIMENT:{experiment_id}")
    return out_file


if __name__ == "__main__":
    print("Capturing nvidia-smi to results/LOG.md...")
    out = capture_nvidia_smi()
    print("NVIDIA-SMI captured successfully.")
