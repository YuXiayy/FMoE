"""Bounded, crash-safe runtime telemetry for training jobs."""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections.abc import Mapping
from collections import deque
from pathlib import Path
from statistics import fmean
from typing import Deque, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback


def percentile(values, quantile: float) -> Optional[float]:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scalar_loss(outputs) -> Optional[float]:
    """Extract Lightning's scalar training loss without assuming one output shape."""

    if isinstance(outputs, Mapping):
        outputs = outputs.get("loss")
    if isinstance(outputs, torch.Tensor):
        if outputs.numel() != 1:
            return None
        return float(outputs.detach().item())
    if isinstance(outputs, (int, float)) and not isinstance(outputs, bool):
        return float(outputs)
    return None


class TrainingHealthMonitor(Callback):
    """Log fixed-memory telemetry and atomically refresh one JSON report."""

    def __init__(self, report_path: str, *, log_interval_steps: int, window_size: int):
        super().__init__()
        if log_interval_steps < 1:
            raise ValueError("log_interval_steps must be positive")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.report_path = str(report_path)
        self.log_interval_steps = int(log_interval_steps)
        self.batch_durations: Deque[float] = deque(maxlen=int(window_size))
        self.started_at: Optional[float] = None
        self.batch_started_at: Optional[float] = None
        self.last_logged_step = -1
        self.latest_loss: Optional[float] = None
        self.latest_loss_is_finite: Optional[bool] = None
        self.latest_loss_step: Optional[int] = None
        self.write_failures = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self.started_at = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self.batch_started_at = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.batch_started_at is not None:
            self.batch_durations.append(time.perf_counter() - self.batch_started_at)
        step = int(trainer.global_step)
        if (
            not trainer.is_global_zero
            or step == self.last_logged_step
            or step % self.log_interval_steps != 0
        ):
            return
        self.last_logged_step = step
        if self.batch_durations:
            pl_module.log(
                "system/step_seconds_mean",
                fmean(self.batch_durations),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            pl_module.log(
                "system/step_seconds_p95",
                percentile(self.batch_durations, 0.95),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        if torch.cuda.is_available():
            gib = float(1024**3)
            pl_module.log(
                "system/cuda_peak_allocated_gib",
                torch.cuda.max_memory_allocated() / gib,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            pl_module.log(
                "system/cuda_peak_reserved_gib",
                torch.cuda.max_memory_reserved() / gib,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        loss = _scalar_loss(outputs)
        if loss is not None:
            self.latest_loss_is_finite = math.isfinite(loss)
            self.latest_loss_step = step
            # Strict JSON deliberately cannot encode NaN/Inf. Preserve the
            # finite-state signal while keeping the report machine-readable.
            self.latest_loss = loss if self.latest_loss_is_finite else None
        self._write(trainer, status="running")

    def _report(self, trainer, *, status: str, exception=None) -> Dict[str, object]:
        durations = list(self.batch_durations)
        wall_time = (
            time.perf_counter() - self.started_at
            if self.started_at is not None
            else None
        )
        report: Dict[str, object] = {
            "status": status,
            "global_step": int(trainer.global_step),
            "wall_time_seconds": wall_time,
            "timing_window_count": len(durations),
            "step_seconds_mean": fmean(durations) if durations else None,
            "step_seconds_p50": percentile(durations, 0.50),
            "step_seconds_p95": percentile(durations, 0.95),
            "latest_loss": self.latest_loss,
            "latest_loss_is_finite": self.latest_loss_is_finite,
            "latest_loss_step": self.latest_loss_step,
            "write_failures": self.write_failures,
            "exception": repr(exception) if exception is not None else None,
        }
        if torch.cuda.is_available():
            gib = float(1024**3)
            report.update(
                {
                    "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                    "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
                }
            )
        return report

    def _write(self, trainer, *, status: str, exception=None) -> None:
        if not trainer.is_global_zero:
            return
        path = Path(self.report_path)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(
                    self._report(trainer, status=status, exception=exception),
                    output,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        except OSError as error:
            self.write_failures += 1
            # Telemetry must not terminate an otherwise valid long-running
            # training job because a shared filesystem had a transient error.
            warnings.warn(
                f"Could not write training health report {path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def on_train_end(self, trainer, pl_module) -> None:
        self._write(trainer, status="complete")

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._write(trainer, status="exception", exception=exception)


__all__ = ["TrainingHealthMonitor", "percentile"]
