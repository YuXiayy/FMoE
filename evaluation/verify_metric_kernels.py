#!/usr/bin/env python3
"""Numerically verify the GPU ports against untouched paper-repository code."""

import ast
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
AUDIOSR = ROOT / "metric_sources/audiosr/demo_comparison.py"
NUSSL = ROOT / "metric_sources/nussl/bss_eval.py"


def load_functions(path: Path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[name] for name in names]


def main():
    rng = np.random.default_rng(20260724)

    unify, official_lsd = load_functions(
        AUDIOSR, ["unify_lower_frequency_energy", "lsd"], {"np": np, "torch": torch, "EPS": 1e-8}
    )
    ref_mag = np.abs(rng.normal(size=(2, 129, 57))).astype(np.float32) + 1e-3
    pred_mag = np.abs(rng.normal(size=(2, 129, 57))).astype(np.float32) + 1e-3
    official_channels = []
    for ch in range(2):
        scaled, _ = unify(ref_mag[ch], pred_mag[ch], 10)
        official_channels.append(float(official_lsd(ref_mag[ch], scaled).mean()))
    official_lsd_value = float(np.mean(official_channels))

    ref_t = torch.from_numpy(ref_mag)
    pred_t = torch.from_numpy(pred_mag)
    lower_bins = int(ref_t.shape[-2] * 10 / 100)
    scale = ref_t[..., :lower_bins, :].mean((-2, -1)) / pred_t[..., :lower_bins, :].mean((-2, -1))
    pred_t = pred_t * scale[:, None, None]
    port_lsd = torch.log10(pred_t.square() / (ref_t + 1e-8).square() + 1e-8).square()
    port_lsd_value = float(port_lsd.mean(-2).sqrt().mean())

    _, official_scale_bss_eval = load_functions(
        NUSSL, ["_scale_bss_eval", "scale_bss_eval"], {"np": np}
    )
    ref = rng.normal(size=16000)
    estimate = ref + 0.3 * rng.normal(size=16000)
    mixture = ref + 0.7 * rng.normal(size=16000)
    ref -= ref.mean(); estimate -= estimate.mean(); mixture -= mixture.mean()
    official = official_scale_bss_eval(ref[:, None], estimate, mixture, 0, compute_sir_sar=False)

    def port_si_sdr(est, target):
        alpha = np.sum(target * est) / np.sum(target ** 2)
        projected = target * alpha
        return 10 * np.log10(np.sum(projected ** 2) / np.sum((est - projected) ** 2))

    port_si = port_si_sdr(estimate, ref)
    port_sdri = port_si - port_si_sdr(mixture, ref)
    report = {
        "audiosr_source": str(AUDIOSR.relative_to(ROOT)).replace("\\", "/"),
        "audiosr_lsd_official": official_lsd_value,
        "audiosr_lsd_gpu_port": port_lsd_value,
        "audiosr_abs_error": abs(official_lsd_value - port_lsd_value),
        "nussl_source": str(NUSSL.relative_to(ROOT)).replace("\\", "/"),
        "nussl_si_sdr_official": float(official[0]),
        "nussl_si_sdr_port": float(port_si),
        "nussl_si_sdri_official": float(official[6]),
        "nussl_si_sdri_port": float(port_sdri),
        "nussl_si_sdri_abs_error": abs(float(official[6]) - float(port_sdri)),
    }
    assert report["audiosr_abs_error"] < 1e-6, report
    assert report["nussl_si_sdri_abs_error"] < 1e-10, report
    out = ROOT / "metric_kernel_equivalence.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


