from pathlib import Path

from evaluate import COMMANDS, build_parser


def test_evaluate_exposes_the_supported_workflows():
    expected = {
        "sonicmaster",
        "prepare",
        "signal",
        "visqol",
        "merge",
        "validate",
        "self-test",
        "kl-ssim",
        "fad",
        "pq",
        "verify-kernels",
    }
    assert set(COMMANDS) == expected
    assert set(build_parser()._actions[-1].choices) == expected


def test_internal_workflow_modules_are_not_standalone_cli_entrypoints():
    root = Path(__file__).resolve().parents[1]
    internal_modules = (
        "src/train_core.py",
        "src/evaluate_sonicmaster.py",
        "evaluation/prepare_protocol.py",
        "evaluation/run_audio_metrics.py",
        "evaluation/run_model_metrics.py",
        "evaluation/verify_metric_kernels.py",
        "evaluation/metric_sources/audiosr/demo_comparison.py",
    )
    for relative_path in internal_modules:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "__main__" not in source, relative_path
