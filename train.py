"""Train F-MoE with the five-term objective described in the paper."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from options import train_options


def _checkpoint_callbacks(opt, checkpoint_path):
    from lightning.pytorch.callbacks import ModelCheckpoint

    callbacks = []
    if opt.validation_enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename=(
                    "best-epoch={epoch:03d}-step={step}-"
                    "val_restoration={val/restoration:.6f}"
                ),
                monitor="val/restoration",
                mode="min",
                save_top_k=opt.checkpoint_save_top_k,
                save_last=False,
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
            )
        )
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="latest-epoch={epoch:03d}-step={step}",
            monitor=None,
            save_top_k=1,
            save_last="link",
            save_on_exception=True,
            every_n_train_steps=opt.checkpoint_every_n_train_steps,
            every_n_epochs=None,
            auto_insert_metric_name=False,
        )
    )
    return callbacks


def main() -> None:
    opt = train_options()
    if opt.fine_tune_from is not None:
        raise RuntimeError("fine_tune_from is not supported by the released recipe")
    if float(opt.segment_seconds) != 5.0:
        raise RuntimeError("the released F-MoE recipe uses 5-second windows")

    is_resume = opt.resume_from is not None
    run_dir = (Path(opt.ckpt_dir) / opt.run_id).resolve()
    if is_resume:
        reference = Path(str(opt.resume_from))
        if str(opt.resume_from) != opt.run_id:
            resolved_reference = reference.resolve()
            if run_dir not in resolved_reference.parents and resolved_reference != run_dir:
                raise RuntimeError(
                    f"resume checkpoint must belong to this run: {resolved_reference}"
                )
        evidence_path = run_dir / "SCRATCH_INIT_EVIDENCE.json"
        if not evidence_path.is_file():
            raise RuntimeError("resume refused: missing scratch-origin evidence")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("init_from_scratch") is not True:
            raise RuntimeError("resume refused: run did not originate from scratch")

    opt.init_from_scratch = True
    opt.loaded_checkpoint = str(opt.resume_from) if is_resume else None
    opt.nonoverlap_segments = True
    opt.segment_tail_policy = "pad"
    opt.window_stride_seconds = float(opt.segment_seconds)
    opt.activation_checkpointing = True
    opt.activation_checkpointing_scope = (
        "pure_attention_crossattention_ffn_band_expert_rope_attention_and_audio_loss"
    )
    opt.scratch_seed = int(opt.seed)
    opt.initial_resume_from = None
    opt.initial_fine_tune_from = None
    opt.continuation_resume = bool(is_resume)

    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "init_from_scratch": True,
        "seed": int(opt.seed),
        "loaded_checkpoint": None if not is_resume else str(opt.resume_from),
        "resume_from": None if not is_resume else str(opt.resume_from),
        "fine_tune_from": None,
        "continuation_resume": bool(is_resume),
        "pid": os.getpid(),
    }
    evidence_name = (
        "SCRATCH_INIT_EVIDENCE.json"
        if not is_resume
        else f"RESUME_EVIDENCE_{os.getpid()}.json"
    )
    (run_dir / evidence_name).write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[Scratch evidence] " + json.dumps(evidence, sort_keys=True))

    import train_core
    from data.segment_dataset import NonOverlappingSegmentDataset
    from utils.activation_checkpointing import (
        checkpoint_pure_forward,
        enable_pure_activation_checkpointing,
    )

    class SegmentTrainModel(train_core.PLTrainModel):
        """Lightning module with activation checkpointing for 5-second training."""

        def __init__(self, model_opt):
            super().__init__(model_opt)
            checkpointed = enable_pure_activation_checkpointing(self.net)
            checkpoint_pure_forward(self.loss_module)
            print(f"[Activation checkpointing] pure_modules={checkpointed}")

    train_core.AudioTrainDataset = NonOverlappingSegmentDataset
    train_core.PLTrainModel = SegmentTrainModel
    train_core.build_checkpoint_callbacks = _checkpoint_callbacks
    train_core.main(opt)


if __name__ == "__main__":
    main()
