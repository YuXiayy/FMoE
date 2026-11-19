"""Shared Lightning training implementation used by the root train.py launcher."""

import os
import pathlib
import math
import json
import torch

from tqdm import tqdm

import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger

from net.model_factory import build_fmoe

from data.audio_dataset import AudioTrainDataset, audio_collate_fn
from data.data_preflight import (
    format_sonicmaster_preflight,
)
from data.degradation_metadata import SONICMASTER_DEGRADATION_LABELS
from utils.checkpoint import (
    load_compatible_checkpoint,
    resolve_checkpoint_path,
    validate_resume_training_contract,
)
from utils.loss_utils import AudioRestorationLoss
from utils.experiment import write_experiment_artifacts
from utils.routing_analysis import RoutingAnalysisAccumulator
from utils.training_preflight import (
    enforce_training_budget,
    format_training_budget,
    project_training_budget,
)
from utils.training_health import TrainingHealthMonitor
from utils.training_runtime import (
    build_adamw,
    build_training_contract,
    coordinated_data_preflight,
    resolve_distributed_topology,
    resolve_run_id,
    warmup_cosine_multiplier,
)


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.save_hyperparameters({"config": vars(opt)})
        self.opt = opt
        if opt.representation != "complex":
            raise ValueError("F-MoE training requires complex STFT representation.")
        if str(opt.loss_type).lower() != "audio":
            raise ValueError("Only the five-term F-MoE restoration objective is supported.")
        self.use_audio_restoration_loss = True
        self.routing_accumulator = (
            RoutingAnalysisAccumulator(
                num_degradation_classes=opt.num_degradation_classes,
                num_experts=opt.num_exp_blocks,
                degradation_names=SONICMASTER_DEGRADATION_LABELS,
            )
            if opt.enable_routing_logging
            else None
        )
        # Completed epochs are globally merged here on rank zero.  Per-rank
        # current-epoch summaries stay separate until on_train_epoch_end.
        self.routing_history = (
            RoutingAnalysisAccumulator(
                num_degradation_classes=opt.num_degradation_classes,
                num_experts=opt.num_exp_blocks,
                degradation_names=SONICMASTER_DEGRADATION_LABELS,
            )
            if opt.enable_routing_logging
            else None
        )
        self.routing_output_dir = opt.routing_output_dir
        self.routing_flop_profiles = {}

        self.net = build_fmoe(opt)


        self.loss_module = AudioRestorationLoss(
            lambda_complex=opt.lambda_complex,
            lambda_wave=opt.lambda_wave,
            lambda_mrstft=opt.lambda_mrstft,
            lambda_degradation=opt.lambda_degradation,
            lambda_balance=opt.lambda_balance,
            stft_config={
                "n_fft": opt.n_fft,
                "hop_length": opt.hop_length,
                "win_length": opt.win_length,
                "center": opt.stft_center,
            },
            mrstft_config={
                "fft_sizes": opt.mrstft_fft_sizes,
                "hop_sizes": opt.mrstft_hop_sizes,
                "win_lengths": opt.mrstft_win_lengths,
                "spectral_convergence_normalization": (
                    opt.mrstft_sc_normalization
                ),
                "center": opt.stft_center,
            },
            check_finite=opt.loss_check_finite,
        )

    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            degrad_patch = batch["degraded_spec"]
            clean_patch = batch["clean_spec"]
            de_id = batch["degradation_ids"]
        else:
            ([clean_name, de_id], degrad_patch, clean_patch) = batch
        model_outputs = self.net(degrad_patch)
        if isinstance(model_outputs, dict):
            restored = model_outputs["restored_spec"]
            balance_loss = model_outputs.get("balance_loss", self.net.total_loss)
        else:
            restored = model_outputs
            balance_loss = self.net.total_loss

        if self.routing_accumulator is not None:
            assert isinstance(model_outputs, dict)
            routing_stats = model_outputs.get("routing_stats", {})
            if routing_stats:
                self._capture_routing_flop_profiles()
                if isinstance(batch, dict):
                    degradation_targets = batch["degradation_multi_hot"]
                    class_mask = batch.get("degradation_class_mask")
                    if class_mask is not None:
                        degradation_targets = degradation_targets * class_mask
                    label_mask = batch.get("degradation_label_mask")
                    compound_level = batch.get("compound_level")
                else:
                    degradation_targets = torch.zeros(
                        degrad_patch.size(0),
                        self.opt.num_degradation_classes,
                        device=degrad_patch.device,
                    )
                    label_mask = torch.zeros(
                        degrad_patch.size(0), device=degrad_patch.device
                    )
                    compound_level = torch.zeros(
                        degrad_patch.size(0),
                        device=degrad_patch.device,
                        dtype=torch.long,
                    )
                self.routing_accumulator.update(
                    routing_stats,
                    degradation_targets,
                    degradation_label_mask=label_mask,
                    compound_level=compound_level,
                )
                self._log_live_routing(routing_stats, degrad_patch.size(0))

        if self.use_audio_restoration_loss:
            outputs = (
                dict(model_outputs)
                if isinstance(model_outputs, dict)
                else {"restored_spec": restored, "balance_loss": balance_loss}
            )
            losses = self.loss_module(outputs, batch)
            for name, value in losses.items():
                self.log(f"train/{name}", value, sync_dist=True)
                if name in self.loss_module.weights:
                    self.log(
                        f"train/weighted_{name}",
                        self.loss_module.weights[name] * value,
                        sync_dist=True,
                    )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, sync_dist=True)
            return losses["total"]


    def validation_step(self, batch, batch_idx):
        """Evaluate deterministic held-out pairs without updating router EMA."""

        if isinstance(batch, dict):
            degrad_patch = batch["degraded_spec"]
            clean_patch = batch["clean_spec"]
            de_id = batch["degradation_ids"]
        else:
            ([clean_name, de_id], degrad_patch, clean_patch) = batch

        model_outputs = self.net(degrad_patch)
        if isinstance(model_outputs, dict):
            restored = model_outputs["restored_spec"]
            balance_loss = model_outputs.get("balance_loss", self.net.total_loss)
        else:
            restored = model_outputs
            balance_loss = self.net.total_loss

        batch_size = int(degrad_patch.size(0))
        if self.use_audio_restoration_loss:
            outputs = (
                dict(model_outputs)
                if isinstance(model_outputs, dict)
                else {"restored_spec": restored, "balance_loss": balance_loss}
            )
            losses = self.loss_module(outputs, batch)
            restoration = losses["total"].new_zeros(())
            for name in ("complex", "wave", "mrstft"):
                restoration = restoration + self.loss_module.weights[name] * losses[name]
            self.log(
                "val/restoration",
                restoration,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            for name, value in losses.items():
                self.log(
                    f"val/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                if name in self.loss_module.weights:
                    self.log(
                        f"val/weighted_{name}",
                        self.loss_module.weights[name] * value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
            return restoration


    @staticmethod
    def _distributed_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("RANK", "0"))

    def _capture_routing_flop_profiles(self):
        if self._distributed_rank() != 0:
            return
        for stage_index, stage in enumerate(self.net.dec):
            group = stage[2]
            if not group.layers or group.layers[0].adapter is None:
                continue
            metadata = group.layers[0].adapter.flop_profile_metadata
            if metadata:
                self.routing_flop_profiles[f"decoder_stage_{stage_index}"] = dict(
                    metadata
                )

    def _log_live_routing(self, routing_stats, batch_size: int):
        """Expose pilot-time collapse signals without retaining graphs."""

        for stage_name, statistics in routing_stats.items():
            probabilities = statistics.get("gates")
            topk_indices = statistics.get("topk_indices")
            if not isinstance(probabilities, torch.Tensor):
                continue
            probabilities = probabilities.float().clamp_min(1.0e-12)
            num_experts = probabilities.size(-1)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
            if num_experts > 1:
                entropy = entropy / math.log(num_experts)
            else:
                entropy = entropy * 0.0
            self.log(
                f"train/routing/{stage_name}/normalized_entropy",
                entropy,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=int(batch_size),
            )
            self.log(
                f"train/routing/{stage_name}/max_probability",
                probabilities.max(dim=-1).values.mean(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=int(batch_size),
            )
            if isinstance(topk_indices, torch.Tensor):
                fractions = F.one_hot(
                    topk_indices,
                    num_classes=num_experts,
                ).float().mean(dim=tuple(range(topk_indices.ndim)))
                for expert_index, fraction in enumerate(fractions):
                    self.log(
                        f"train/routing/{stage_name}/expert_{expert_index}_fraction",
                        fraction,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                        batch_size=int(batch_size),
                    )

    def _gather_current_routing_epoch(self):
        if self.routing_accumulator is None:
            return None
        local_state = self.routing_accumulator.state_dict()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local_state)
            if self._distributed_rank() != 0:
                return None
            merged = RoutingAnalysisAccumulator.from_state_dict(gathered[0])
            for state in gathered[1:]:
                merged.merge_(RoutingAnalysisAccumulator.from_state_dict(state))
            return merged
        return RoutingAnalysisAccumulator.from_state_dict(local_state)

    def on_train_epoch_end(self):
        """Globally merge fixed-size statistics, then release epoch state."""

        if self.routing_accumulator is None:
            return
        merged_epoch = self._gather_current_routing_epoch()
        if self._distributed_rank() == 0:
            assert self.routing_history is not None and merged_epoch is not None
            self.routing_history.merge_(merged_epoch)
        self.routing_accumulator.reset()

    def on_save_checkpoint(self, checkpoint):
        if self.routing_history is not None and self._distributed_rank() == 0:
            checkpoint["routing_analysis_state"] = self.routing_history.state_dict()
            checkpoint["routing_flop_profiles"] = self.routing_flop_profiles

    def on_load_checkpoint(self, checkpoint):
        if self.routing_history is None or self._distributed_rank() != 0:
            return
        state = checkpoint.get("routing_analysis_state")
        if state is not None:
            self.routing_history.load_state_dict(state)
        profiles = checkpoint.get("routing_flop_profiles")
        if isinstance(profiles, dict):
            self.routing_flop_profiles = profiles

    def on_train_end(self):
        if self.routing_accumulator is not None and self.routing_accumulator.stage_names:
            merged_epoch = self._gather_current_routing_epoch()
            if self._distributed_rank() == 0:
                assert self.routing_history is not None and merged_epoch is not None
                self.routing_history.merge_(merged_epoch)
            self.routing_accumulator.reset()
        if self.routing_history is None or self._distributed_rank() != 0:
            return
        if not self.routing_output_dir:
            self.routing_output_dir = os.path.join(
                str(self.trainer.default_root_dir), "routing_analysis"
            )
        paths = self.routing_history.export_csv(self.routing_output_dir)
        summary_path = self.routing_history.export_json(self.routing_output_dir)
        profile_path = pathlib.Path(self.routing_output_dir) / "flop_profiles.json"
        with profile_path.open("w", encoding="utf-8") as output:
            json.dump(
                self.routing_flop_profiles,
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
        print(
            "[Routing] exported {}, {}, and {}".format(
                ", ".join(str(path) for path in paths.values()),
                summary_path,
                profile_path,
            )
        )

    def configure_optimizers(self):
        optimizer = build_adamw(self.parameters(), self.opt)

        # Lightning computes effective step count using dataloader size, devices,
        # gradient accumulation, and max_epochs.
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if total_steps < 1:
            total_steps = max(1, int(self.opt.epochs))

        def lr_lambda(current_step: int) -> float:
            return warmup_cosine_multiplier(
                current_step,
                total_steps=total_steps,
                warmup_ratio=self.opt.lr_warmup_ratio,
                min_ratio=self.opt.lr_min_ratio,
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }



def deterministic_subset_indices(dataset_size: int, max_samples: int):
    """Pick one centered item from each equal-width interval."""

    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if max_samples <= 0 or max_samples >= dataset_size:
        return list(range(dataset_size))
    return [
        min(dataset_size - 1, int((index + 0.5) * dataset_size / max_samples))
        for index in range(max_samples)
    ]


def build_checkpoint_callbacks(opt, checkpoint_path):
    """Bound checkpoint storage while keeping best and resumable states."""

    callbacks = []
    if opt.validation_enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="best-step={step}-val_restoration={val/restoration:.6f}",
                monitor="val/restoration",
                mode="min",
                save_top_k=opt.checkpoint_save_top_k,
                save_last=False,
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
            )
        )
    if opt.checkpoint_every_n_train_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="latest-step={step}",
                monitor=None,
                save_top_k=1,
                save_last="link",
                save_on_exception=True,
                every_n_train_steps=opt.checkpoint_every_n_train_steps,
                every_n_epochs=None,
                auto_insert_metric_name=False,
            )
        )
    if not callbacks:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="last-epoch={epoch}",
                monitor=None,
                save_top_k=1,
                save_last="link",
                save_on_exception=True,
                every_n_epochs=1,
                auto_insert_metric_name=False,
            )
        )
    return callbacks


def main(opt):
    pl.seed_everything(opt.seed, workers=True)
    launch_rank = int(os.environ.get("RANK", "0"))
    run_id = resolve_run_id(opt)
    opt.run_id = run_id
    print("Options")
    print(opt)

    log_dir = os.path.join("logs/", run_id)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(opt.ckpt_dir, run_id)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Fail closed on the complete metadata/audio inventory before allocating
    # the full model or constructing worker processes.
    data_preflight_report = None
    if opt.data_preflight_enabled:
        data_preflight_report = coordinated_data_preflight(
            opt,
            log_dir=log_dir,
            checkpoint_path=checkpoint_path,
            rank=launch_rank,
        )
        print(format_sonicmaster_preflight(data_preflight_report))

    if opt.wblogger:
        name = opt.model + "_" + run_id
        logger  = WandbLogger(name=name, save_dir=log_dir, config=opt)

    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    model = PLTrainModel(opt)
    warm_start_report = None
    if opt.fine_tune_from:
        warm_start_path = resolve_checkpoint_path(
            ckpt_dir=opt.ckpt_dir,
            reference=opt.fine_tune_from,
            selection="latest",
        )
        warm_start_report = load_compatible_checkpoint(model, warm_start_path)
        print(
            "[Warm-start] loaded {}/{} tensors from {}".format(
                warm_start_report["loaded_tensor_count"],
                warm_start_report["source_tensor_count"],
                warm_start_path,
            )
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[Model] total_parameters={total_parameters:,}, "
        f"trainable_parameters={trainable_parameters:,}"
    )
    if opt.enable_routing_logging and not model.routing_output_dir:
        model.routing_output_dir = os.path.join(
            checkpoint_path, "routing_analysis"
        )
    # Create datasets and dataloaders
    trainset = AudioTrainDataset(opt, split="train")

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=audio_collate_fn,
    )

    valset_full = None
    valset = None
    valloader = None
    if opt.validation_enabled:
        valset_full = AudioTrainDataset(opt, split=opt.validation_split)
        validation_indices = deterministic_subset_indices(
            len(valset_full), opt.validation_max_samples
        )
        valset = (
            valset_full
            if len(validation_indices) == len(valset_full)
            else Subset(valset_full, validation_indices)
        )
        valloader = DataLoader(
            valset,
            batch_size=opt.validation_batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=opt.validation_num_workers,
            collate_fn=audio_collate_fn,
        )

    # Resolve local devices independently from the global process count.
    visible_gpus = torch.cuda.device_count()
    topology = resolve_distributed_topology(
        requested_gpus=opt.num_gpus,
        visible_gpus=visible_gpus,
    )
    effective_gpus = topology["devices"]
    if not topology["external"] and opt.num_gpus > visible_gpus:
        print(
            f"[Warning] Requested num_gpus={opt.num_gpus}, but only "
            f"{visible_gpus} visible. Falling back to {visible_gpus}."
        )
    strategy = (
        "ddp_find_unused_parameters_true"
        if topology["world_size"] > 1
        else "auto"
    )
    print(f"[Distributed topology] {topology}")

    training_budget = project_training_budget(
        dataset_size=len(trainset),
        micro_batch_size_per_device=opt.batch_size,
        accumulate_grad_batches=opt.accum_grad,
        max_epochs=opt.epochs,
        max_steps=opt.max_steps,
        world_size=topology["world_size"],
        drop_last=True,
    )
    enforce_training_budget(
        training_budget,
        max_steps_without_override=opt.max_training_steps_without_override,
        allow_large_training_budget=opt.allow_large_training_budget,
    )
    print(format_training_budget(training_budget))

    training_contract = build_training_contract(
        opt,
        training_budget,
        data_preflight_report=data_preflight_report,
    )
    model.hparams["training_contract"] = training_contract
    resume_checkpoint_path = None
    resume_contract_report = None
    if opt.resume_from:
        resume_checkpoint_path = resolve_checkpoint_path(
            ckpt_dir=opt.ckpt_dir,
            reference=opt.resume_from,
            selection="latest",
        )
        resume_contract_report = validate_resume_training_contract(
            resume_checkpoint_path,
            training_contract,
        )

    run_config = vars(opt).copy()
    run_config["run_id"] = run_id
    run_config["distributed_topology"] = topology
    run_config["training_budget"] = training_budget.as_dict()
    run_config["training_contract"] = training_contract
    if valset is not None and valset_full is not None:
        run_config["validation_dataset_size_full"] = len(valset_full)
        run_config["validation_dataset_size_selected"] = len(valset)
    if data_preflight_report is not None:
        run_config["data_preflight"] = data_preflight_report.as_dict()
    if opt.enable_routing_logging:
        run_config["routing_output_dir_resolved"] = model.routing_output_dir
    if warm_start_report is not None:
        run_config["checkpoint_migration"] = warm_start_report
    if resume_contract_report is not None:
        run_config["resume_contract_validation"] = resume_contract_report
    if model.use_audio_restoration_loss:
        run_config["loss_formula"] = (
            "lambda_complex*complex + lambda_wave*wave + lambda_mrstft*mrstft + "
            "lambda_degradation*degradation + lambda_balance*balance"
        )
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if launch_rank == 0:
        run_config = write_experiment_artifacts(
            (log_dir, checkpoint_path),
            run_config,
            model,
            repo_root,
        )
        logger.log_hyperparams(run_config)
    checkpoint_callbacks = build_checkpoint_callbacks(opt, checkpoint_path)
    if opt.enable_training_health_monitor:
        checkpoint_callbacks.append(
            TrainingHealthMonitor(
                os.path.join(checkpoint_path, "training_health.json"),
                log_interval_steps=opt.health_log_interval_steps,
                window_size=opt.health_timing_window_size,
            )
        )

    trainer = pl.Trainer(max_epochs=opt.epochs,
                         max_steps=opt.max_steps,
                         accelerator="gpu",
                         devices=effective_gpus,
                         num_nodes=topology["num_nodes"],
                         strategy=strategy,
                         precision=opt.precision,
                         gradient_clip_val=opt.gradient_clip_val,
                         gradient_clip_algorithm="norm",
                         logger=logger,
                         callbacks=checkpoint_callbacks,
                         accumulate_grad_batches=opt.accum_grad,
                         val_check_interval=(
                             opt.validation_interval_batches
                             if opt.validation_enabled
                             else None
                         ),
                         check_val_every_n_epoch=(
                             None if opt.validation_enabled else 1
                         ),
                         num_sanity_val_steps=(
                             opt.validation_sanity_steps
                             if opt.validation_enabled
                             else 0
                         ),
                         deterministic=True)

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=resume_checkpoint_path
    )



