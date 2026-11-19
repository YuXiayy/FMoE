import argparse
from typing import Any, Dict, Optional, Sequence

import yaml

from data.degradation_metadata import SONICMASTER_DEGRADATION_LABELS

# Helpers
def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def base_parser():
    parser = argparse.ArgumentParser()

    # Basic training settings
    parser.add_argument('--config', type=str, default=None, help='Optional flat YAML configuration file. Explicit CLI values take precedence.')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Stable run identifier shared by all distributed ranks; generated for single-process launches.')
    parser.add_argument('--distributed_preflight_timeout_seconds', type=float, default=900.0,
                        help='Maximum wait for rank-zero data preflight in an external distributed launch.')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed used by Lightning and dataloader workers.')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs.')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Hard optimizer-step limit; -1 uses the epoch-derived limit.')
    parser.add_argument('--max_training_steps_without_override', type=int, default=1_000_000,
                        help='Fail-fast threshold for accidental unbounded training runs.')
    parser.add_argument('--allow_large_training_budget', type=str2bool, default=False,
                        help='Explicit acknowledgement required above the fail-fast step threshold.')
    parser.add_argument('--validation_enabled', type=str2bool, default=False,
                        help='Run a deterministic held-out validation loop during training.')
    parser.add_argument('--validation_split', type=str, default='val',
                        help='Exact SonicMaster metadata split used for validation.')
    parser.add_argument('--validation_batch_size', type=int, default=1)
    parser.add_argument('--validation_num_workers', type=int, default=4)
    parser.add_argument('--validation_max_samples', type=int, default=0,
                        help='Deterministic validation subset size; 0 uses the full split.')
    parser.add_argument('--validation_interval_batches', type=int, default=1000,
                        help='Run validation after this many training micro-batches.')
    parser.add_argument('--validation_sanity_steps', type=int, default=1)
    parser.add_argument('--checkpoint_save_top_k', type=int, default=3,
                        help='Number of best validation checkpoints to retain.')
    parser.add_argument('--checkpoint_every_n_train_steps', type=int, default=5000,
                        help='Rolling resumable checkpoint interval; 0 disables it.')
    parser.add_argument('--data_preflight_enabled', type=str2bool, default=False,
                        help='Audit the complete SonicMaster metadata/audio inventory before training.')
    parser.add_argument('--data_preflight_required_splits', nargs='+',
                        default=['train', 'val', 'test'])
    parser.add_argument('--data_preflight_require_source_disjoint', type=str2bool, default=True)
    parser.add_argument('--data_preflight_max_extra_pairs', type=int, default=0,
                        help='Maximum known audio pairs without metadata allowed by the audited snapshot.')
    parser.add_argument('--data_preflight_example_limit', type=int, default=10)
    parser.add_argument('--data_preflight_verify_audio_headers', type=str2bool, default=False,
                        help='Read matched audio headers before training and reject metadata or declared pair-length mismatches.')
    parser.add_argument('--data_preflight_verify_audio_decoder_tails', type=str2bool, default=False,
                        help='Probe each declared final frame with the runtime decoder; fully decode only suspicious files.')
    parser.add_argument('--data_preflight_header_workers', type=int, default=8,
                        help='Bounded thread count for audio header/tail-probe I/O.')
    parser.add_argument('--data_preflight_header_log_interval', type=int, default=10000,
                        help='Matched-pair progress interval for audio-header preflight; zero disables progress output.')
    parser.add_argument('--audio_excluded_pair_ids', nargs='*', default=[],
                        help='Explicit audited metadata pair IDs to quarantine without deleting source files.')
    parser.add_argument('--enable_training_health_monitor', type=str2bool, default=False)
    parser.add_argument('--health_log_interval_steps', type=int, default=10)
    parser.add_argument('--health_timing_window_size', type=int, default=200)
    parser.add_argument(
        '--precision',
        choices=['32-true', '16-mixed', 'bf16-mixed'],
        default='32-true',
        help='Lightning numerical precision; legacy configurations remain FP32.',
    )
    parser.add_argument(
        '--gradient_clip_val',
        type=float,
        default=0.0,
        help='Gradient norm clipping threshold; zero disables clipping.',
    )
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Physical DataLoader batch size per GPU; gradient accumulation never rewrites it.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW weight decay used by the released F-MoE training recipe.')
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--lr_warmup_ratio', type=float, default=0.02,
                        help='Fraction of optimizer steps used for linear warmup.')
    parser.add_argument('--lr_min_ratio', type=float, default=0.0,
                        help='Final cosine learning-rate multiplier in [0, 1].')
    parser.add_argument('--de_type', nargs='+', help='SonicMaster degradation-set members.')
    parser.add_argument('--loss_type', choices=['audio'], default='audio',
                        help='The five-term F-MoE restoration objective.')
    parser.add_argument('--lambda_complex', type=float, default=1.0)
    parser.add_argument('--lambda_wave', type=float, default=1.0)
    parser.add_argument('--lambda_mrstft', type=float, default=1.0)
    parser.add_argument('--lambda_degradation', type=float, default=0.1)
    parser.add_argument('--lambda_balance', type=float, default=0.1,
                        help='Weight of the MoE routing-balance term.')
    parser.add_argument('--mrstft_fft_sizes', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument('--mrstft_hop_sizes', nargs='+', type=int, default=[128, 256, 512, 1024])
    parser.add_argument('--mrstft_win_lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument(
        '--mrstft_sc_normalization',
        choices=['target', 'symmetric'],
        default='target',
        help=(
            'Spectral-convergence normalization. symmetric is bounded on silent '
            'targets; target preserves the legacy denominator.'
        ),
    )
    parser.add_argument('--loss_check_finite', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers.')
    parser.add_argument('--accum_grad', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint.')
    parser.add_argument('--fine_tune_from', type=str, default=None,
                        help='Warm-start from matching checkpoint tensors; unlike resume, architecture changes are allowed.')
    parser.add_argument('--checkpoint_id', type=str, help='checkpoint id')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Explicit checkpoint file for inference; overrides checkpoint_id selection.')
    parser.add_argument('--checkpoint_select', choices=['best', 'latest', 'last'], default='latest',
                        help='Checkpoint selected inside checkpoint_id when checkpoint_path is omitted.')
    parser.add_argument('--save_results', action="store_true", help="Save restored outputs.")
    parser.add_argument('--eval_shard_index', type=int, default=0,
                        help='Zero-based audio-evaluation shard index.')
    parser.add_argument('--eval_shard_count', type=int, default=1,
                        help='Number of deterministic interleaved audio-evaluation shards.')
    parser.add_argument('--skip_internal_metrics', action='store_true',
                        help='Save restoration outputs without computing local diagnostic metrics.')

    # Paths
    parser.add_argument('--wblogger', action="store_true", help='Log to Weights & Biases.')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help='Checkpoint directory.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for training.')

    parser.add_argument('--input_freq_bins', type=int, default=None,
                        help='Padded input frequency size; derived from n_fft when omitted.')
    parser.add_argument('--expert_embed_dims', nargs='+', type=int, default=[48, 64, 96, 128])
    parser.add_argument('--expert_num_bands', nargs='+', type=int, default=[16, 24, 32, 48])
    parser.add_argument('--expert_depths', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--expert_num_heads', nargs='+', type=int, default=[2, 4, 4, 8])
    parser.add_argument('--degradation_metadata_jsonl', type=str, default=None)
    parser.add_argument('--degradation_ignored_classes', nargs='+', default=[],
                        help='Degradation classes excluded from routing-analysis label summaries.')
    parser.add_argument('--degradation_ignore_hidden_clipping_labels', type=str2bool, default=False,
                        help='Exclude hidden-clipping rows from routing-analysis labels while retaining restoration pairs.')
    parser.add_argument('--routing_balance_ema_decay', type=float, default=0.99,
                        help='Cross-step expert-utilization decay.')
    parser.add_argument('--enable_routing_logging', type=str2bool, default=False)
    parser.add_argument('--profile_expert_flops', type=str2bool, default=False)
    parser.add_argument('--routing_output_dir', type=str, default=None)

    # Audio spectrogram I/O channels
    parser.add_argument('--inp_channels', type=int, default=1, help='Number of input spectrogram channels.')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output spectrogram channels.')

    # SonicMaster paired-audio settings
    parser.add_argument('--audio_data_dir', type=str, default='data/pairs', help='Root directory containing SonicMaster train/val/test audio splits.')
    parser.add_argument('--audio_ext', type=str, default='.flac', help='Audio file extension to scan recursively.')
    parser.add_argument('--audio_clean_dt', type=str, default='dt0', help='Legacy clean identifier for converted *_DT0 files; official clean files are unsuffixed source_id.flac.')
    parser.add_argument('--audio_pair_length_tolerance_samples', type=int, default=0,
                        help='Maximum decoded clean/degraded length mismatch before failing; zero requires exact alignment.')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Target audio sample rate.')
    parser.add_argument('--segment_seconds', type=float, default=4.0, help='Audio segment duration used for training.')
    parser.add_argument('--n_fft', type=int, default=1024, help='STFT FFT size.')
    parser.add_argument('--hop_length', type=int, default=256, help='STFT hop length.')
    parser.add_argument('--win_length', type=int, default=1024, help='STFT window length.')
    parser.add_argument('--stft_center', type=str2bool, default=True, help='Use centered STFT padding.')
    parser.add_argument('--spec_log', type=str2bool, default=True, help='Use log1p magnitude for STFT features.')
    parser.add_argument('--audio_single_random', action='store_true', help='For SonicMaster test: randomly pick one source and evaluate its available degradations.')
    parser.add_argument('--audio_random_seed', type=int, default=None, help='Optional random seed for reproducible source selection.')
    parser.add_argument('--audio_full_song', action='store_true', help='For SonicMaster test: evaluate full-length tracks without fixed-length cropping.')
    parser.add_argument('--audio_test_random_crop', action='store_true', help='For audio test: randomly crop each source to segment_seconds.')
    parser.add_argument('--metric_segment_seconds', type=float, default=3.0, help='Segment duration in seconds for chunk-wise audio metric computation (arithmetic mean over chunks).')
    parser.add_argument('--inference_segment_seconds', type=float, default=None,
                        help='Model inference chunk duration; defaults to the training segment duration and is independent of metric chunking.')
    parser.add_argument('--inference_overlap_ratio', type=float, default=0.25,
                        help='Fractional overlap used for cross-faded long-audio inference chunks.')

    parser.add_argument("--input_audio", type=str, default=None,
                    help="Path to one input audio file.")
    parser.add_argument("--output_audio", type=str, default=None,
                        help="Path to save restored audio.")
    parser.add_argument("--model_input_sr", type=int, default=44100,
                        help="Sample rate before feeding audio to model.")
    parser.add_argument("--output_sr", type=int, default=44100,
                        help="Sample rate for final saved audio.")
    return parser

def fmoe(parser):
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8])
    parser.add_argument('--num_dec_blocks', nargs='+', type=int, default=[2, 4, 4])
    parser.add_argument('--num_exp_blocks', type=int, default=4)
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', nargs='+', type=int, default=[1, 2, 4, 8])
    return parser

def _yaml_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    """Load a flat, serializable YAML mapping used as argparse defaults."""
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping, got {type(loaded).__name__}.")
    return loaded


def train_options(argv: Optional[Sequence[str]] = None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    config_defaults = _yaml_defaults(bootstrap_args.config)
    parser = fmoe(base_parser())

    valid_options = {action.dest for action in parser._actions}
    unknown_options = sorted(set(config_defaults) - valid_options)
    if unknown_options:
        parser.error(f"Unknown YAML option(s): {', '.join(unknown_options)}")
    parser.set_defaults(**config_defaults)

    options = parser.parse_args(argv)
    # The public release contains the paper's F-MoE configuration only.
    options.model = "F_MoE"
    options.trainset = "sonicmaster"
    options.representation = "complex"
    options.prediction_mode = "residual"
    options.stereo_mode = "stereo"
    options.use_complex_input = True
    options.degradation_encoder_type = "compound"
    options.degradation_embed_dim = 128
    options.num_degradation_classes = len(SONICMASTER_DEGRADATION_LABELS)
    options.return_dict = True
    options.audio_use_predegraded = True

    options.degradation_ignored_classes = [
        str(name).lower() for name in options.degradation_ignored_classes
    ]
    unknown_ignored_classes = sorted(
        set(options.degradation_ignored_classes) - set(SONICMASTER_DEGRADATION_LABELS)
    )
    if unknown_ignored_classes:
        parser.error(
            'unknown --degradation_ignored_classes: '
            + ', '.join(unknown_ignored_classes)
        )
    if len(options.degradation_ignored_classes) != len(
        set(options.degradation_ignored_classes)
    ):
        parser.error('--degradation_ignored_classes must not contain duplicates')

    if options.loss_type != "audio":
        parser.error("only the five-term F-MoE restoration objective is supported")
    if options.representation != "complex":
        parser.error("F-MoE training/evaluation requires complex STFT representation")

    if options.epochs < 1:
        parser.error('--epochs must be positive')
    if options.max_steps == 0 or options.max_steps < -1:
        parser.error('--max_steps must be -1 or a positive integer')
    if options.max_training_steps_without_override < 1:
        parser.error('--max_training_steps_without_override must be positive')
    if options.validation_batch_size < 1:
        parser.error('--validation_batch_size must be positive')
    if options.validation_num_workers < 0:
        parser.error('--validation_num_workers must be non-negative')
    if options.validation_max_samples < 0:
        parser.error('--validation_max_samples must be non-negative')
    if options.validation_interval_batches < 1:
        parser.error('--validation_interval_batches must be positive')
    if options.validation_sanity_steps < 0:
        parser.error('--validation_sanity_steps must be non-negative')
    if options.checkpoint_save_top_k < 1:
        parser.error('--checkpoint_save_top_k must be positive')
    if options.checkpoint_every_n_train_steps < 0:
        parser.error('--checkpoint_every_n_train_steps must be non-negative')
    if options.data_preflight_max_extra_pairs < 0:
        parser.error('--data_preflight_max_extra_pairs must be non-negative')
    if options.data_preflight_example_limit < 1:
        parser.error('--data_preflight_example_limit must be positive')
    if options.data_preflight_header_workers < 1:
        parser.error('--data_preflight_header_workers must be positive')
    if options.data_preflight_header_log_interval < 0:
        parser.error('--data_preflight_header_log_interval must be non-negative')
    if (
        options.data_preflight_verify_audio_decoder_tails
        and not options.data_preflight_verify_audio_headers
    ):
        parser.error(
            '--data_preflight_verify_audio_decoder_tails requires '
            '--data_preflight_verify_audio_headers true'
        )
    options.audio_excluded_pair_ids = [
        str(pair_id).casefold() for pair_id in options.audio_excluded_pair_ids
    ]
    if len(options.audio_excluded_pair_ids) != len(
        set(options.audio_excluded_pair_ids)
    ):
        parser.error('--audio_excluded_pair_ids must not contain duplicates')
    if options.health_log_interval_steps < 1:
        parser.error('--health_log_interval_steps must be positive')
    if options.health_timing_window_size < 1:
        parser.error('--health_timing_window_size must be positive')
    if options.gradient_clip_val < 0.0:
        parser.error('--gradient_clip_val must be non-negative')
    if options.batch_size < 1:
        parser.error('--batch_size must be positive')
    if options.accum_grad < 1:
        parser.error('--accum_grad must be positive')
    if options.lr <= 0.0:
        parser.error('--lr must be positive')
    if options.weight_decay < 0.0:
        parser.error('--weight_decay must be non-negative')
    if not 0.0 <= options.adam_beta1 < 1.0:
        parser.error('--adam_beta1 must be in [0, 1)')
    if not 0.0 <= options.adam_beta2 < 1.0:
        parser.error('--adam_beta2 must be in [0, 1)')
    if options.adam_eps <= 0.0:
        parser.error('--adam_eps must be positive')
    if not 0.0 <= options.lr_warmup_ratio < 1.0:
        parser.error('--lr_warmup_ratio must be in [0, 1)')
    if not 0.0 <= options.lr_min_ratio <= 1.0:
        parser.error('--lr_min_ratio must be in [0, 1]')
    if options.audio_pair_length_tolerance_samples < 0:
        parser.error('--audio_pair_length_tolerance_samples must be non-negative')
    if options.num_gpus < 1:
        parser.error('--num_gpus must be positive')
    if options.num_workers < 0:
        parser.error('--num_workers must be non-negative')
    if options.sample_rate < 1 or options.model_input_sr < 1 or options.output_sr < 1:
        parser.error('audio sample rates must be positive')
    if options.segment_seconds <= 0.0 or options.metric_segment_seconds <= 0.0:
        parser.error('segment durations must be positive')
    if options.inference_segment_seconds is None:
        options.inference_segment_seconds = options.segment_seconds
    if options.inference_segment_seconds <= 0.0:
        parser.error('--inference_segment_seconds must be positive')
    if not 0.0 <= options.inference_overlap_ratio < 1.0:
        parser.error('--inference_overlap_ratio must be in [0, 1)')
    if options.n_fft < 2 or options.hop_length < 1 or options.win_length < 1:
        parser.error('STFT sizes must be positive and n_fft must be at least two')
    if options.win_length > options.n_fft:
        parser.error('--win_length must not exceed --n_fft')
    if not (
        len(options.mrstft_fft_sizes)
        == len(options.mrstft_hop_sizes)
        == len(options.mrstft_win_lengths)
        and options.mrstft_fft_sizes
    ):
        parser.error('MRSTFT size lists must have equal non-zero length')
    for fft_size, hop_size, win_size in zip(
        options.mrstft_fft_sizes,
        options.mrstft_hop_sizes,
        options.mrstft_win_lengths,
    ):
        if fft_size < 2 or hop_size < 1 or not 1 <= win_size <= fft_size:
            parser.error('each MRSTFT resolution requires fft>=2, hop>0, and 0<win<=fft')
    loss_weights = {
        'lambda_complex': options.lambda_complex,
        'lambda_wave': options.lambda_wave,
        'lambda_mrstft': options.lambda_mrstft,
        'lambda_degradation': options.lambda_degradation,
        'lambda_balance': options.lambda_balance,
    }
    if any(value < 0.0 for value in loss_weights.values()):
        parser.error(f'audio loss weights must be non-negative: {loss_weights}')
    if options.resume_from and options.fine_tune_from:
        parser.error('--resume_from and --fine_tune_from are mutually exclusive')
    if options.run_id is not None:
        if not options.run_id or any(
            character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
            for character in options.run_id
        ):
            parser.error('--run_id may contain only letters, numbers, dot, underscore, and hyphen')
    if options.distributed_preflight_timeout_seconds <= 0.0:
        parser.error('--distributed_preflight_timeout_seconds must be positive')
    options.data_preflight_required_splits = [
        str(split).lower() for split in options.data_preflight_required_splits
    ]
    if len(options.data_preflight_required_splits) != len(
        set(options.data_preflight_required_splits)
    ):
        parser.error('--data_preflight_required_splits must not contain duplicates')
    if options.data_preflight_enabled and not options.degradation_metadata_jsonl:
        parser.error('SonicMaster data preflight requires degradation metadata')
    if options.validation_enabled and not options.degradation_metadata_jsonl:
        parser.error('validation requires --degradation_metadata_jsonl for split isolation')

    if not options.num_blocks or any(value < 1 for value in options.num_blocks):
        parser.error('--num_blocks must contain positive values')
    levels = len(options.num_blocks)
    if len(options.heads) != levels or any(value < 1 for value in options.heads):
        parser.error('--heads must contain one positive value per encoder level')
    if len(options.num_dec_blocks) != levels - 1 or any(
        value < 1 for value in options.num_dec_blocks
    ):
        parser.error('--num_dec_blocks must contain one positive value per decoder level')
    if options.dim < 1 or options.num_exp_blocks < 1:
        parser.error('model dimensions and expert count must be positive')
    for level, heads in enumerate(options.heads):
        channels = options.dim * (2 ** level)
        if channels % heads:
            parser.error(
                f'encoder channels {channels} at level {level} are not divisible by heads={heads}'
            )

    raw_freq_bins = options.n_fft // 2 + 1
    model_multiple = 2 ** (len(options.num_blocks) - 1)
    if options.input_freq_bins is None:
        options.input_freq_bins = (
            (raw_freq_bins + model_multiple - 1) // model_multiple
        ) * model_multiple
    expected_input_freq_bins = (
        (raw_freq_bins + model_multiple - 1) // model_multiple
    ) * model_multiple
    if options.trainset.lower() == 'sonicmaster' and options.input_freq_bins != expected_input_freq_bins:
        parser.error(
            '--input_freq_bins must equal the padded one-sided STFT size '
            f'{expected_input_freq_bins} for n_fft={options.n_fft} and model depth '
            f'{len(options.num_blocks)}; got {options.input_freq_bins}'
        )
    expert_list_lengths = {
        len(options.expert_embed_dims),
        len(options.expert_num_bands),
        len(options.expert_depths),
        len(options.expert_num_heads),
    }
    if expert_list_lengths != {options.num_exp_blocks}:
        parser.error('all band expert lists must match --num_exp_blocks')
    if not 0.0 < options.routing_balance_ema_decay < 1.0:
        parser.error('--routing_balance_ema_decay must be in (0, 1)')
    if options.enable_routing_logging and not options.return_dict:
        parser.error('--enable_routing_logging requires --return_dict true')
    if options.enable_routing_logging and options.num_degradation_classes != 19:
        parser.error('SonicMaster routing analysis requires exactly 19 classes')
    if options.profile_expert_flops and not options.enable_routing_logging:
        parser.error('--profile_expert_flops requires --enable_routing_logging true')

    # Select representation channels automatically for the audio workflow.
    if options.trainset.lower() == "sonicmaster":
        audio_channels = 2 if options.stereo_mode == 'stereo' else 1
        representation_factor = 2 if options.representation == 'complex' else 1
        options.inp_channels = audio_channels * representation_factor
        options.out_channels = audio_channels * representation_factor
        if options.de_type is None:
            if options.audio_use_predegraded:
                options.de_type = [f"dt{i}" for i in range(1, 8)]
            else:
                options.de_type = [f"dt{i}" for i in range(1, 13)]
        options.de_type = [str(name).lower() for name in options.de_type]
        if not options.de_type or len(options.de_type) != len(set(options.de_type)):
            parser.error('--de_type must contain unique degradation names')
        if any(not name.startswith('dt') or not name[2:].isdigit() for name in options.de_type):
            parser.error('audio --de_type entries must use dt<number> names')

    return options
