import glob
import os
import random
import re
import shutil
import subprocess
import warnings
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from deg.degradation import apply_degradation
from data.degradation_metadata import (
    SONICMASTER_DEGRADATION_LABELS,
    degradation_multi_hot,
    load_sonicmaster_metadata,
    pair_metadata_key,
)
from utils.audio_utils import (
    complex_to_channels,
    ensure_audio_channels,
    fit_audio_length,
    pad_spectrogram_for_fmoe,
    stft_complex,
    stft_mag_phase,
)

try:
    import torchaudio
except Exception as exc:  # pragma: no cover
    torchaudio = None
    _TORCHAUDIO_IMPORT_ERROR = exc
else:
    _TORCHAUDIO_IMPORT_ERROR = None

try:
    import librosa
except Exception:
    librosa = None


def _decode_with_ffmpeg(path: str, sample_rate: int, channels: int = 2) -> torch.Tensor:
    """Decode audio file to float32 PCM tensor [C, T] using ffmpeg CLI."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is not available in PATH.")

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed for {path}: {err}")

    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise RuntimeError(f"Decoded empty audio stream from {path}")
    if pcm.size % channels != 0:
        pcm = pcm[: pcm.size - (pcm.size % channels)]
    wav = pcm.reshape(-1, channels).T
    return torch.from_numpy(wav).float()


class AudioTrainDataset(Dataset):
    """Paired-audio dataset for F-MoE training and evaluation."""

    def __init__(self, args, split: str = "train"):
        super().__init__()
        self.args = args
        self.split = split
        self.full_song_test = bool(getattr(args, "audio_full_song", False)) and split == "test"
        self.random_crop_test = bool(getattr(args, "audio_test_random_crop", False)) and split == "test"
        self.use_predegraded_pairs = bool(getattr(args, "audio_use_predegraded", False))
        self.representation = str(getattr(args, "representation", "magnitude"))
        self.audio_channels = 2 if getattr(args, "stereo_mode", "mono") == "stereo" else 1
        self.clean_dt_name = str(getattr(args, "audio_clean_dt", "dt0")).lower()
        self.excluded_pair_ids = {
            str(pair_id).casefold()
            for pair_id in getattr(args, "audio_excluded_pair_ids", [])
        }
        self.de_type = args.de_type
        self.de_dict = {name: idx for idx, name in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: name for idx, name in enumerate(self.de_type)}
        self.degradation_metadata_path = getattr(args, "degradation_metadata_jsonl", None)
        ignored_names = {
            str(name).lower()
            for name in getattr(args, "degradation_ignored_classes", [])
        }
        unknown_ignored = ignored_names - set(SONICMASTER_DEGRADATION_LABELS)
        if unknown_ignored:
            raise ValueError(
                f"Unknown ignored degradation classes: {sorted(unknown_ignored)}"
            )
        self.degradation_class_mask = torch.ones(
            len(SONICMASTER_DEGRADATION_LABELS), dtype=torch.float32
        )
        for index, name in enumerate(SONICMASTER_DEGRADATION_LABELS):
            if name in ignored_names:
                self.degradation_class_mask[index] = 0.0
        self.ignore_hidden_clipping_labels = bool(
            getattr(args, "degradation_ignore_hidden_clipping_labels", False)
        )
        self.degradation_metadata = (
            load_sonicmaster_metadata(self.degradation_metadata_path, split=split)
            if self.degradation_metadata_path
            else {}
        )

        self.segment_len = int(args.segment_seconds * args.sample_rate)
        self.pair_length_tolerance_samples = int(
            getattr(args, "audio_pair_length_tolerance_samples", 0)
        )
        configured_seed = getattr(args, "audio_random_seed", None)
        self.base_seed = int(
            getattr(args, "seed", 2025) if configured_seed is None else configured_seed
        )
        if self.use_predegraded_pairs:
            self.pairs = self._scan_predegraded_pairs()
            self.files = [p[1] for p in self.pairs]
            self.effective_len = len(self.pairs)
        else:
            self.files = self._scan_audio_files()
            # Keep epoch length comparable to "file x degradation" enumeration while sampling online.
            self.effective_len = len(self.files) * len(self.de_type)

    def _scan_audio_files(self) -> List[str]:
        pattern = os.path.join(self.args.audio_data_dir, self.split, "**", f"*{self.args.audio_ext}")
        files = sorted(glob.glob(pattern, recursive=True))
        if len(files) == 0:
            raise ValueError(f"No audio files found for pattern: {pattern}")
        return files

    def _candidate_scan_roots(self) -> List[str]:
        split_dir = os.path.join(self.args.audio_data_dir, self.split)
        if os.path.isdir(split_dir):
            return [split_dir]
        return [self.args.audio_data_dir]

    def _scan_predegraded_pairs(self):
        ext = self.args.audio_ext.lower()
        if not ext.startswith("."):
            ext = "." + ext

        selected = {name.lower() for name in self.de_type}
        clean_map = {}
        degraded_map = {}

        metadata_sources = {
            key.rsplit("_deg", maxsplit=1)[0]
            for key in self.degradation_metadata
            if "_deg" in key
        }
        suffix_pattern = re.compile(
            rf"^(?P<song>.+)_(?P<kind>dt|deg)(?P<dt>\d+){re.escape(ext)}$",
            flags=re.IGNORECASE,
        )

        def add_clean_source(song_key: str, path: str) -> None:
            if song_key in clean_map:
                raise ValueError(
                    "Ambiguous clean audio pair key {!r}: {!r} and {!r}".format(
                        song_key, clean_map[song_key], path
                    )
                )
            clean_map[song_key] = path

        for root in self._candidate_scan_roots():
            pattern = os.path.join(root, "**", f"*{self.args.audio_ext}")
            for path in sorted(glob.glob(pattern, recursive=True)):
                basename = os.path.basename(path)
                match = suffix_pattern.fullmatch(basename)
                if match is None:
                    # Official SonicMaster clean targets are named source_id.flac.
                    song_key = os.path.splitext(basename)[0].casefold()
                    if self.degradation_metadata and song_key not in metadata_sources:
                        continue
                    add_clean_source(song_key, path)
                    continue

                song_key = match.group("song").casefold()
                dt_name = f"dt{int(match.group('dt'))}"
                kind = match.group("kind").lower()
                if kind == "dt" and dt_name == self.clean_dt_name:
                    add_clean_source(song_key, path)
                elif dt_name in selected:
                    degraded_key = (song_key, dt_name)
                    if degraded_key in degraded_map:
                        raise ValueError(
                            "Ambiguous degraded audio pair key {!r}: {!r} and {!r}".format(
                                degraded_key, degraded_map[degraded_key], path
                            )
                        )
                    degraded_map[degraded_key] = path

        pairs = []
        unmatched_metadata_keys = []
        for (song_key, dt_name), degraded_path in degraded_map.items():
            pair_id = f"{song_key}_deg{int(dt_name[2:])}"
            if pair_id in self.excluded_pair_ids:
                continue
            clean_path = clean_map.get(song_key)
            if clean_path is None:
                continue
            if self.degradation_metadata:
                metadata_key = pair_metadata_key(clean_path, dt_name)
                if metadata_key not in self.degradation_metadata:
                    unmatched_metadata_keys.append(metadata_key)
                    continue
            pairs.append((degraded_path, clean_path, dt_name))

        self.unmatched_metadata_pair_keys = tuple(unmatched_metadata_keys)
        if unmatched_metadata_keys:
            examples = ", ".join(unmatched_metadata_keys[:3])
            warnings.warn(
                "Skipped {} pre-degraded pair(s) without matching SonicMaster "
                "metadata for split {!r}; examples: {}".format(
                    len(unmatched_metadata_keys), self.split, examples
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        if len(pairs) == 0:
            roots = ", ".join(self._candidate_scan_roots())
            raise ValueError(
                "No pre-degraded/clean audio pairs found. "
                f"roots={roots}, ext={self.args.audio_ext}, clean_dt={self.clean_dt_name}, selected={sorted(selected)}"
            )

        return pairs

    def __len__(self):
        return self.effective_len

    def _load_audio(self, path: str):
        errors = []
        # Prefer torchaudio for speed and tensor-native decoding.
        if torchaudio is not None:
            try:
                wav, sr = torchaudio.load(path)
                if sr != self.args.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.args.sample_rate)
                    sr = self.args.sample_rate
                return wav, sr
            except Exception as exc:
                errors.append(f"torchaudio: {type(exc).__name__}: {exc}")
        elif _TORCHAUDIO_IMPORT_ERROR is not None:
            errors.append(
                "torchaudio unavailable: {}: {}".format(
                    type(_TORCHAUDIO_IMPORT_ERROR).__name__, _TORCHAUDIO_IMPORT_ERROR
                )
            )

        try:
            wav = _decode_with_ffmpeg(path, self.args.sample_rate, channels=2)
            return wav, self.args.sample_rate
        except Exception as exc:
            errors.append(f"ffmpeg: {type(exc).__name__}: {exc}")

        # Fallback for environments where torchaudio+torchcodec cannot decode mp4 containers.
        if librosa is None:
            errors.append("librosa unavailable")
            raise RuntimeError(
                f"Failed to decode audio {path!r}; " + " | ".join(errors)
            )

        try:
            wav_np, sr = librosa.load(path, sr=self.args.sample_rate, mono=False)
        except Exception as exc:
            errors.append(f"librosa: {type(exc).__name__}: {exc}")
            raise RuntimeError(
                f"Failed to decode audio {path!r}; " + " | ".join(errors)
            ) from exc
        if wav_np.ndim == 1:
            wav_np = wav_np[None, :]
        wav = torch.from_numpy(wav_np).float()
        return wav, sr

    def _degradation_chain(self, clean_path: str, dt_raw: str, legacy_id: int):
        if not self.degradation_metadata:
            return {
                "ids": [],
                "names": [],
                "params": {},
                "order": [],
                "groups": [],
                "compound_level": 0,
                "hidden_clipping": {"enabled": False, "level": 0},
                "legacy_id": legacy_id,
                "has_labels": False,
            }
        key = pair_metadata_key(clean_path, dt_raw)
        if key not in self.degradation_metadata:
            raise KeyError(
                f"No SonicMaster metadata for pair key {key!r}; "
                f"source={clean_path!r}, degradation={dt_raw!r}"
            )
        chain = dict(self.degradation_metadata[key])
        chain["legacy_id"] = legacy_id
        chain["has_labels"] = True
        return chain

    def get_waveform_pair(self, idx: int, return_metadata: bool = False):
        """Load a channel-aligned ``[C, N]`` degraded/clean waveform pair."""
        if self.use_predegraded_pairs:
            pair_idx = idx % len(self.pairs)
            degraded_path, clean_path, dt_raw = self.pairs[pair_idx]
            de_id = self.de_dict[dt_raw]

            degraded_wav, sr = self._load_audio(degraded_path)
            clean_wav, clean_sr = self._load_audio(clean_path)
            if clean_sr != sr:
                raise ValueError(f"Sample rate mismatch between pair: {degraded_path} ({sr}) vs {clean_path} ({clean_sr})")

            degraded_wav = ensure_audio_channels(degraded_wav, self.audio_channels)
            clean_wav = ensure_audio_channels(clean_wav, self.audio_channels)

            degraded_len = int(degraded_wav.size(-1))
            clean_len = int(clean_wav.size(-1))
            length_delta = abs(degraded_len - clean_len)
            if length_delta > self.pair_length_tolerance_samples:
                raise ValueError(
                    "Decoded pair length mismatch exceeds tolerance: "
                    f"degraded={degraded_len}, clean={clean_len}, delta={length_delta}, "
                    f"tolerance={self.pair_length_tolerance_samples}, "
                    f"degraded_path={degraded_path!r}, clean_path={clean_path!r}"
                )
            cur_len = min(degraded_len, clean_len)
            degraded_wav = degraded_wav[..., :cur_len]
            clean_wav = clean_wav[..., :cur_len]

            if self.full_song_test:
                pass
            elif self.split == "train":
                if cur_len > self.segment_len:
                    start = random.randint(0, cur_len - self.segment_len)
                    degraded_wav = degraded_wav[..., start:start + self.segment_len]
                    clean_wav = clean_wav[..., start:start + self.segment_len]
                elif cur_len < self.segment_len:
                    pad = self.segment_len - cur_len
                    degraded_wav = torch.nn.functional.pad(degraded_wav, (0, pad), mode="constant", value=0.0)
                    clean_wav = torch.nn.functional.pad(clean_wav, (0, pad), mode="constant", value=0.0)
            elif self.random_crop_test:
                crop_rng = np.random.default_rng(self.base_seed + pair_idx)
                if cur_len > self.segment_len:
                    start = int(crop_rng.integers(0, cur_len - self.segment_len + 1))
                    degraded_wav = degraded_wav[..., start:start + self.segment_len]
                    clean_wav = clean_wav[..., start:start + self.segment_len]
                elif cur_len < self.segment_len:
                    pad = self.segment_len - cur_len
                    degraded_wav = torch.nn.functional.pad(degraded_wav, (0, pad), mode="constant", value=0.0)
                    clean_wav = torch.nn.functional.pad(clean_wav, (0, pad), mode="constant", value=0.0)
            else:
                if cur_len > self.segment_len:
                    start = (cur_len - self.segment_len) // 2
                    degraded_wav = degraded_wav[..., start:start + self.segment_len]
                    clean_wav = clean_wav[..., start:start + self.segment_len]
                elif cur_len < self.segment_len:
                    pad = self.segment_len - cur_len
                    degraded_wav = torch.nn.functional.pad(degraded_wav, (0, pad), mode="constant", value=0.0)
                    clean_wav = torch.nn.functional.pad(clean_wav, (0, pad), mode="constant", value=0.0)

            if return_metadata:
                chain = self._degradation_chain(clean_path, dt_raw, de_id)
                return degraded_wav, clean_wav, sr, clean_path, de_id, chain
            return degraded_wav, clean_wav, sr, clean_path, de_id

        if self.split == "train":
            # Optional online generation is retained for preparing new paired data.
            audio_path = random.choice(self.files)
            dt_raw = random.choice(self.de_type)
            de_id = self.de_dict[dt_raw]
            dt_name = dt_raw.upper()
            # Python's worker RNG is deterministically seeded by Lightning;
            # deriving NumPy's Generator from it preserves reproducibility while
            # still producing fresh degradation parameters for every sample.
            rng = np.random.default_rng(random.getrandbits(64))
            file_idx = None
        else:
            # Deterministic ordering for test to keep waveforms/metrics reproducible.
            file_idx = idx % len(self.files)
            de_idx = (idx // len(self.files)) % len(self.de_type)
            audio_path = self.files[file_idx]
            dt_raw = self.de_type[de_idx]
            de_id = self.de_dict[dt_raw]
            dt_name = dt_raw.upper()
            rng = np.random.default_rng(self.base_seed + idx)

        clean_wav, sr = self._load_audio(audio_path)
        clean_wav = ensure_audio_channels(clean_wav, self.audio_channels)
        if self.full_song_test:
            pass
        elif self.split == "train":
            clean_wav = fit_audio_length(clean_wav, self.segment_len, train=True)
        elif self.random_crop_test:
            # Same random crop for the same song across all degradations.
            assert file_idx is not None
            crop_rng = np.random.default_rng(self.base_seed + file_idx)
            cur_len = clean_wav.size(-1)
            if cur_len > self.segment_len:
                start = int(crop_rng.integers(0, cur_len - self.segment_len + 1))
                clean_wav = clean_wav[..., start:start + self.segment_len]
            elif cur_len < self.segment_len:
                clean_wav = torch.nn.functional.pad(clean_wav, (0, self.segment_len - cur_len), mode="constant", value=0.0)
        else:
            # Default test behavior: deterministic center crop/pad.
            clean_wav = fit_audio_length(clean_wav, self.segment_len, train=False)

        degraded_wav, degradation_params = apply_degradation(clean_wav, sr, dt_name, rng=rng)
        if return_metadata:
            chain = {
                "ids": [],
                "names": [],
                "params": degradation_params,
                "order": [],
                "groups": [],
                "compound_level": 0,
                "hidden_clipping": {"enabled": False, "level": 0},
                "legacy_id": de_id,
                "has_labels": False,
            }
            return degraded_wav, clean_wav, sr, audio_path, de_id, chain
        return degraded_wav, clean_wav, sr, audio_path, de_id

    def __getitem__(self, idx):
        degraded_wav, clean_wav, _, audio_path, de_id, chain = self.get_waveform_pair(
            idx, return_metadata=True
        )

        if self.representation == "complex":
            degraded_spec = complex_to_channels(
                stft_complex(
                    degraded_wav,
                    n_fft=self.args.n_fft,
                    hop_length=self.args.hop_length,
                    win_length=self.args.win_length,
                    center=self.args.stft_center,
                )
            )
            clean_spec = complex_to_channels(
                stft_complex(
                    clean_wav,
                    n_fft=self.args.n_fft,
                    hop_length=self.args.hop_length,
                    win_length=self.args.win_length,
                    center=self.args.stft_center,
                )
            )
            degraded_spec, pad_info = pad_spectrogram_for_fmoe(degraded_spec, multiple=8)
            clean_spec, clean_pad_info = pad_spectrogram_for_fmoe(clean_spec, multiple=8)
            if clean_pad_info != pad_info:
                raise RuntimeError(
                    f"Degraded/clean STFT padding mismatch: {pad_info} vs {clean_pad_info}"
                )
            hidden_clipping = dict(
                chain.get("hidden_clipping", {"enabled": False, "level": 0})
            )
            label_is_trusted = bool(chain["has_labels"])
            if (
                getattr(self, "ignore_hidden_clipping_labels", False)
                and hidden_clipping["enabled"]
            ):
                label_is_trusted = False
            class_mask = getattr(
                self,
                "degradation_class_mask",
                torch.ones(len(SONICMASTER_DEGRADATION_LABELS), dtype=torch.float32),
            )
            return {
                "path": audio_path,
                "degradation_ids": torch.tensor(chain["ids"], dtype=torch.long),
                "degradation_names": list(chain["names"]),
                "degradation_params": chain["params"],
                "degradation_order": torch.tensor(chain["order"], dtype=torch.long),
                "degradation_groups": list(chain["groups"]),
                "degradation_hidden_clipping": hidden_clipping,
                "degradation_multi_hot": degradation_multi_hot(chain["names"]),
                "degradation_class_mask": class_mask.clone(),
                "degradation_label_mask": float(label_is_trusted),
                "compound_level": int(chain["compound_level"]),
                "legacy_degradation_id": de_id,
                "degraded_wave": degraded_wav,
                "clean_wave": clean_wav,
                "degraded_spec": degraded_spec,
                "clean_spec": clean_spec,
                "pad_info": pad_info,
                "audio_length": degraded_wav.size(-1),
            }

        degraded_mag, _, _ = stft_mag_phase(
            degraded_wav,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            win_length=self.args.win_length,
            spec_log=self.args.spec_log,
        )
        clean_mag, _, _ = stft_mag_phase(
            clean_wav,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            win_length=self.args.win_length,
            spec_log=self.args.spec_log,
        )

        # Pad frequency/time dimensions to satisfy the encoder's downsampling divisibility.
        degraded_mag, _ = pad_spectrogram_for_fmoe(degraded_mag, multiple=8)
        clean_mag, _ = pad_spectrogram_for_fmoe(clean_mag, multiple=8)

        # Return magnitude-spectrogram tensors with shape [C, F, T].
        return [audio_path, de_id], degraded_mag, clean_mag


class AudioTestDataset(AudioTrainDataset):
    """Audio test dataset with deterministic center-crop behavior."""

    def __init__(self, args):
        super().__init__(args, split="test")


def audio_collate_fn(batch):
    """Collate extensible audio dictionaries without padding metadata lists.

    Tensor fields are stacked. Variable-length names/ids and parameter
    dictionaries remain Python lists for audit/export. ``pad_info`` is
    converted to a mapping accepted by ``unpad_spectrogram`` after collation.
    Historical tuple samples continue to use PyTorch's default collation.
    """
    if not batch or not isinstance(batch[0], dict):
        return default_collate(batch)
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key == "pad_info":
            result[key] = {
                "pad_f": torch.tensor([value[0] for value in values], dtype=torch.long),
                "pad_t": torch.tensor([value[1] for value in values], dtype=torch.long),
            }
        elif key in {
            "path",
            "degradation_ids",
            "degradation_names",
            "degradation_params",
            "degradation_order",
            "degradation_groups",
            "degradation_hidden_clipping",
        }:
            result[key] = values
        else:
            result[key] = default_collate(values)
    return result
