from pathlib import Path

import torch

from inference import _destination_paths, _inputs, restore_waveform
from utils.inference_utils import crossfade_audio_chunks, inference_chunk_starts


def test_folder_input_accepts_unicode_spaces_and_mixed_case_extensions(tmp_path: Path):
    nested = tmp_path / "子目录 with spaces"
    nested.mkdir()
    expected = {
        nested / "中文歌曲 (现场版).FLAC",
        tmp_path / "artist.song.v1 [remaster].wav",
        tmp_path / "--leading-dash.OGG",
    }
    for path in expected:
        path.touch()
    (tmp_path / "ignore.txt").touch()

    assert set(_inputs(tmp_path)) == expected


def test_destination_paths_preserve_subdirectories_and_avoid_same_stem_collisions(
    tmp_path: Path,
):
    source_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    nested = source_root / "album"
    nested.mkdir(parents=True)
    wav = nested / "song.wav"
    flac = nested / "song.flac"
    wav.touch()
    flac.touch()

    destinations = _destination_paths(source_root, [wav, flac], output_root)

    assert destinations[wav] == output_root / "album" / "song_wav_restored.wav"
    assert destinations[flac] == output_root / "album" / "song_flac_restored.wav"


def test_hann_overlap_add_reconstructs_identical_chunks_without_seams():
    total = 16
    starts = inference_chunk_starts(total, segment_length=10, overlap_ratio=0.4)
    signal = torch.linspace(-1.0, 1.0, total).repeat(2, 1)
    chunks = [signal[:, start : start + 10] for start in starts]

    restored = crossfade_audio_chunks(chunks, starts, total)

    torch.testing.assert_close(restored, signal)


def test_empty_audio_is_rejected_before_model_execution():
    try:
        inference_chunk_starts(0, segment_length=10, overlap_ratio=0.25)
    except ValueError as error:
        assert "at least one sample" in str(error)
    else:
        raise AssertionError("empty audio must be rejected")



class _IdentitySpectrogramModel(torch.nn.Module):
    def forward(self, spectrogram):
        return {"restored_spec": spectrogram}


def test_full_inference_frontend_roundtrips_audio_with_overlap_and_padded_tail():
    torch.manual_seed(7)
    waveform = torch.randn(2, 2 * 44_100 + 12_345) * 0.1

    restored = restore_waveform(
        _IdentitySpectrogramModel(),
        waveform,
        sample_rate=44_100,
        device=torch.device("cpu"),
        segment_seconds=1.0,
        overlap=0.25,
    )

    assert restored.shape == waveform.shape
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, waveform, rtol=1e-4, atol=1e-5)
