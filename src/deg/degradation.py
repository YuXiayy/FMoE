# degradations.py
import os
import io
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any
import logging
import warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import torch

try:
    import augment  # type: ignore
    _HAS_AUGMENT = True
except Exception:
    augment = None  # type: ignore
    _HAS_AUGMENT = False

try:
    import scipy.signal as scipy_signal  # type: ignore
except Exception:
    scipy_signal = None  # type: ignore

try:
    from scipy.interpolate import interp1d as scipy_interp1d  # type: ignore
except Exception:
    scipy_interp1d = None  # type: ignore

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', message='.*apply_codec.*deprecated.*')
warnings.filterwarnings('ignore', message='.*torchaudio.save_with_torchcodec.*')
warnings.filterwarnings('ignore', message='.*TorchCodec.*encoder.*')
warnings.filterwarnings('ignore', message='.*torchcodec.encoders.AudioEncoder.*')
# Try imports that may or may not be present
try:
    import torchaudio
    try:
        # newer torchaudio exposes apply_codec in functional
        from torchaudio.functional import apply_codec as ta_apply_codec  # type: ignore
    except Exception:
        ta_apply_codec = None
    _HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None  # type: ignore
    ta_apply_codec = None
    _HAS_TORCHAUDIO = False

# Try to import encodec (Meta's Encodec) if available
_ENCODEC_AVAILABLE = False
_encodec_model_ctor = None
try:
    import encodec  # type: ignore
    from encodec import EncodecModel  # type: ignore
    # Provide model constructor wrappers for common sample rates if available
    def _encodec_get_model(sr: int):
        # This tries common helpers — API can vary between package versions.
        # Many users instantiate EncodecModel.encodec_model_48khz() for 48k models.
        if hasattr(EncodecModel, "encodec_model_48khz"):
            if sr == 48000:
                return EncodecModel.encodec_model_48khz()
        # fallback: try default constructor and set sample rate
        try:
            m = EncodecModel()
            if hasattr(m, "set_sample_rate"):
                m.set_sample_rate(sr)
            return m
        except Exception:
            return None
    _ENCODEC_AVAILABLE = True
    _encodec_model_ctor = _encodec_get_model
except Exception:
    _ENCODEC_AVAILABLE = False
    _encodec_model_ctor = None

# Try to import a Descript Audio Codec (DAC) python binding if available.
# There isn't a single canonical package name; try a couple of plausible names.
_DAC_AVAILABLE = False
_dac_encoder = None
for _dac_name in ("descript_audio_codec", "descript_audio", "dac", "descript"):
    try:
        _mod = __import__(_dac_name)
        # if it exposes a helpful API, keep reference (we won't assume details)
        _dac_encoder = _mod
        _DAC_AVAILABLE = True
        break
    except Exception:
        pass

# Check if AudioEffector is available (torchaudio >= 2.0.0)
_HAS_AUDIO_EFFECTOR = False
if _HAS_TORCHAUDIO:
    try:
        from torchaudio.io import AudioEffector
        _HAS_AUDIO_EFFECTOR = True
    except Exception:
        _HAS_AUDIO_EFFECTOR = False

def _ensure_tensor_audio(wav: torch.Tensor) -> torch.Tensor:
    """Ensure audio is Tensor with shape [C, T] and dtype float32 on CPU for processing."""
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.shape[0] > wav.shape[1]:
        pass
    return wav.float().cpu()

def _copy_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(options) if options is not None else {}

def _power_per_channel(wav: torch.Tensor) -> torch.Tensor:
    # wav shape [C, T]
    return (wav.pow(2).mean(dim=-1)).clamp(min=1e-20)

def add_awgn_snr(wav: torch.Tensor, snr_db: float, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Add Gaussian noise to achieve target SNR (dB) per channel."""
    if rng is None:
        rng = np.random.default_rng()
    sig_pow = _power_per_channel(wav)  # per-channel power
    sig_pow_np = sig_pow.numpy()

    snr_db = np.clip(snr_db, -80, 80)
    snr_linear = 10.0 ** (snr_db / 10.0)
    snr_linear = max(snr_linear, 1e-20)
    noise_pow = sig_pow_np / snr_linear
    noise_pow = np.maximum(noise_pow, 1e-20)
    noise = []
    for p, c in zip(noise_pow, range(wav.shape[0])):
        std = float(np.sqrt(p))
        n = rng.normal(0.0, std, size=(wav.shape[1],)).astype(np.float32)
        noise.append(n)
    noise = np.stack(noise, axis=0)
    return wav + torch.from_numpy(noise)

def _fft_convolve_channel(wav_ch: np.ndarray, ir: np.ndarray) -> np.ndarray:
    # wav_ch: (T,), ir: (L,)
    n = wav_ch.shape[0] + ir.shape[0] - 1
    N = 1 << (int(np.ceil(np.log2(n))))
    W = np.fft.rfft(wav_ch, n=N)
    H = np.fft.rfft(ir, n=N)
    y = np.fft.irfft(W * H, n=N)
    y = y[:wav_ch.shape[0]]  # trim to input length (keep same length)
    return y

def convolve_ir(wav: torch.Tensor, ir: np.ndarray) -> torch.Tensor:
    """Convolve each channel with IR (numpy) and return same-length output (trim or pad)."""
    # We use fft convolution per channel (on CPU numpy) for simplicity
    wav_np = wav.numpy()
    out = []
    for c in range(wav_np.shape[0]):
        y = _fft_convolve_channel(wav_np[c], ir)
        # ensure same length as input
        if y.shape[0] < wav_np.shape[1]:
            y = np.pad(y, (0, wav_np.shape[1] - y.shape[0]), mode='constant')
        else:
            y = y[:wav_np.shape[1]]
        out.append(y)
    out = np.stack(out, axis=0).astype(np.float32)
    return torch.from_numpy(out)

def _make_multipath_ir(sr: int, delays_ms: list, gains: list, decay: float = 0.0) -> np.ndarray:
    """Construct simple FIR IR with given delays (ms) and gains. decay is extra exponential tail length (s)."""
    assert len(delays_ms) == len(gains)
    max_delay = int(max(delays_ms) * sr / 1000.0)
    L = max_delay + 1 + int(decay * sr)
    ir = np.zeros(L, dtype=np.float32)
    for d_ms, g in zip(delays_ms, gains):
        idx = int(round(d_ms * sr / 1000.0))
        if idx < L:
            ir[idx] += g
    # optional exponential tail to simulate reverberation
    if decay > 0:
        tail = np.exp(-np.arange(int(decay * sr)) / float(decay * sr)) * 1e-3
        tail = tail.astype(np.float32)
        ir = np.concatenate([ir, tail])
    # normalize energy to ~1 (so convolution doesn't blow up)
    s = np.sum(np.abs(ir)) + 1e-20
    ir = ir / s
    return ir

def _rayleigh_envelope(length: int, sr: int, corr_time: float = 0.2, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a smooth Rayleigh envelope of given length (samples). corr_time in seconds controls smoothing."""
    if rng is None:
        rng = np.random.default_rng()
    # generate Rayleigh samples, then lowpass-smooth using simple moving-average
    x = rng.rayleigh(scale=1.0, size=(int(np.ceil(length / max(1, int(sr * corr_time)))),))
    # upsample to length via linear interpolation
    xp = np.linspace(0, 1, num=x.shape[0])
    xq = np.linspace(0, 1, num=length)
    env = np.interp(xq, xp, x)
    # normalize to mean 1
    env = env / (np.mean(env) + 1e-20)
    return env.astype(np.float32)

def dt1_radio(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    snr_db = float(opts.get('snr_db', 26.0))
    delays = opts.get('multipath_delays_ms', [0.0, 5.0, 30.0])
    gains = opts.get('multipath_gains', [1.0, 0.6, 0.3])
    corr_time = float(opts.get('rayleigh_corr_time', 0.25))
    if rng is None:
        rng = np.random.default_rng()

    wav0 = _ensure_tensor_audio(wav)  # [C,T]
    # make multipath IR
    ir = _make_multipath_ir(sr, delays, gains, decay=0.0)
    # convolve
    wav_mp = convolve_ir(wav0, ir)
    # apply Rayleigh envelope (same to all channels)
    env = _rayleigh_envelope(wav_mp.shape[1], sr, corr_time=corr_time, rng=rng)
    wav_faded = wav_mp * torch.from_numpy(env).unsqueeze(0)
    # add AWGN at target SNR
    wav_noisy = add_awgn_snr(wav_faded, snr_db, rng=rng)
    meta = {'dt': 'DT1', 'params': {'snr_db': snr_db, 'delays_ms': delays, 'gains': gains, 'rayleigh_corr_time': corr_time}}
    return wav_noisy, meta

def _tape_eq_freq_response(n_fft: int, sr: int, profile: Optional[Dict[str, float]] = None) -> np.ndarray:
    freqs = np.linspace(0, sr/2, num=(n_fft//2 + 1))
    if profile is None:
        # default: slight bass boost below 200Hz, gentle mid cut 1-4k, high rolloff above 8k
        profile = {20: 3.0, 200: 2.0, 1000: -1.5, 4000: -3.0, 8000: -6.0, sr/2: -12.0}
    xp = np.array(list(profile.keys()))
    yp = np.array(list(profile.values()))
    gains_db = np.interp(freqs, xp, yp)
    gains = 10 ** (gains_db / 20.0)
    return gains.astype(np.float32)

def dt2_cassette(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    hiss_snr = float(opts.get('hiss_snr_db', 30.0))
    wow_depth_ms = float(opts.get('wow_depth_ms', 5.0))
    wow_rate = float(opts.get('wow_rate_hz', 1.0))
    eq_profile = opts.get('eq_profile', None)
    sat = float(opts.get('saturation', 0.05))
    if rng is None:
        rng = np.random.default_rng()

    wav0 = _ensure_tensor_audio(wav)
    C, T = wav0.shape
    # apply EQ via FFT-domain multiplication
    n_fft = 1 << (int(np.ceil(np.log2(T))) + 1)
    wav_np = wav0.numpy()
    out = np.zeros_like(wav_np)
    H = _tape_eq_freq_response(n_fft, sr, profile=eq_profile)
    for c in range(C):
        X = np.fft.rfft(wav_np[c], n=n_fft)
        Y = X * H
        y = np.fft.irfft(Y, n=n_fft)[:T]
        out[c] = y
    out_t = torch.from_numpy(out.astype(np.float32))
    # add hiss (high-frequency-ish noise)
    # generate white noise and highpass by multiplying FFT with rising gain
    # simpler: generate noise and highpass via spectral shaping
    noise = rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
    # shape noise spectrum to emphasize 2k-10k region
    Nfft = 1 << (int(np.ceil(np.log2(T))) + 1)
    freq = np.linspace(0, sr/2, Nfft//2 + 1)
    noise_shape = 1.0 / (1.0 + (freq / 2000.0)**2)  # decaying with freq (tunable)
    # apply shaping per channel
    shaped_noise = np.zeros_like(out)
    for c in range(C):
        Nspec = np.fft.rfft(noise[c], n=Nfft) * noise_shape
        ntime = np.fft.irfft(Nspec, n=Nfft)[:T]
        shaped_noise[c] = ntime
    out_noisy = out_t + torch.from_numpy(shaped_noise.astype(np.float32))
    # scale noise to desired hiss_snr relative to signal
    out_noisy = add_awgn_snr(out_noisy - out_t, -np.inf, rng=rng)  # noop, placeholder - we will scale explicitly
    # Better: compute signal power and scale the shaped_noise to meet hiss_snr
    sig_pow = _power_per_channel(out_t).numpy()
    noise_pow = (sig_pow / (10.0 ** (hiss_snr / 10.0)))
    shaped_noise_t = torch.from_numpy(shaped_noise.astype(np.float32))
    scaled_noise = torch.zeros_like(out_t)
    for c in range(C):
        cur_noise = shaped_noise_t[c]
        cur_pow = float(np.mean(cur_noise.cpu().numpy() ** 2) + 1e-20)
        target_pow = noise_pow[c]
        scale = np.sqrt(target_pow / cur_pow)
        scaled_noise[c] = cur_noise * float(scale)
    out_noisy = out_t + scaled_noise
    # wow/flutter: simple time-varying fractional delay implemented by resampling via np.interp
    # produce time index shift array
    if wow_depth_ms > 0 and wow_rate > 0:
        max_delay = wow_depth_ms / 1000.0
        t = np.arange(T) / float(sr)
        mod = max_delay * np.sin(2 * np.pi * wow_rate * t)  # seconds
        # target time positions
        idx = np.arange(T) / float(sr) + mod
        idx_samples = idx * sr
        # interpolation per channel (on CPU numpy)
        out_np = out_noisy.numpy()
        fluttered = np.zeros_like(out_np)
        for c in range(C):
            fluttered[c] = np.interp(np.arange(T), idx_samples, out_np[c], left=out_np[c, 0], right=out_np[c, -1])
        out_final = torch.from_numpy(fluttered.astype(np.float32))
    else:
        out_final = out_noisy
    # soft saturation
    if sat > 0:
        out_final = out_final / (out_final.abs().max() + 1e-20)
        out_final = torch.tanh(out_final * (1.0 + sat * 10.0)) * 0.9  # small nonlinearity
    meta = {'dt': 'DT2', 'params': {'hiss_snr_db': hiss_snr, 'wow_depth_ms': wow_depth_ms, 'wow_rate_hz': wow_rate, 'saturation': sat}}
    return out_final, meta

def _generate_crackle_events(T: int, sr: int, density_per_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Generate sparse crackle impulse train (amplitudes and times)."""
    n_events = int(max(1, int(density_per_sec * (T / sr))))
    # uniformly random times
    times = rng.integers(low=0, high=T, size=n_events)
    amps = rng.uniform(low=0.01, high=0.5, size=n_events).astype(np.float32)
    impulse = np.zeros(T, dtype=np.float32)
    for t, a in zip(times, amps):
        impulse[t] += a
    # optional small smoothing to make each crackle a short impulse
    win_len = min(256, int(0.01 * sr))  # up to 10ms
    if win_len > 1:
        w = np.hanning(win_len).astype(np.float32)
        impulse = np.convolve(impulse, w, mode='same')
    return impulse

def dt3_vinyl(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    crackle_density = float(opts.get('crackle_density', 1.5))
    crackle_amp = float(opts.get('crackle_amp', 0.2))
    rumble_snr = float(opts.get('rumble_snr_db', 30.0))
    hf_cut = float(opts.get('hf_rolloff_freq', 12000.0))
    if rng is None:
        rng = np.random.default_rng()
    wav0 = _ensure_tensor_audio(wav)
    C, T = wav0.shape
    # HF rolloff via FFT multiplication
    n_fft = 1 << (int(np.ceil(np.log2(T))) + 1)
    freqs = np.linspace(0, sr/2, n_fft//2 + 1)
    # design gentle rolloff above hf_cut
    roll = 1.0 / (1.0 + ((freqs - hf_cut) / (2000.0 + 1e-9))**2)
    roll[freqs <= hf_cut] = 1.0
    out = np.zeros_like(wav0.numpy())
    for c in range(C):
        X = np.fft.rfft(wav0.numpy()[c], n=n_fft)
        Y = X * roll
        y = np.fft.irfft(Y, n=n_fft)[:T]
        out[c] = y
    out_t = torch.from_numpy(out.astype(np.float32))
    # add rumble: low-frequency noise (filtered noise)
    rumble_noise = rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
    # lowpass shape
    freq = np.linspace(0, sr/2, n_fft//2 + 1)
    lp_shape = np.exp(-freq / 200.0)  # strong lowpass
    rumble_shaped = np.zeros_like(out)
    for c in range(C):
        Nspec = np.fft.rfft(rumble_noise[c], n=n_fft) * lp_shape
        ntime = np.fft.irfft(Nspec, n=n_fft)[:T]
        rumble_shaped[c] = ntime
    # scale rumble to target SNR relative to signal
    sig_pow = _power_per_channel(out_t).numpy()
    noise_pow = (sig_pow / (10.0 ** (rumble_snr / 10.0)))
    rumble_t = torch.from_numpy(rumble_shaped.astype(np.float32))
    rumble_scaled = torch.zeros_like(out_t)
    for c in range(C):
        cur = rumble_t[c]
        cur_pow = float(np.mean(cur.cpu().numpy() ** 2) + 1e-20)
        target = noise_pow[c]
        scale = np.sqrt(target / cur_pow)
        rumble_scaled[c] = cur * float(scale)
    # add crackle impulses
    impulses = _generate_crackle_events(T, sr, crackle_density, rng) * crackle_amp
    crackle = np.stack([impulses for _ in range(C)], axis=0)
    out_final = out_t + rumble_scaled + torch.from_numpy(crackle.astype(np.float32))
    meta = {'dt': 'DT3', 'params': {'crackle_density': crackle_density, 'crackle_amp': crackle_amp, 'rumble_snr_db': rumble_snr, 'hf_cut': hf_cut}}
    return out_final, meta

def _make_simple_ir(sr: int, reverb_time: float = 0.3, early_reflections: int = 3, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a simple exponential-decay IR with a few early taps (approximation of room IR)."""
    if rng is None:
        rng = np.random.default_rng()
    L = int(min(sr * (reverb_time + 0.05), sr * 2))  # limit IR length
    t = np.arange(L) / float(sr)
    # early reflections at some ms offsets
    ir = np.zeros(L, dtype=np.float32)
    early_delays = rng.integers(low=0, high=int(0.05 * sr), size=early_reflections)
    early_gains = rng.uniform(0.2, 0.7, size=early_reflections)
    for d, g in zip(early_delays, early_gains):
        if d < L:
            ir[d] += float(g)
    # main exponential tail
    tail = np.exp(-t / (reverb_time + 1e-9)).astype(np.float32)
    ir += tail * 0.3
    # normalize
    ir /= (np.sum(np.abs(ir)) + 1e-20)
    return ir

def dt4_live(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    ambient_snr = float(opts.get('ambient_snr_db', 20.0))
    phone_band = opts.get('phone_band', (300.0, 3400.0))
    ir = opts.get('ir', None)
    reverb_time = float(opts.get('reverb_time', 0.35))
    ambient_noise = opts.get('ambient_noise', None)
    agc_target = float(opts.get('agc_target', -12.0))

    if rng is None:
        rng = np.random.default_rng()

    try:
        wav0 = _ensure_tensor_audio(wav)

        original_rms = torch.sqrt(torch.mean(wav0.pow(2)))
        original_peak = torch.max(torch.abs(wav0))

        if ir is None:
            ir = _make_simple_ir(sr, reverb_time=reverb_time, rng=rng)
        wav_reverb = convolve_ir(wav0, ir)

        C, T = wav_reverb.shape
        n_fft = 2 ** int(np.ceil(np.log2(T)) + 1)
        freqs = np.linspace(0, sr/2, n_fft//2 + 1)

        mask = np.ones_like(freqs)

        low_freq, high_freq = phone_band
        low_rolloff_width = 100.0  # Hz
        high_rolloff_width = 500.0  # Hz

        low_mask = freqs < low_freq
        low_transition = (freqs >= (low_freq - low_rolloff_width)) & (freqs < low_freq)
        mask[low_mask] = 0.01
        mask[low_transition] = 0.01 + 0.99 * (freqs[low_transition] - (low_freq - low_rolloff_width)) / low_rolloff_width

        high_mask = freqs > high_freq
        high_transition = (freqs > high_freq) & (freqs <= (high_freq + high_rolloff_width))
        mask[high_mask] = 0.05
        mask[high_transition] = 1.0 - 0.95 * (freqs[high_transition] - high_freq) / high_rolloff_width

        out = np.zeros_like(wav_reverb.numpy())
        for c in range(C):
            X = np.fft.rfft(wav_reverb.numpy()[c], n=n_fft)
            X = X * mask
            y = np.fft.irfft(X, n=n_fft)[:T]
            out[c] = y

        out_t = torch.from_numpy(out.astype(np.float32))

        current_rms = torch.sqrt(torch.mean(out_t.pow(2))) + 1e-20
        if current_rms > 0:
            volume_compensation = (original_rms * 0.7) / current_rms
            out_t = out_t * volume_compensation

        if ambient_noise is not None:
            an = ambient_noise
            if isinstance(an, torch.Tensor):
                an = an.numpy()

            if an.shape[0] < C:
                an = np.tile(an[0:1], (C, 1))
            elif an.shape[0] > C:
                an = an[:C]

            if an.shape[1] >= T:
                start = int(rng.integers(0, an.shape[1] - T + 1))
                noise_seg = an[:, start:start+T]
            else:
                reps = int(np.ceil(T / an.shape[1]))
                noise_seg = np.tile(an, (1, reps))[:, :T]

            noise_t = torch.from_numpy(noise_seg.astype(np.float32))

            sig_pow = _safe_power_per_channel(out_t).numpy()

            ambient_snr = np.clip(ambient_snr, -50, 50)
            snr_linear = 10.0 ** (ambient_snr / 10.0)
            snr_linear = max(snr_linear, 1e-20)
            noise_pow = sig_pow / snr_linear
            noise_pow = np.maximum(noise_pow, 1e-20)

            scaled_noise = torch.zeros_like(out_t)
            for c in range(C):
                cur = noise_t[c].float()
                cur_pow = float(cur.pow(2).mean().item() + 1e-20)
                scale = float(np.sqrt(noise_pow[c] / cur_pow))
                scaled_noise[c] = cur * scale

            out_final = out_t + scaled_noise
        else:
            noise = rng.normal(0.0, 1.0, size=out_t.numpy().shape).astype(np.float32)

            Nfft = 2 ** int(np.ceil(np.log2(T)) + 1)
            freq = np.linspace(0, sr/2, Nfft//2 + 1)


            shape = 1.0 / (1.0 + (freq / 500.0)**1.2)
            shape = shape * (1.0 + 0.3 * np.exp(-(freq - 1000)**2 / (2 * 800**2)))

            shaped = np.zeros_like(noise)
            for c in range(C):
                Nspec = np.fft.rfft(noise[c], n=Nfft) * shape
                ntime = np.fft.irfft(Nspec, n=Nfft)[:T]
                shaped[c] = ntime

            shaped_t = torch.from_numpy(shaped.astype(np.float32))

            sig_pow = _safe_power_per_channel(out_t).numpy()

            ambient_snr = np.clip(ambient_snr, -50, 50)
            snr_linear = 10.0 ** (ambient_snr / 10.0)
            snr_linear = max(snr_linear, 1e-20)
            noise_pow = sig_pow / snr_linear
            noise_pow = np.maximum(noise_pow, 1e-20)

            scaled_noise = torch.zeros_like(out_t)
            for c in range(C):
                cur = shaped_t[c]
                cur_pow = float(cur.pow(2).mean().item() + 1e-20)
                scale = float(np.sqrt(noise_pow[c] / cur_pow))
                scaled_noise[c] = cur * scale

            out_final = out_t + scaled_noise

        threshold = 0.8
        ratio = 4.0

        peak_level = torch.max(torch.abs(out_final))
        if peak_level > threshold:
            over_threshold = torch.abs(out_final) > threshold
            compressed = torch.where(
                over_threshold,
                torch.sign(out_final) * (threshold + (torch.abs(out_final) - threshold) / ratio),
                out_final
            )
            out_final = compressed

        final_peak = torch.max(torch.abs(out_final))
        if final_peak > 0:
            target_peak = original_peak * 0.8
            if final_peak < target_peak * 0.1:
                gain = target_peak * 0.5 / final_peak
                out_final = out_final * gain
            elif final_peak > target_peak:
                gain = target_peak / final_peak
                out_final = out_final * gain

        out_final = torch.tanh(out_final * 1.1) * 0.95

        quantization_noise = torch.randn_like(out_final) * 1e-5
        out_final = out_final + quantization_noise

        meta = {
            'dt': 'DT4',
            'params': {
                'ambient_snr_db': ambient_snr,
                'phone_band': phone_band,
                'reverb_time': reverb_time,
                'volume_preserved': True,
                'original_peak': float(original_peak),
                'final_peak': float(torch.max(torch.abs(out_final)))
            }
        }

        return out_final, meta

    except Exception as e:
        print(f"DT4 processing failed: {e}")
        try:
            from scipy import signal
            wav0 = _ensure_tensor_audio(wav)

            nyquist = sr / 2
            low = phone_band[0] / nyquist
            high = min(phone_band[1] / nyquist, 0.95)

            if 0 < low < high < 1:
                b, a = signal.butter(4, [low, high], btype='band')
                wav_np = wav0.numpy()
                for c in range(wav_np.shape[0]):
                    wav_np[c] = signal.filtfilt(b, a, wav_np[c])
                wav_filtered = torch.from_numpy(wav_np.astype(np.float32))
            else:
                wav_filtered = wav0


            original_peak = torch.max(torch.abs(wav0))
            current_peak = torch.max(torch.abs(wav_filtered))
            if current_peak > 0:
                wav_filtered = wav_filtered * (original_peak * 0.8) / current_peak


            noise = torch.randn_like(wav_filtered) * 0.02
            result = wav_filtered + noise

            return result, {'dt': 'DT4', 'error': str(e), 'fallback': True, 'volume_preserved': True}
        except Exception as e2:
            return wav, {'dt': 'DT4', 'error': f'Both main and fallback failed: {e}, {e2}'}

def _safe_power_per_channel(wav: torch.Tensor) -> torch.Tensor:
    if wav.numel() == 0:
        return torch.tensor([1e-20])

    if torch.any(torch.isnan(wav)) or torch.any(torch.isinf(wav)):
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    power = torch.mean(wav.pow(2), dim=-1)
    return torch.clamp(power, min=1e-20)

def _apply_codec_torchaudio(wav: torch.Tensor, sr: int, fmt: str, bitrate: int) -> torch.Tensor:
    if not _HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio not available in this environment")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if _HAS_AUDIO_EFFECTOR:
            try:
                from torchaudio.io import AudioEffector

                if fmt == 'aac':
                    effector = AudioEffector(effect="aac", format="m4a")
                elif fmt == 'mp3':
                    effector = AudioEffector(effect="mp3")
                else:
                    raise RuntimeError(f"Unsupported format: {fmt}")

                effector.set_src_info(sample_rate=sr, num_channels=wav.shape[0])
                effector.set_dst_info(sample_rate=sr, num_channels=wav.shape[0], bit_rate=bitrate)

                return effector.apply(wav)

            except Exception as e:
                pass

        if ta_apply_codec is not None:
            try:
                compression = int(bitrate // 1000)
                params = {'format': fmt, 'compression': compression}
                out = ta_apply_codec(wav, sr, **params)
                return out
            except Exception:
                try:
                    params = {'format': fmt, 'bitrate': str(bitrate)}
                    out = ta_apply_codec(wav, sr, **params)
                    return out
                except Exception as e:
                    raise e

        raise RuntimeError("No codec method available")

def _apply_codec_ffmpeg(wav: torch.Tensor, sr: int, fmt: str, bitrate: int) -> torch.Tensor:
    import tempfile
    import subprocess
    import os

    C, T = wav.shape


    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
        tmp_in_name = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
        tmp_out_name = tmp_out.name

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _HAS_TORCHAUDIO:
                try:
                    torchaudio.save(tmp_in_name, wav, sr, encoding='PCM_F', bits_per_sample=32)
                except:
                    _save_wav_numpy(tmp_in_name, wav.numpy(), sr)
            else:
                _save_wav_numpy(tmp_in_name, wav.numpy(), sr)

        if fmt == 'mp3':
            codec_name = 'libmp3lame'
        elif fmt == 'aac':
            codec_name = 'aac'
        else:
            codec_name = fmt
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', tmp_in_name,
            '-c:a', codec_name,
            '-b:a', str(bitrate),
            '-ar', str(sr),
            '-f', 'wav',
            tmp_out_name
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _HAS_TORCHAUDIO:
                try:
                    out_wave, out_sr = torchaudio.load(tmp_out_name)
                except:
                    out_wave, out_sr = _load_wav_numpy(tmp_out_name)
            else:
                out_wave, out_sr = _load_wav_numpy(tmp_out_name)

        if out_sr != sr and _HAS_TORCHAUDIO:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out_wave = torchaudio.functional.resample(out_wave, out_sr, sr)

        return out_wave

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"FFmpeg processing failed: {e}")
    finally:
        try:
            os.unlink(tmp_in_name)
        except:
            pass
        try:
            os.unlink(tmp_out_name)
        except:
            pass

def _save_wav_numpy(path: str, wav_np: np.ndarray, sr: int):
    try:
        import soundfile as sf
        sf.write(path, wav_np.T, sr, subtype='FLOAT')
    except ImportError:
        raise RuntimeError("soundfile is required for FFmpeg fallback when torchaudio.save is not available")
    except Exception as e:
        raise RuntimeError(f"Failed to save WAV file: {e}")

def _load_wav_numpy(path: str) -> Tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        if len(audio.shape) == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T
        return torch.from_numpy(audio.astype(np.float32)), sr
    except ImportError:
        raise RuntimeError("soundfile is required for FFmpeg fallback when torchaudio.load is not available")
    except Exception as e:
        raise RuntimeError(f"Failed to load WAV file: {e}")

def _apply_codec_wrapper_robust(wav: torch.Tensor, sr: int, fmt: str, bitrate: int, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options) if options else {}
    wav0 = _ensure_tensor_audio(wav)
    original_length = wav0.shape[1]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _HAS_TORCHAUDIO and ta_apply_codec is not None:
                out = _apply_codec_torchaudio(wav0, sr, fmt, bitrate)

                if out.shape[1] >= original_length:
                    out = out[:, :original_length]
                else:
                    pad_length = original_length - out.shape[1]
                    out = torch.nn.functional.pad(out, (0, pad_length))

                meta = {'dt': f'DT_{fmt}_{bitrate//1000}k', 'params': {'format': fmt, 'bitrate': bitrate, 'method': 'torchaudio'}}
                return out, meta
    except Exception as e:
        pass

    try:
        out = _apply_codec_ffmpeg(wav0, sr, fmt, bitrate)

        if out.shape[1] >= original_length:
            out = out[:, :original_length]
        else:
            pad_length = original_length - out.shape[1]
            out = torch.nn.functional.pad(out, (0, pad_length))

        meta = {'dt': f'DT_{fmt}_{bitrate//1000}k', 'params': {'format': fmt, 'bitrate': bitrate, 'method': 'ffmpeg'}}
        return out, meta

    except Exception as e:
        pass

    try:
        out = _robust_codec_simulation(wav0, fmt, bitrate)

        if out.shape[1] != original_length:
            if out.shape[1] > original_length:
                out = out[:, :original_length]
            else:
                pad_length = original_length - out.shape[1]
                out = torch.nn.functional.pad(out, (0, pad_length))

        meta = {'dt': f'DT_{fmt}_{bitrate//1000}k', 'params': {'format': fmt, 'bitrate': bitrate, 'method': 'simulation'}}
        return out, meta

    except Exception as e:
        pass

    try:
        max_val = torch.max(torch.abs(wav0)) + 1e-20
        if max_val > 0:
            wav_norm = wav0 / max_val
            levels = 32 if bitrate < 100000 else 64
            quantized = torch.round(wav_norm * levels) / levels
            result = quantized * max_val * 0.95
        else:
            result = wav0

        meta = {'dt': f'DT_{fmt}_{bitrate//1000}k', 'params': {'format': fmt, 'bitrate': bitrate, 'method': 'minimal_quantization'}}
        return result, meta

    except Exception as e:
        meta = {'dt': f'DT_{fmt}_{bitrate//1000}k', 'error': f'All methods failed: {e}', 'method': 'passthrough'}
        return wav0, meta


def _robust_codec_simulation(wav: torch.Tensor, fmt: str, bitrate: int) -> torch.Tensor:
    """Fallback codec approximation when runtime encoders are unavailable.

    The goal is not bit-accurate codec emulation, but a stable degradation that
    roughly follows bitrate severity and keeps shape/dtype consistent.
    """
    wav0 = _ensure_tensor_audio(wav)
    max_val = torch.max(torch.abs(wav0)).clamp(min=1e-12)
    wav_norm = (wav0 / max_val).clamp(-1.0, 1.0)

    # Lower bitrate -> stronger quantization.
    # 24k maps to ~32 levels, 128k maps to ~128 levels.
    kbps = max(1.0, float(bitrate) / 1000.0)
    levels = int(np.clip(16.0 + 0.875 * kbps, 32.0, 128.0))
    degraded = torch.round(wav_norm * levels) / levels

    # Add tiny deterministic-ish noise floor scaled by bitrate to mimic codec residue.
    noise_scale = float(np.interp(kbps, [24.0, 128.0], [0.0030, 0.0008]))
    degraded = (degraded + torch.randn_like(degraded) * noise_scale).clamp(-1.0, 1.0)
    return (degraded * max_val).to(wav0.dtype)

def dt5_aac_64k(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """DT5: AAC 64kbps - """
    return _apply_codec_wrapper_robust(wav, sr, fmt='aac', bitrate=64000, options=options)

def dt6_mp3_64k(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """DT6: MP3 64kbps - """
    return _apply_codec_wrapper_robust(wav, sr, fmt='mp3', bitrate=64000, options=options)

def dt7_aac_128k(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """DT7: AAC 128kbps - """
    return _apply_codec_wrapper_robust(wav, sr, fmt='aac', bitrate=128000, options=options)

def dt8_mp3_128k(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """DT8: MP3 128kbps - """
    return _apply_codec_wrapper_robust(wav, sr, fmt='mp3', bitrate=128000, options=options)

def _resample_numpy(wave_np: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampler (channels x T). Use torchaudio if available for better quality."""
    if src_sr == dst_sr:
        return wave_np
    if _HAS_TORCHAUDIO:
        # torchaudio expects [C, T] torch tensor
        w = torch.from_numpy(wave_np).float()
        w_r = torchaudio.functional.resample(w, orig_freq=src_sr, new_freq=dst_sr)
        return w_r.numpy()
    # numpy linear interpolation per channel
    C, T = wave_np.shape
    t_src = np.arange(T) / float(src_sr)
    T_dst = int(np.round(wave_np.shape[1] * (dst_sr / src_sr)))
    t_dst = np.arange(T_dst) / float(dst_sr)
    out = np.zeros((C, T_dst), dtype=np.float32)
    for c in range(C):
        out[c] = np.interp(t_dst, t_src, wave_np[c])
    return out

def _mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    # expects x in [-1,1]
    x_clamped = np.clip(x, -1.0, 1.0)
    encoded = np.sign(x_clamped) * np.log1p(mu * np.abs(x_clamped)) / np.log1p(mu)
    return encoded

def _mu_law_decode(y: np.ndarray, mu: int = 255) -> np.ndarray:
    decoded = np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)
    return decoded

def _quantize_uniform(x: np.ndarray, n_levels: int) -> np.ndarray:
    # Uniform quantization between -1 and 1
    x = np.clip(x, -1.0, 1.0)
    levels = np.linspace(-1.0, 1.0, n_levels)
    idx = np.round((x - (-1.0)) / (2.0 / (n_levels - 1))).astype(np.int32)
    idx = np.clip(idx, 0, n_levels - 1)
    return levels[idx]

def _simulate_neural_codec(wav: torch.Tensor, sr: int, target_sr: int, bitrate_kbps: float, mode: str = "simulate") -> torch.Tensor:

    wav_np = wav.numpy()
    # resample to target rate
    res = _resample_numpy(wav_np, src_sr=sr, dst_sr=target_sr)
    # normalize to -1..1 (safeguard)
    peak = np.max(np.abs(res)) + 1e-20
    res = res / peak

    # EnCodec uses residual vector quantization (RVQ) with multiple codebooks
    # For neural codecs, we estimate more realistically based on frame size and bandwidth
    # Bitrate formula: bitrate = num_codes * bits_per_code * sr / frame_samples
    # EnCodec typically uses 4 codebooks with adaptive bandwidth allocation

    # More realistic quantization levels based on EnCodec architecture
    if bitrate_kbps >= 128:
        n_levels = 256  # Full precision
        smoothing = 1.0
    elif bitrate_kbps >= 64:
        n_levels = 180   # High quality
        smoothing = 0.9
    elif bitrate_kbps >= 32:
        n_levels = 120   # Medium-high quality
        smoothing = 0.8
    elif bitrate_kbps >= 24:
        n_levels = 80    # Medium quality
        smoothing = 0.7
    elif bitrate_kbps >= 16:
        n_levels = 48    # Medium-low quality
        smoothing = 0.6
    elif bitrate_kbps >= 12:
        n_levels = 32    # Lower quality
        smoothing = 0.5
    elif bitrate_kbps >= 6:
        # DT11: 6kbps - moderate compression artifacts
        n_levels = 24    # Increased from 8 to 24 for better quality
        smoothing = 0.4
    elif bitrate_kbps >= 3:
        # DT12: 3kbps - heavy compression artifacts
        n_levels = 16    # Increased from 4 to 16 for better quality
        smoothing = 0.3
    else:
        n_levels = 8
        smoothing = 0.2

    # apply mu-law then uniform quantization in companded domain to mimic discrete latents
    companded = _mu_law_encode(res, mu=255)
    quant = _quantize_uniform(companded, n_levels)

    # Use adaptive smoothing to emulate decoder interpolation
    # Higher smoothing for lower bitrates
    kernel_len = max(3, int(5 + (1 - smoothing) * 10))  # 5-15 samples
    kernel = np.hanning(kernel_len).astype(np.float32)
    kernel = kernel / (kernel.sum() + 1e-20)
    smoothed = np.array([np.convolve(quant[c], kernel, mode='same') for c in range(quant.shape[0])])

    # decode mu-law
    decoded = _mu_law_decode(smoothed, mu=255)

    # Add subtle quantization noise to more realistically represent neural codec artifacts
    # Rather than aggressive quantization, add noise proportional to bitrate
    if bitrate_kbps <= 12:
        noise_level = (1 - smoothing) * 0.01  # Small noise floor
        noise = np.random.normal(0, noise_level, decoded.shape).astype(np.float32)
        decoded = decoded + noise

    # restore original peak amplitude
    decoded = decoded * peak
    # resample back to original sr
    out = _resample_numpy(decoded, src_sr=target_sr, dst_sr=sr)
    return torch.from_numpy(out.astype(np.float32))

def _apply_neural_codec_wrapper(wav: torch.Tensor, sr: int, codec: str, target_sr: int, bitrate_kbps: float, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options) if options else {}
    wav0 = _ensure_tensor_audio(wav)

    if codec == 'encodec' and _ENCODEC_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = EncodecModel.encodec_model_48khz() if target_sr == 48000 else EncodecModel.encodec_model_24khz()
                model.set_target_bandwidth(bitrate_kbps)

                with torch.no_grad():
                    wav_t = wav0.unsqueeze(0) if wav0.dim() == 2 else wav0
                    encoded_frames = model.encode(wav_t)
                    decoded = model.decode(encoded_frames)
                    decoded = decoded.squeeze(0) if decoded.dim() == 3 and decoded.shape[0] == 1 else decoded

                    if target_sr != sr and _HAS_TORCHAUDIO:
                        decoded = torchaudio.functional.resample(decoded, target_sr, sr)

                    out = decoded.cpu().float()
                    meta = {'dt': f'{codec}_native', 'params': {'codec': codec, 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps, 'used': 'encodec_native'}}
                    return out, meta
        except Exception as e:
            print(f"Native Encodec failed: {e}")

    try:
        out_sim = _simulate_neural_codec(wav0, sr, target_sr=target_sr, bitrate_kbps=bitrate_kbps, mode="simulate")

        if out_sim.shape[1] != wav0.shape[1]:
            if out_sim.shape[1] > wav0.shape[1]:
                out_sim = out_sim[:, :wav0.shape[1]]
            else:
                pad_len = wav0.shape[1] - out_sim.shape[1]
                out_sim = torch.nn.functional.pad(out_sim, (0, pad_len))

        meta = {'dt': f'{codec}_simulated', 'params': {'codec': codec, 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps, 'simulate_levels_hint': options.get('simulate_levels') if options else None}}
        return out_sim, meta

    except Exception as e:
        print(f"Neural codec simulation failed: {e}")
        meta = {'dt': f'{codec}_failed', 'error': str(e), 'used': 'passthrough'}
        return wav0, meta

    # Fallback simulation
    out_sim = _simulate_neural_codec(wav0, sr, target_sr=target_sr, bitrate_kbps=bitrate_kbps, mode=opts.get('simulate_mode', 'default'))
    meta = {'dt': f'{codec}_simulated', 'params': {'codec': codec, 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps, 'simulate_levels_hint': options.get('simulate_levels') if options else None if options else None}}
    return out_sim, meta

def dt9_dac_22k(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """
    Descript Audio Codec (DAC) at 22 kHz (DT9).
    Tries to use a DAC binding if available, otherwise simulates neural-quantization artifacts.
    """
    opts = _copy_options(options)
    target_sr = int(opts.get('target_sr', 22050))
    # DAC is typically variable bitrate; we just flag target rate here. Use moderate bitrate mapping.
    bitrate_kbps = float(opts.get('bitrate_kbps', 64.0))
    out, meta = _apply_neural_codec_wrapper(wav, sr, codec='dac', target_sr=target_sr, bitrate_kbps=bitrate_kbps, options=opts)
    meta_final = {'dt': 'DT9', 'params': {'codec': 'dac', 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps}, **meta}
    return out, meta_final

def dt10_dac_44k(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """
    Descript Audio Codec (DAC) at 44 kHz (DT10).
    """
    opts = _copy_options(options)
    target_sr = int(opts.get('target_sr', 44100))
    bitrate_kbps = float(opts.get('bitrate_kbps', 128.0))
    out, meta = _apply_neural_codec_wrapper(wav, sr, codec='dac', target_sr=target_sr, bitrate_kbps=bitrate_kbps, options=opts)
    meta_final = {'dt': 'DT10', 'params': {'codec': 'dac', 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps}, **meta}
    return out, meta_final

def dt11_encodec_48k_6kbps(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """
    Encodec (48 kHz) at 6 kbps (DT11).
    Tries to use encodec library if present; otherwise falls back to simulation approximating latent quantization.
    """
    opts = _copy_options(options)
    target_sr = 48000
    bitrate_kbps = float(opts.get('bitrate_kbps', 6.0))
    out, meta = _apply_neural_codec_wrapper(wav, sr, codec='encodec', target_sr=target_sr, bitrate_kbps=bitrate_kbps, options=opts)
    meta_final = {'dt': 'DT11', 'params': {'codec': 'encodec', 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps}, **meta}
    return out, meta_final

def dt12_encodec_48k_3kbps(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng=None) -> Tuple[torch.Tensor, Dict]:
    """
    Encodec (48 kHz) at 3 kbps (DT12).
    """
    opts = _copy_options(options)
    target_sr = 48000
    bitrate_kbps = float(opts.get('bitrate_kbps', 3.0))
    out, meta = _apply_neural_codec_wrapper(wav, sr, codec='encodec', target_sr=target_sr, bitrate_kbps=bitrate_kbps, options=opts)
    meta_final = {'dt': 'DT12', 'params': {'codec': 'encodec', 'target_sr': target_sr, 'bitrate_kbps': bitrate_kbps}, **meta}
    return out, meta_final

_DT_REGISTRY = {
    'DT1': dt1_radio,
    'DT2': dt2_cassette,
    'DT3': dt3_vinyl,
    'DT4': dt4_live,
    'DT5': dt5_aac_64k,
    'DT6': dt6_mp3_64k,
    'DT7': dt7_aac_128k,
    'DT8': dt8_mp3_128k,
    'DT9': dt9_dac_22k,
    'DT10': dt10_dac_44k,
    'DT11': dt11_encodec_48k_6kbps,
    'DT12': dt12_encodec_48k_3kbps,
}

# External degradation registry for custom integrations (e.g., Apollo-style codecs).
_CUSTOM_DT_REGISTRY = {}


def _available_degradation_keys() -> list:
    return list(_DT_REGISTRY.keys()) + list(_CUSTOM_DT_REGISTRY.keys())


def register_custom_degradation(name: str, fn, overwrite: bool = False) -> None:
    """
    Register a custom degradation callable with signature:
        fn(wav: torch.Tensor, sr: int, options: Optional[Dict], rng: Optional[np.random.Generator])
            -> Tuple[torch.Tensor, Dict]

    Args:
        name: degradation key, e.g. "APOLLO_MP3"
        fn: callable degradation function
        overwrite: if False, raise when key already exists
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not overwrite and (name in _DT_REGISTRY or name in _CUSTOM_DT_REGISTRY):
        raise ValueError(f"Degradation key '{name}' already exists.")
    _CUSTOM_DT_REGISTRY[name] = fn


def unregister_custom_degradation(name: str) -> None:
    """Unregister a previously registered custom degradation."""
    _CUSTOM_DT_REGISTRY.pop(name, None)


def _vf_check_augment_available() -> None:
    if not _HAS_AUGMENT:
        raise RuntimeError(
            "VoiceFixer-style effects require python-audio-augment. "
            "Install package 'augment' to use tempo/speed/pitch/eq/reverb style effects."
        )


def _vf_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(low) if abs(float(high) - float(low)) < 1e-12 else float(rng.uniform(low, high))


def _vf_select_up_down(rng: np.random.Generator, up_range: Tuple[float, float], down_range: Tuple[float, float]) -> float:
    if bool(rng.integers(0, 2)):
        return _vf_uniform(rng, up_range[0], up_range[1])
    return _vf_uniform(rng, down_range[0], down_range[1])


def _vf_smooth(signal_1d: np.ndarray, smooth_center: int, smooth_samples: int = 50, interval: int = 4) -> np.ndarray:
    # Mirrors voicefixer_main/tools/others/audio_op.py::smooth behavior.
    x = signal_1d
    if smooth_center < smooth_samples or x.shape[0] - smooth_center < smooth_samples:
        return x
    samples = x[smooth_center - smooth_samples:smooth_center + smooth_samples]
    length = samples.shape[0]
    xnew = np.arange(length)
    samples_sub = samples[0::interval]
    x_sub = xnew[0::interval]
    xnew = xnew[:x_sub[-1] + 1]
    if scipy_interp1d is not None and x_sub.shape[0] >= 4:
        smoothed = scipy_interp1d(x_sub, samples_sub, kind="cubic")(xnew)
    else:
        smoothed = np.interp(xnew, x_sub, samples_sub)
    start = smooth_center - smooth_samples
    x[start:start + smoothed.shape[0]] = smoothed
    return x


def _vf_quantification(samples: np.ndarray, bins: int) -> np.ndarray:
    level = 2 ** int(bins)
    max_val = float(np.max(np.abs(samples)) + 1e-20)
    samples_n = samples / max_val
    space = np.linspace(
        start=-(1.0 / level) * level + (1.0 / level),
        stop=(1.0 / level) * level - (1.0 / level),
        num=level,
        endpoint=True,
    )
    digitalized_samples = np.digitize(samples_n, space, right=True) - 1
    digitalized_samples = np.clip(digitalized_samples, 0, level - 1)
    digitalized_samples = space[digitalized_samples]
    return (digitalized_samples * max_val).astype(np.float32)


def _vf_reverb_rir_channel(frames: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if scipy_signal is None:
        raise RuntimeError("scipy is required for VoiceFixer reverb_rir effect.")
    orig_len = frames.shape[0]
    y = scipy_signal.convolve(np.squeeze(frames), np.squeeze(rir))
    actlev = float(np.max(np.abs(y)) + 1e-20)
    if actlev > 0.99:
        y = (y / actlev) * 0.98
    y = y[:orig_len]
    return y.astype(np.float32)


def _vf_apply_per_channel(wav: torch.Tensor, channel_fn) -> torch.Tensor:
    wav0 = _ensure_tensor_audio(wav)
    out = []
    for c in range(wav0.shape[0]):
        x = wav0[c].detach().cpu().numpy().astype(np.float32)
        y = channel_fn(x)
        if y.shape[0] < x.shape[0]:
            y = np.pad(y, (0, x.shape[0] - y.shape[0]), mode="constant")
        out.append(y.astype(np.float32))
    return torch.from_numpy(np.stack(out, axis=0))


def _vf_apply_effect_chain(wav: torch.Tensor, sr: int, build_chain_fn) -> torch.Tensor:
    _vf_check_augment_available()

    def _per_ch(x: np.ndarray) -> np.ndarray:
        chain = build_chain_fn(x)
        x_t = torch.tensor(x, dtype=torch.float32)
        y = chain.apply(x_t, src_info={"rate": sr}, target_info={"rate": sr})
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        return np.asarray(y, dtype=np.float32).squeeze()

    return _vf_apply_per_channel(wav, _per_ch)


def vf_tempo(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    speed = float(opts.get("speed", _vf_select_up_down(rng, (1.1, 1.6), (0.7, 0.95))))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().tempo(speed))
    return out, {"dt": "TEMPO", "params": {"speed": speed}}


def vf_speed(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    speed = float(opts.get("speed", _vf_select_up_down(rng, (1.1, 1.6), (0.7, 0.95))))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().speed(speed))
    return out, {"dt": "SPEED", "params": {"speed": speed}}


def vf_fade(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    fade_in_portion = float(opts.get("fade_in_portion", _vf_uniform(rng, 0.05, 0.1)))
    fade_out_portion = float(opts.get("fade_out_portion", _vf_uniform(rng, 0.05, 0.1)))

    def _build(x: np.ndarray):
        audio_len = float(x.shape[0]) / float(sr)
        return augment.EffectChain().fade(float(fade_in_portion * audio_len), float(audio_len), float(audio_len * fade_out_portion))

    out = _vf_apply_effect_chain(wav, sr, _build)
    return out, {"dt": "FADE", "params": {"fade_in_portion": fade_in_portion, "fade_out_portion": fade_out_portion}}


def vf_pitch(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    pitch = float(opts.get("pitch", _vf_select_up_down(rng, (100.0, 350.0), (-350.0, -100.0))))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().pitch(pitch).rate(sr))
    return out, {"dt": "PITCH", "params": {"pitch": pitch}}


def vf_treble(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    level = float(opts.get("level", _vf_uniform(rng, 3.0, 20.0)))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().treble(level))
    return out, {"dt": "TREBLE", "params": {"level": level}}


def vf_bass(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    level = float(opts.get("level", _vf_uniform(rng, 3.0, 20.0)))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().bass(level))
    return out, {"dt": "BASS", "params": {"level": level}}


def vf_tremolo(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    level = float(opts.get("level", _vf_uniform(rng, 5.0, 50.0)))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().tremolo(level))
    return out, {"dt": "TREMOLO", "params": {"level": level}}


def vf_clip(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    louder_time = float(opts.get("louder_time", _vf_uniform(rng, 1.5, 10.0)))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().clip(float(1.0 / louder_time)))
    return out, {"dt": "CLIP", "params": {"louder_time": louder_time}}


def vf_reverse(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    _ = options
    _ = rng
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().reverse())
    return out, {"dt": "REVERSE", "params": {}}


def vf_reverb_freeverb(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    reverberance = float(opts.get("reverb_level", _vf_uniform(rng, 0.0, 100.0)))
    dumping = float(opts.get("dumping_factor", _vf_uniform(rng, 0.0, 100.0)))
    room_size = float(opts.get("room_size", _vf_uniform(rng, 0.0, 100.0)))
    out = _vf_apply_effect_chain(
        wav,
        sr,
        lambda _x: augment.EffectChain().reverb(reverberance, dumping, room_size).channels(1),
    )
    return out, {"dt": "REVERB_FREEVERB", "params": {"reverb_level": reverberance, "dumping_factor": dumping, "room_size": room_size}}


def vf_reverb_rir(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    rir = opts.get("rir", None)
    rir_path = opts.get("rir_path", None)
    rir_root = opts.get("rir_root", None)

    if rir is None and rir_path is not None:
        rir = np.load(rir_path)
    if rir is None and rir_root is not None:
        rir_files = [os.path.join(rir_root, f) for f in os.listdir(rir_root)]
        if len(rir_files) == 0:
            raise RuntimeError(f"No rir file found in {rir_root}")
        rir = np.load(rir_files[int(rng.integers(0, len(rir_files)))])
    used_fallback = False
    if rir is None:
        # Fallback: auto-synthesize a simple room impulse response so reverb_rir
        # can run without external RIR files.
        reverb_time = float(opts.get("reverb_time", 0.35))
        early_reflections = int(opts.get("early_reflections", 3))
        rir = _make_simple_ir(sr=sr, reverb_time=reverb_time, early_reflections=early_reflections, rng=rng)
        used_fallback = True

    out = _vf_apply_per_channel(wav, lambda x: _vf_reverb_rir_channel(x, np.asarray(rir, dtype=np.float32)))
    return out, {
        "dt": "REVERB_RIR",
        "params": {
            "rir_provided": not used_fallback,
            "fallback_synthetic_rir": used_fallback,
        },
    }


def vf_low_pass(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    low_pass_hz = float(opts.get("low_pass_hz", _vf_uniform(rng, 3000.0, 7000.0)))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().lowpass(low_pass_hz))
    return out, {"dt": "LOW_PASS", "params": {"low_pass_hz": low_pass_hz}}


def vf_high_pass(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    high_pass_hz = float(opts.get("high_pass_hz", _vf_uniform(rng, 500.0, 2000.0)))
    out = _vf_apply_effect_chain(wav, sr, lambda _x: augment.EffectChain().highpass(high_pass_hz))
    return out, {"dt": "HIGH_PASS", "params": {"high_pass_hz": high_pass_hz}}


def vf_time_dropout(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    max_segment = float(opts.get("max_segment", 0.2))
    drop_range = opts.get("drop_range", [0.0, 1.0])
    start = float(opts.get("start", _vf_uniform(rng, float(drop_range[0]), float(drop_range[1]) - max_segment)))
    trunk_length = float(opts.get("trunk_length", _vf_uniform(rng, 0.0, max_segment)))

    def _per_ch(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        length = y.shape[0]
        s_idx = int(length * start)
        e_idx = int(length * start + length * trunk_length)
        s_idx = max(0, min(length, s_idx))
        e_idx = max(s_idx, min(length, e_idx))
        y[s_idx:e_idx] = np.zeros_like(y[s_idx:e_idx])
        y = _vf_smooth(y, smooth_center=s_idx)
        y = _vf_smooth(y, smooth_center=e_idx)
        return y.astype(np.float32)

    out = _vf_apply_per_channel(wav, _per_ch)
    return out, {"dt": "TIME_DROPOUT", "params": {"start": start, "trunk_length": trunk_length, "max_segment": max_segment}}


def vf_quant(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    opts = _copy_options(options)
    if rng is None:
        rng = np.random.default_rng()
    bins = int(opts.get("bins", int(np.round(_vf_uniform(rng, 3.0, 12.0)))))
    out = _vf_apply_per_channel(wav, lambda x: _vf_quantification(x, bins=bins))
    return out, {"dt": "QUANT", "params": {"bins": bins}}


def vf_empty_c(wav: torch.Tensor, sr: int, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    _ = sr
    _ = options
    _ = rng
    wav0 = _ensure_tensor_audio(wav)
    return torch.zeros_like(wav0), {"dt": "EMPTY_C", "params": {}}


_VF_EFFECT_REGISTRY = {
    "TEMPO": vf_tempo,
    "SPEED": vf_speed,
    "FADE": vf_fade,
    "PITCH": vf_pitch,
    "TREBLE": vf_treble,
    "BASS": vf_bass,
    "TREMOLO": vf_tremolo,
    "CLIP": vf_clip,
    "REVERSE": vf_reverse,
    "REVERB_FREEVERB": vf_reverb_freeverb,
    "REVERB_RIR": vf_reverb_rir,
    "LOW_PASS": vf_low_pass,
    "HIGH_PASS": vf_high_pass,
    "TIME_DROPOUT": vf_time_dropout,
    "QUANT": vf_quant,
    "EMPTY_C": vf_empty_c,
}


for _vf_name, _vf_fn in _VF_EFFECT_REGISTRY.items():
    register_custom_degradation(_vf_name, _vf_fn, overwrite=True)
    register_custom_degradation(f"VF_{_vf_name}", _vf_fn, overwrite=True)


def _apollo_match_shift(x: torch.Tensor, d: torch.Tensor) -> int:
    """Estimate 1D shift like Apollo's phase-alignment helper using normalized GCC in FFT domain."""
    x0 = x.reshape(-1)
    d0 = d.reshape(-1)
    n = min(x0.numel(), d0.numel())
    if n == 0:
        return 0
    x0 = x0[:n]
    d0 = d0[:n]
    fx = torch.fft.rfft(x0)
    fd = torch.fft.rfft(d0)
    phi = fd * torch.conj(fx)
    phi = phi / (torch.abs(phi) + 1e-3)
    if phi.numel() > 0:
        phi[0] = 0
    corr = torch.fft.irfft(phi, n=n)
    tau = int(torch.argmax(torch.abs(corr)).item())
    return tau


def apollo_musdb_mp3_degradation(
    wav: torch.Tensor,
    sr: int,
    options: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Apollo-style MUSDB codec degradation:
      - MP3 codec
      - bitrate sampled from {24k, 32k, 48k, 64k, 96k, 128k} by default
      - length match and optional FFT-domain shift alignment
    """
    opts = _copy_options(options)
    wav0 = _ensure_tensor_audio(wav)
    if rng is None:
        rng = np.random.default_rng()

    bitrate_candidates = opts.get("bitrate_candidates", [24000, 32000, 48000, 64000, 96000, 128000])
    bitrate = opts.get("bitrate", "random")
    if bitrate == "random":
        bitrate = int(rng.choice(np.array(bitrate_candidates, dtype=np.int64)))
    else:
        bitrate = int(bitrate)

    # Apollo uses compression=bitrate//1000 for torchaudio apply_codec(mp3).
    compression = int(max(1, bitrate // 1000))
    align = bool(opts.get("align", True))

    out = None
    method = "none"
    if ta_apply_codec is not None:
        try:
            out = ta_apply_codec(wav0, sr, format="mp3", compression=compression)
            method = "torchaudio.apply_codec"
        except Exception:
            out = None

    if out is None:
        try:
            out = _apply_codec_ffmpeg(wav0, sr, fmt="mp3", bitrate=bitrate)
            method = "ffmpeg"
        except Exception:
            out = _robust_codec_simulation(wav0, fmt="mp3", bitrate=bitrate)
            method = "simulation"

    # Match target length exactly.
    if out.shape[-1] >= wav0.shape[-1]:
        out = out[..., :wav0.shape[-1]]
    else:
        out = torch.cat([out, wav0[..., out.shape[-1]:]], dim=-1)

    # Optional Apollo-like alignment against reference waveform.
    if align and out.numel() > 0 and wav0.numel() > 0:
        tau = _apollo_match_shift(wav0[0], out[0])
        out = torch.roll(out, -tau, dims=-1)
    else:
        tau = 0

    meta = {
        "dt": "APOLLO_MUSDB_MP3",
        "params": {
            "format": "mp3",
            "bitrate": bitrate,
            "compression": compression,
            "align": align,
            "shift": tau,
            "method": method,
        },
    }
    return out, meta


def _make_apollo_mp3_fixed_bitrate_degradation(bitrate: int):
    def _fn(
        wav: torch.Tensor,
        sr: int,
        options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        opts = _copy_options(options)
        opts["bitrate"] = int(bitrate)
        return apollo_musdb_mp3_degradation(wav, sr, options=opts, rng=rng)

    return _fn


# Register built-in Apollo-style degradation key.
register_custom_degradation("APOLLO_MUSDB_MP3", apollo_musdb_mp3_degradation)
register_custom_degradation("APOLLO_MP3_24K", _make_apollo_mp3_fixed_bitrate_degradation(24000))
register_custom_degradation("APOLLO_MP3_32K", _make_apollo_mp3_fixed_bitrate_degradation(32000))
register_custom_degradation("APOLLO_MP3_48K", _make_apollo_mp3_fixed_bitrate_degradation(48000))
register_custom_degradation("APOLLO_MP3_64K", _make_apollo_mp3_fixed_bitrate_degradation(64000))
register_custom_degradation("APOLLO_MP3_96K", _make_apollo_mp3_fixed_bitrate_degradation(96000))
register_custom_degradation("APOLLO_MP3_128K", _make_apollo_mp3_fixed_bitrate_degradation(128000))

def apply_degradation(wav: torch.Tensor, sr: int, dt: str, options: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    if rng is None:
        rng = np.random.default_rng()
    if dt in _DT_REGISTRY:
        fn = _DT_REGISTRY[dt]
    elif dt in _CUSTOM_DT_REGISTRY:
        fn = _CUSTOM_DT_REGISTRY[dt]
    else:
        raise ValueError(f"Unknown dt '{dt}'. Available: {_available_degradation_keys()}")
    wav0 = _ensure_tensor_audio(wav)
    out, meta = fn(wav0, sr, options=options, rng=rng)
    # ensure length consistency and dtype
    if out.shape != wav0.shape:
        # trim or pad with input tail
        if out.shape[1] >= wav0.shape[1]:
            out = out[:, :wav0.shape[1]]
        else:
            tail = wav0[:, out.shape[1]:wav0.shape[1]]
            out = torch.cat([out, tail], dim=1)
    meta['dt_requested'] = dt
    return out, meta

# Optional: small test util
def demo():
    import math
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.1 * np.sin(2 * math.pi * 440.0 * t).astype(np.float32)
    wav = torch.from_numpy(np.stack([sine, sine], axis=0))
    out, meta = apply_degradation(wav, sr, 'DT1')
    print(meta)
    out2, meta2 = apply_degradation(wav, sr, 'DT6')
    print(meta2)
    # test new neural codecs
    out3, meta3 = apply_degradation(wav, sr, 'DT9')
    print(meta3)
    out4, meta4 = apply_degradation(wav, sr, 'DT11')
    print(meta4)


# =====================================================================
# Utility Functions: Waveform Normalization & Length Adjustment
# =====================================================================

def normalize_waveform(
    wav: torch.Tensor,
    target_level: float = -3.0
) -> torch.Tensor:
    """
    Normalize waveform to a target loudness level (dB).

    Args:
        wav: Input waveform [C, T] or [T] (batch/single channel)
        target_level: Target loudness in dB (default: -3dB,
                      roughly 0.707 amplitude for 1.0 peak)

    Returns:
        Normalized waveform with same shape as input
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Compute RMS per channel
    rms = torch.sqrt(torch.mean(wav.pow(2), dim=-1, keepdim=True) + 1e-20)

    # Convert target level (dB) to linear scale
    # target_level_db = 20 * log10(amplitude)
    # amplitude = 10^(target_level_db / 20)
    target_amplitude = 10.0 ** (target_level / 20.0)

    # Compute scaling factor
    scale = target_amplitude / (rms + 1e-20)

    # Apply normalization
    normalized = wav * scale

    # Soft clip to prevent clipping artifacts
    normalized = torch.clamp(normalized, min=-1.0, max=1.0)

    if squeeze_output:
        normalized = normalized.squeeze(0)

    return normalized


def adjust_waveform_length(
    wav: torch.Tensor,
    target_length: int,
    pad_mode: str = 'constant',
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Adjust waveform length to target length via padding or truncation.
    Ensures length is compliant with STFT requirements (multiple of 8).

    Args:
        wav: Input waveform [C, T] or [T]
        target_length: Target length in samples (automatically adjusted to be multiple of 8)
        pad_mode: Padding mode - 'constant', 'reflect', 'replicate'
        pad_value: Value for constant padding (default: 0.0)

    Returns:
        Adjusted waveform with shape [..., adjusted_length]
        where adjusted_length is the smallest multiple of 8 >= target_length
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Ensure target_length is multiple of 8 for STFT compatibility
    adjusted_target = ((target_length + 7) // 8) * 8

    current_length = wav.shape[-1]

    if current_length == adjusted_target:
        result = wav
    elif current_length < adjusted_target:
        # Padding needed
        pad_amount = adjusted_target - current_length

        if pad_mode == 'constant':
            result = torch.nn.functional.pad(
                wav,
                (0, pad_amount),
                mode='constant',
                value=pad_value
            )
        elif pad_mode == 'reflect':
            result = torch.nn.functional.pad(
                wav,
                (0, pad_amount),
                mode='reflect'
            )
        elif pad_mode == 'replicate':
            result = torch.nn.functional.pad(
                wav,
                (0, pad_amount),
                mode='replicate'
            )
        else:
            raise ValueError(f"Unknown pad_mode: {pad_mode}")
    else:
        # Truncation needed
        result = wav[:, :adjusted_target]

    if squeeze_output:
        result = result.squeeze(0)

    return result


# =====================================================================
# AudioDegradationPipeline: Composite & Single Degradation Handler
# =====================================================================

class AudioDegradationPipeline:
    """
    Audio degradation pipeline supporting single and composite degradation modes.

    This class orchestrates the application of audio degradation functions with
    support for:
    - Single degradation: Apply one random degradation
    - Composite degradation: Randomly apply 2-3 degradations in sequence
    - Probabilistic application with configurable probability
    - Batch processing support for [B, T] waveforms

    Attributes:
        degradation_list: List of degradation keys (e.g., ['DT1', 'DT2', 'DT3'])
        mode: 'single' or 'composite' degradation mode
        prob: Probability of applying degradation (0.0 to 1.0)
        rng: NumPy random generator for reproducibility
    """

    def __init__(
        self,
        degradation_list: list = None,
        mode: str = 'single',
        prob: float = 0.9,
        seed: int = None
    ):
        """
        Initialize AudioDegradationPipeline.

        Args:
            degradation_list: List of degradation type strings
                            (default: all 12 DTs ['DT1', 'DT2', ..., 'DT12'])
            mode: Degradation mode - 'single' (one random) or 'composite' (2-3 random)
            prob: Probability of applying degradation (default: 0.9)
            seed: Random seed for reproducibility (optional)
        """
        if degradation_list is None:
            self.degradation_list = _available_degradation_keys()
        else:
            self.degradation_list = degradation_list
            # Validate all keys exist
            for dt_key in self.degradation_list:
                if dt_key not in _DT_REGISTRY and dt_key not in _CUSTOM_DT_REGISTRY:
                    raise ValueError(
                        f"Unknown degradation type: {dt_key}. "
                        f"Available: {_available_degradation_keys()}"
                    )

        if mode not in ('single', 'composite'):
            raise ValueError(f"mode must be 'single' or 'composite', got {mode}")
        self.mode = mode

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be in [0.0, 1.0], got {prob}")
        self.prob = prob

        self.rng = np.random.default_rng(seed)
        self.last_degradation_info = None

    def __call__(
        self,
        wav: torch.Tensor,
        sr: int = 48000,
        options: Dict[str, Any] = None,
        return_metadata: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Apply degradation pipeline to input waveform.

        Args:
            wav: Input waveform [T] or [C, T] (single sample or batch not supported here,
                 use loop for batch)
            sr: Sample rate (default: 48000 Hz)
            options: Dictionary of degradation-specific options
            return_metadata: If True, return metadata about applied degradations

        Returns:
            Tuple of (degraded_waveform, metadata_dict) or just degraded_waveform

            metadata_dict contains:
                - 'applied': Boolean, whether degradation was applied
                - 'degradation_type': String or list of strings (single or composite)
                - 'degradations': List of dicts with 'dt' and 'params' for each applied degradation
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Decide whether to apply degradation
        if self.rng.random() > self.prob:
            # No degradation applied
            result = wav
            metadata = {
                'applied': False,
                'degradation_type': None,
                'degradations': []
            }
        else:
            # Apply degradation(s)
            if self.mode == 'single':
                result, metadata = self._apply_single_degradation(
                    wav, sr, options
                )
            else:  # composite mode
                result, metadata = self._apply_composite_degradation(
                    wav, sr, options
                )

        self.last_degradation_info = metadata

        if squeeze_output:
            result = result.squeeze(0)

        if return_metadata:
            return result, metadata
        else:
            return result

    def _apply_single_degradation(
        self,
        wav: torch.Tensor,
        sr: int,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply a single random degradation from the list.

        Args:
            wav: Input waveform [C, T]
            sr: Sample rate
            options: Degradation options

        Returns:
            Tuple of (degraded_waveform, metadata_dict)
        """
        # Select random degradation type
        dt_type = self.rng.choice(self.degradation_list)

        # Apply degradation
        degraded, meta = apply_degradation(
            wav, sr, dt_type, options=options, rng=self.rng
        )

        metadata = {
            'applied': True,
            'degradation_type': dt_type,
            'degradations': [meta]
        }

        return degraded, metadata

    def _apply_composite_degradation(
        self,
        wav: torch.Tensor,
        sr: int,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply 2-3 random degradations in sequence.

        Args:
            wav: Input waveform [C, T]
            sr: Sample rate
            options: Degradation options (applied to all degradations)

        Returns:
            Tuple of (degraded_waveform, metadata_dict)
        """
        # Randomly choose 2 or 3 degradations
        num_degradations = self.rng.integers(2, 4)  # 2 or 3

        # Sample without replacement from available degradations
        if len(self.degradation_list) >= num_degradations:
            selected_dts = self.rng.choice(
                self.degradation_list,
                size=num_degradations,
                replace=False
            )
        else:
            # If fewer degradations available than requested, use all
            selected_dts = self.degradation_list

        # Apply degradations sequentially
        current_wav = wav.clone()
        degradation_sequence = []

        for dt_type in selected_dts:
            degraded, meta = apply_degradation(
                current_wav, sr, dt_type, options=options, rng=self.rng
            )
            degradation_sequence.append(meta)
            current_wav = degraded

        metadata = {
            'applied': True,
            'degradation_type': list(selected_dts),
            'degradations': degradation_sequence
        }

        return current_wav, metadata

    def get_supported_degradations(self) -> list:
        """Return list of supported degradation types."""
        return _available_degradation_keys()

    def get_last_degradation_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata from the last degradation applied."""
        return self.last_degradation_info


# =====================================================================
# Batch Processing Utility
# =====================================================================

def apply_degradation_batch(
    wav_batch: torch.Tensor,
    sr: int = 48000,
    pipeline: AudioDegradationPipeline = None,
    dt_type: str = None,
    options: Dict[str, Any] = None,
) -> Tuple[torch.Tensor, list]:
    """
    Apply degradation to a batch of waveforms [B, T].

    Args:
        wav_batch: Batch of waveforms [B, T]
        sr: Sample rate (default: 48000 Hz)
        pipeline: AudioDegradationPipeline instance (if provided, uses pipeline)
                 otherwise uses dt_type
        dt_type: Single degradation type string (ignored if pipeline is provided)
        options: Degradation options

    Returns:
        Tuple of (degraded_batch, metadata_list)
        where metadata_list contains metadata for each sample in batch
    """
    if wav_batch.dim() != 2:
        raise ValueError(f"Expected 2D tensor [B, T], got shape {wav_batch.shape}")

    B, T = wav_batch.shape
    degraded_batch = []
    metadata_list = []

    for b in range(B):
        wav_single = wav_batch[b]  # Shape: [T]

        if pipeline is not None:
            degraded, meta = pipeline(wav_single, sr=sr, options=options)
        elif dt_type is not None:
            degraded, meta = apply_degradation(
                wav_single, sr, dt_type, options=options
            )
        else:
            raise ValueError("Either pipeline or dt_type must be provided")

        degraded_batch.append(degraded)
        metadata_list.append(meta)

    # Stack back into batch
    degraded_batch = torch.stack(degraded_batch, dim=0)

    return degraded_batch, metadata_list
