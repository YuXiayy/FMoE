import numpy as np
import torch

try:
    from scipy import signal
except Exception:
    signal = None

try:
    import librosa
except Exception:
    librosa = None


class EffectChain:
    def __init__(self):
        self._ops = []

    def tempo(self, factor):
        self._ops.append(("tempo", float(factor)))
        return self

    def speed(self, factor):
        self._ops.append(("speed", float(factor)))
        return self

    def fade(self, fade_in_sec, total_sec, fade_out_start_sec):
        self._ops.append(("fade", (float(fade_in_sec), float(total_sec), float(fade_out_start_sec))))
        return self

    def pitch(self, cents):
        self._ops.append(("pitch", float(cents)))
        return self

    def rate(self, sample_rate):
        self._ops.append(("rate", int(sample_rate)))
        return self

    def treble(self, level):
        self._ops.append(("treble", float(level)))
        return self

    def bass(self, level):
        self._ops.append(("bass", float(level)))
        return self

    def tremolo(self, level):
        self._ops.append(("tremolo", float(level)))
        return self

    def clip(self, value):
        self._ops.append(("clip", float(value)))
        return self

    def reverb(self, reverberance, dumping, room_size):
        self._ops.append(("reverb", (float(reverberance), float(dumping), float(room_size))))
        return self

    def lowpass(self, cutoff):
        self._ops.append(("lowpass", float(cutoff)))
        return self

    def highpass(self, cutoff):
        self._ops.append(("highpass", float(cutoff)))
        return self

    def reverse(self):
        self._ops.append(("reverse", None))
        return self

    def channels(self, _n):
        # VoiceFixer uses channels(1); our pipeline already runs per channel.
        return self

    def apply(self, audio, src_info=None, target_info=None):
        if isinstance(audio, torch.Tensor):
            y = audio.detach().cpu().numpy().astype(np.float32).squeeze()
        else:
            y = np.asarray(audio, dtype=np.float32).squeeze()

        sr = 44100
        if src_info is not None and "rate" in src_info:
            sr = int(src_info["rate"])

        pitch_sr = sr
        for name, value in self._ops:
            if name == "rate":
                pitch_sr = int(value)
                continue
            if name == "tempo":
                y = _tempo(y, float(value))
            elif name == "speed":
                y = _speed(y, float(value))
            elif name == "fade":
                y = _fade(y, sr, value)
            elif name == "pitch":
                y = _pitch(y, pitch_sr, float(value))
            elif name == "treble":
                y = _treble(y, sr, float(value))
            elif name == "bass":
                y = _bass(y, sr, float(value))
            elif name == "tremolo":
                y = _tremolo(y, sr, float(value))
            elif name == "clip":
                y = _clip(y, float(value))
            elif name == "reverb":
                y = _reverb(y, sr, value)
            elif name == "lowpass":
                y = _lowpass(y, sr, float(value))
            elif name == "highpass":
                y = _highpass(y, sr, float(value))
            elif name == "reverse":
                y = y[::-1].copy()

        return torch.from_numpy(y.astype(np.float32))


def _resample_len(y, new_len):
    if new_len <= 1:
        return y
    x_old = np.linspace(0.0, 1.0, num=y.shape[0], endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, y).astype(np.float32)


def _tempo(y, factor):
    factor = max(0.1, float(factor))
    if librosa is not None:
        try:
            return librosa.effects.time_stretch(y, rate=factor).astype(np.float32)
        except Exception:
            pass
    new_len = max(1, int(round(y.shape[0] / factor)))
    return _resample_len(y, new_len)


def _speed(y, factor):
    factor = max(0.1, float(factor))
    new_len = max(1, int(round(y.shape[0] / factor)))
    return _resample_len(y, new_len)


def _fade(y, sr, params):
    fade_in_sec, _total_sec, fade_out_start_sec = params
    n = y.shape[0]
    env = np.ones(n, dtype=np.float32)
    fi = int(max(0, round(fade_in_sec * sr)))
    fo = int(max(0, round(fade_out_start_sec * sr)))
    if fi > 1:
        env[:fi] = np.linspace(0.0, 1.0, num=fi, endpoint=True, dtype=np.float32)
    if fo < n:
        tail = n - fo
        if tail > 1:
            env[fo:] = np.linspace(1.0, 0.0, num=tail, endpoint=True, dtype=np.float32)
    return (y * env).astype(np.float32)


def _pitch(y, sr, cents):
    if librosa is not None:
        try:
            semitones = float(cents) / 100.0
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)
        except Exception:
            pass
    # fallback: no-op when pitch shift backend is unavailable
    return y.astype(np.float32)


def _lowpass(y, sr, cutoff):
    if signal is None:
        return y.astype(np.float32)
    nyq = 0.5 * sr
    wn = min(max(cutoff / nyq, 1e-4), 0.999)
    b, a = signal.butter(4, wn, btype="low")
    return signal.filtfilt(b, a, y).astype(np.float32)


def _highpass(y, sr, cutoff):
    if signal is None:
        return y.astype(np.float32)
    nyq = 0.5 * sr
    wn = min(max(cutoff / nyq, 1e-4), 0.999)
    b, a = signal.butter(4, wn, btype="high")
    return signal.filtfilt(b, a, y).astype(np.float32)


def _treble(y, sr, level):
    hp = _highpass(y, sr, 3000.0)
    gain = max(0.0, level) / 20.0
    return (y + gain * hp).astype(np.float32)


def _bass(y, sr, level):
    lp = _lowpass(y, sr, 250.0)
    gain = max(0.0, level) / 20.0
    return (y + gain * lp).astype(np.float32)


def _tremolo(y, sr, level):
    freq = float(np.clip(level, 0.1, 50.0))
    t = np.arange(y.shape[0], dtype=np.float32) / float(sr)
    mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * freq * t))
    depth = 0.5
    env = (1.0 - depth) + depth * mod
    return (y * env).astype(np.float32)


def _clip(y, value):
    thr = float(np.clip(value, 0.05, 1.0))
    return np.clip(y, -thr, thr).astype(np.float32)


def _reverb(y, sr, params):
    reverberance, dumping, room_size = params
    rev = float(np.clip(reverberance / 100.0, 0.0, 1.0))
    damp = float(np.clip(dumping / 100.0, 0.0, 1.0))
    room = float(np.clip(room_size / 100.0, 0.0, 1.0))

    ir_len = max(64, int(sr * (0.05 + 0.45 * room)))
    t = np.arange(ir_len, dtype=np.float32)
    decay = np.exp(-t / (ir_len * (0.15 + 0.85 * (1.0 - damp))))
    ir = decay
    ir[0] += 1.0
    ir = ir * (0.2 + 0.8 * rev)
    ir = ir / (np.sum(np.abs(ir)) + 1e-8)

    if signal is None:
        y_rev = np.convolve(y, ir, mode="full")[: y.shape[0]]
    else:
        y_rev = signal.fftconvolve(y, ir, mode="full")[: y.shape[0]]
    return y_rev.astype(np.float32)
