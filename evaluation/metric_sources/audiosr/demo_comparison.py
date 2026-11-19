#!/usr/bin/env python3
"""
Demo script to show the effect of energy scaling on LSD calculation.
Compares LSD values with and without energy unification.
"""

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
EPS = 1e-8
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
LOWER_PERCENTILE = 10


def unify_lower_frequency_energy(gt_spectrogram, target_spectrogram, lower_percentile=10):
    """Scale target spectrogram to match GT lower frequency energy."""
    freq_bins, time_frames = gt_spectrogram.shape
    num_lower_bins = int(freq_bins * lower_percentile / 100)
    
    if num_lower_bins == 0:
        return target_spectrogram
    
    gt_lower_energy = np.mean(gt_spectrogram[:num_lower_bins, :])
    target_lower_energy = np.mean(target_spectrogram[:num_lower_bins, :])
    
    if target_lower_energy > 0:
        scaling_factor = gt_lower_energy / target_lower_energy
    else:
        scaling_factor = 1.0
    
    return target_spectrogram * scaling_factor, scaling_factor


def lsd(est, target):
    """Calculate Log Spectral Distance."""
    if isinstance(est, np.ndarray):
        est = torch.from_numpy(est).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    if est.dim() == 2:
        est = est.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    
    if est.dim() == 3:
        est = est.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    lsd_result = torch.log10((target**2/((est + EPS)**2)) + EPS)**2
    lsd_per_frame = torch.mean(lsd_result, dim=2)**0.5
    lsd_per_frame = torch.mean(lsd_per_frame, dim=(0, 1))
    
    return lsd_per_frame
