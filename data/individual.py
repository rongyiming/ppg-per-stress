"""
Individual (subject-level) feature extraction.

For 125Hz PPG baseline (120s = 15000 samples), extracts 10 features:
  MPP, MHR, SDNN, RMSSD, mean_T_up, mean_DR, mean_T_d,
  mean_S_max, mean_DDR, CVA
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def _preprocess_ppg_poly(ppg_data, fs=125):
    """3rd-order polynomial baseline removal."""
    t = np.arange(len(ppg_data)) / fs
    baseline = np.polyfit(t, ppg_data, 3)
    baseline_curve = np.polyval(baseline, t)
    return ppg_data - baseline_curve


def _detect_peaks(ppg_clean, fs=125):
    """Detect systolic peaks with physiological constraints."""
    min_distance = int(0.5 * fs)          # <= 120 bpm
    # Only accept peaks above the signal mean: the systolic peak of a
    # detrended PPG sits well above zero, secondary waves do not.
    peaks, _ = find_peaks(ppg_clean, distance=min_distance,
                          height=np.mean(ppg_clean))
    pp_intervals = np.diff(peaks) / fs * 1000
    mean_pp = np.mean(pp_intervals)
    std_pp = np.std(pp_intervals) + 1      # +1 ms keeps the bound finite
    mask = (pp_intervals > mean_pp - 3 * std_pp) & (pp_intervals < mean_pp + 3 * std_pp)
    pp_intervals = pp_intervals[mask]
    peaks = peaks[:len(pp_intervals) + 1]
    hr_series = 60000 / pp_intervals
    return peaks, pp_intervals, hr_series


def _extract_single_wave_features(ppg_clean, peaks, fs=125):
    """Per-wave morphological features, averaged across beats."""
    step_ms = 1000 / fs
    t_up_list, dr_list, t_d_list = [], [], []
    s_max_list, ddr_list, amp_list = [], [], []

    for i in range(len(peaks) - 1):
        start_idx = max(0, peaks[i] - 10)
        end_idx = min(len(ppg_clean) - 1, peaks[i + 1] - 10)
        single_wave = ppg_clean[start_idx:end_idx]

        wave_min, wave_max = np.min(single_wave), np.max(single_wave)
        if wave_max - wave_min == 0:
            continue
        single_wave_norm = (single_wave - wave_min) / (wave_max - wave_min)

        p_idx = peaks[i] - start_idx
        t0_idx = np.argmin(single_wave_norm[:p_idx]) if p_idx > 0 else 0
        t2_idx = len(single_wave_norm) - 1

        t_up = (p_idx - t0_idx) * step_ms
        t_up_list.append(t_up)

        t_total = (t2_idx - t0_idx) * step_ms
        t_diastole = (t2_idx - p_idx) * step_ms
        dr_list.append((t_diastole / t_total) * 100 if t_total > 0 else 0)

        descending = single_wave_norm[p_idx:]
        if len(descending) >= 5:
            descending_smooth = savgol_filter(descending, 5, 2)
            deriv = np.gradient(descending_smooth)
            zero_cross = np.where(np.diff(np.sign(deriv)))[0]
            if len(zero_cross) >= 2:
                d_idx_rel = zero_cross[1]
                d_idx = p_idx + d_idx_rel
                t_d_list.append((d_idx - p_idx) * step_ms)
                ddr_list.append((1 - single_wave_norm[d_idx]) * 100)

        ascending = single_wave_norm[t0_idx:p_idx + 1]
        if len(ascending) >= 3:
            slope, _ = np.polyfit(np.arange(len(ascending)), ascending, 1)
            s_max_list.append(slope)

        amp_list.append(single_wave[p_idx])

    return {
        "mean_T_up": float(np.mean(t_up_list)) if t_up_list else 0.0,
        "mean_DR": float(np.mean(dr_list)) if dr_list else 0.0,
        "mean_T_d": float(np.mean(t_d_list)) if t_d_list else 0.0,
        "mean_S_max": float(np.mean(s_max_list)) if s_max_list else 0.0,
        "mean_DDR": float(np.mean(ddr_list)) if ddr_list else 0.0,
        "CVA": (float((np.std(amp_list) / (np.mean(amp_list) + 1e-8)) * 100)
                if amp_list else 0.0),
    }


def new_individual(ppg_data, fs=125):
    """Compute 10-dim individual feature vector from ~120s baseline PPG.

    Args:
        ppg_data: 1D numpy array, baseline PPG signal.
        fs: Sampling rate (Hz).

    Returns:
        List of 10 floats: [MPP, MHR, SDNN, RMSSD, mean_T_up,
                            mean_DR, mean_T_d, mean_S_max, mean_DDR, CVA]
    """
    ppg_clean = _preprocess_ppg_poly(np.asarray(ppg_data, dtype=np.float64), fs)
    peaks, pp_intervals, hr_series = _detect_peaks(ppg_clean, fs)

    if len(pp_intervals) > 0:
        mpp = float(np.mean(pp_intervals))
        mhr = float(np.mean(hr_series))
        sdnn = (float(np.std(pp_intervals, ddof=1))
                if len(pp_intervals) > 1 else 0.0)
        rmssd = (float(np.sqrt(np.mean(np.square(np.diff(pp_intervals)))))
                 if len(pp_intervals) > 1 else 0.0)
    else:
        mpp = mhr = sdnn = rmssd = 0.0

    wave_feat = _extract_single_wave_features(ppg_clean, peaks, fs)

    return [
        mpp,                            # MPP (ms)
        mhr,                            # MHR (bpm)
        sdnn,                           # SDNN (ms)
        rmssd,                          # RMSSD (ms)
        wave_feat["mean_T_up"],         # mean rise time (ms)
        wave_feat["mean_DR"],           # mean diastolic ratio (%)
        wave_feat["mean_T_d"],          # mean dicrotic delay (ms)
        wave_feat["mean_S_max"],        # mean max slope
        wave_feat["mean_DDR"],          # mean dicrotic depth ratio (%)
        wave_feat["CVA"],               # amplitude CV (%)
    ]
