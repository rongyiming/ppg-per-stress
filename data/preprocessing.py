"""
PPG preprocessing.

Training-time pipeline, applied per 10 s window:
  z-scored signal (from the segment file)
    -> zero-phase Chebyshev type I band-pass, 0.5-4.0 Hz, order 4
    -> 3rd-order polynomial detrend

Per-sample amplitude normalisation is left to the model (InstanceNorm1d).

Also includes heart-rate / HRV computation and signal-quality utilities.
"""

import numpy as np
from scipy.signal import butter, filtfilt, cheby1, find_peaks
from scipy.stats import skew


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


# ── Band-pass filters ─────────────────────────────────────────────────────────

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                           fs: int, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (available, not used by default)."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ── Baseline removal ──────────────────────────────────────────────────────────

def remove_baseline_poly(data: np.ndarray, fs: int, poly_order: int = 3) -> np.ndarray:
    """Polynomial baseline removal (3rd order by default)."""
    t = np.arange(len(data)) / fs
    baseline = np.polyfit(t, data, poly_order)
    baseline_curve = np.polyval(baseline, t)
    return data - baseline_curve


def preprocess_ppg(data: np.ndarray, fs: int,
                   lowcut: float = 0.5, highcut: float = 4.0,
                   order: int = 4, filter_type: str = 'cheby') -> np.ndarray:
    """Polynomial baseline removal + bandpass filtering.

    order=0 applies the polynomial detrend only (the segment file is already
    z-scored and is band-pass filtered here, per window).
    ``filter_type`` selects Butterworth or Chebyshev type I; the reported
    results use Chebyshev type I, order 4.
    """
    data = np.asarray(data, dtype=np.float64)
    if order > 0:
        if filter_type == 'butter':
            data = butter_bandpass_filter(data, lowcut, highcut, fs, order=order)
        else:
            data = chebyshev_bandpass(data, lowcut, highcut, fs, order=order)
    data = remove_baseline_poly(data, fs, poly_order=3)
    return data


def chebyshev_bandpass(data: np.ndarray, lowcut: float, highcut: float,
                       fs: int, order: int = 4) -> np.ndarray:
    """Zero-phase Chebyshev Type I band-pass filter."""
    nyq = 0.5 * fs
    b, a = cheby1(order, 0.1, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


# ── HR / HRV computation ───────────────────────────────────────────────────────

def calculate_hr_hrv(ppg_signal: np.ndarray, fs: int = 125) -> dict:
    """Compute heart rate and HRV metrics from preprocessed PPG.

    Args:
        ppg_signal: Preprocessed 1D PPG signal.
        fs: Sampling rate (Hz).

    Returns:
        Dict with keys: n_beats, hr_bpm, sdnn, rmssd, nn50, pnn50, sqi.
        Values are None if computation fails.
    """
    result = {
        'n_beats': None,
        'hr_bpm': None,
        'sdnn': None,
        'rmssd': None,
        'nn50': None,
        'pnn50': None,
        'sqi': None,
    }

    if len(ppg_signal) < fs * 10:
        return result

    result['sqi'] = _compute_sqi(ppg_signal, fs)

    peak_height = np.mean(ppg_signal) + 1.0 * np.std(ppg_signal)
    min_peak_distance = int(0.3 * fs)
    peaks, _ = find_peaks(
        ppg_signal,
        height=peak_height,
        distance=min_peak_distance,
        prominence=0.1 * np.std(ppg_signal),
    )

    if len(peaks) < 3:
        return result

    peak_times = peaks / fs
    pp_intervals = np.diff(peak_times)
    pp_intervals_ms = pp_intervals * 1000

    result['n_beats'] = len(peaks)
    result['hr_bpm'] = round(60.0 / np.mean(pp_intervals), 1)

    if result['hr_bpm'] < 30 or result['hr_bpm'] > 220:
        result['n_beats'] = None
        result['hr_bpm'] = None
        return result

    result['sdnn'] = round(float(np.std(pp_intervals_ms)), 1)
    diff_pp = np.diff(pp_intervals_ms)
    result['rmssd'] = round(float(np.sqrt(np.mean(diff_pp ** 2))), 1)
    nn50 = int(np.sum(np.abs(diff_pp) > 50))
    result['nn50'] = nn50
    result['pnn50'] = round((nn50 / len(diff_pp)) * 100, 1) if len(diff_pp) > 0 else 0.0

    return result


def _compute_sqi(ppg: np.ndarray, fs: int, window_sec: float = 5.0) -> float:
    """Signal Quality Index via skewness of 5s windows."""
    window_size = int(fs * window_sec)
    n_windows = max(len(ppg) // window_size, 1)
    skews = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        skews.append(skew(ppg[start:end]))
    return float(np.mean(skews))
