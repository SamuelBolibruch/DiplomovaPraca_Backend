import os
import pandas as pd
import numpy as np

# -----------------------------
# Config: filtering
# -----------------------------
MAX_INTERVAL_MS = 2750  # hranica na "dlhé pauzy" (ms) medzi znakmi
MIN_INTERVAL_MS = 0     # vyhadzujeme <= 0 (glitche)
MICRO_PAUSE_MS = 700    # hranica na "mikropauzy" (ms)

CONFIGS = [
    {"data_dir": "data/common_training", "output_dir": "data/vectors", "keystrokes_file": "keystrokes_common.csv"},
    {"data_dir": "data/personal_training", "output_dir": "data/vectors_personal", "keystrokes_file": "keystrokes_personal.csv"},
]

# -----------------------------
# IMU normalization / resampling config
# -----------------------------
IMU_GRAVITY_TAU_SEC = 0.7
IMU_RESAMPLE_HZ = 50.0
IMU_MIN_FFT_SAMPLES = 64

# -----------------------------
# Cross-modal config (keystrokes <-> IMU)
# -----------------------------
ACC_KEY_PRE_MS = 100
ACC_KEY_POST_MS = 150
ACC_LAG_MAX_MS = 150
FAST_DD_MS = 300
EPS = 1e-9

XCORR_MAX_LAG_MS = 300
RISE_FRAC = 0.8


# -----------------------------
# Helpers: keystroke stats
# -----------------------------
def filter_intervals_ms(s: pd.Series, min_ms: float, max_ms: float) -> pd.Series:
    s = s.dropna()
    return s[(s > min_ms) & (s <= max_ms)]

def safe_skew(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.skew()) if len(s) >= 3 else 0.0

def safe_kurt(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.kurtosis()) if len(s) >= 4 else 0.0

def series_stats(prefix: str, s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurt": 0.0,
        }

    return {
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=1)) if len(s) >= 2 else 0.0,
        f"{prefix}_median": float(s.median()),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_max": float(s.max()),
        f"{prefix}_skew": safe_skew(s),
        f"{prefix}_kurt": safe_kurt(s),
    }


# -----------------------------
# Helpers: active typing segments + mapping sensors -> rounds + filtering sensors by segments
# -----------------------------
def build_active_segments_from_keystrokes(veta: pd.DataFrame, *, max_gap_ms: int) -> list[tuple[int, int]]:
    if "ActionType" not in veta.columns or "TimestampBeforeNs" not in veta.columns:
        return []

    action_l = veta["ActionType"].astype(str).str.lower()
    insert_mask = (action_l == "insert")

    ins = veta.loc[insert_mask, "TimestampBeforeNs"].dropna()
    if len(ins) < 2:
        return []

    ins = pd.to_numeric(ins, errors="coerce").dropna().astype(np.int64).to_numpy()
    if len(ins) < 2:
        return []

    gaps_ms = (ins[1:] - ins[:-1]) / 1_000_000.0

    segments: list[tuple[int, int]] = []
    seg_start = int(ins[0])
    prev_t = int(ins[0])

    for i in range(1, len(ins)):
        t = int(ins[i])
        gap = float(gaps_ms[i - 1])

        if gap <= max_gap_ms:
            prev_t = t
        else:
            if prev_t > seg_start:
                segments.append((seg_start, prev_t))
            seg_start = t
            prev_t = t

    if prev_t > seg_start:
        segments.append((seg_start, prev_t))

    return segments

def assign_round_ids_by_bounds(sensor_df: pd.DataFrame, round_bounds: pd.DataFrame) -> pd.DataFrame:
    s = sensor_df.copy()
    s["timestamp"] = pd.to_numeric(s["timestamp"], errors="coerce")
    s = s.dropna(subset=["timestamp"])
    s["timestamp"] = s["timestamp"].astype(np.int64)

    if len(round_bounds) == 0 or len(s) == 0:
        s["RoundId"] = pd.Series([], dtype="Int64")
        return s.iloc[0:0].copy()

    intervals = pd.IntervalIndex.from_arrays(
        round_bounds["start_ns"].to_numpy(dtype=np.int64),
        round_bounds["end_ns"].to_numpy(dtype=np.int64),
        closed="both",
    )

    idx = intervals.get_indexer(s["timestamp"].to_numpy(dtype=np.int64))
    rid = round_bounds["RoundId"].to_numpy()

    s["RoundId"] = np.where(idx >= 0, rid[idx], np.nan)
    s = s.dropna(subset=["RoundId"])
    s["RoundId"] = s["RoundId"].astype(int)
    return s

def filter_sensor_by_segments(sensor_round_df: pd.DataFrame, segments_ns: list[tuple[int, int]]) -> pd.DataFrame:
    if len(sensor_round_df) == 0 or len(segments_ns) == 0:
        return sensor_round_df.iloc[0:0].copy()

    ts = sensor_round_df["timestamp"].to_numpy(dtype=np.int64)
    mask = np.zeros(len(sensor_round_df), dtype=bool)

    for a, b in segments_ns:
        mask |= (ts >= int(a)) & (ts <= int(b))

    return sensor_round_df.loc[mask].copy()

def load_sensor_csv(path: str) -> pd.DataFrame:
    s = pd.read_csv(path)
    for c in ["timestamp", "x", "y", "z"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["timestamp", "x", "y", "z"])
    s["timestamp"] = s["timestamp"].astype(np.int64)
    return s


# -----------------------------
# Helpers: IMU normalization + resampling
# -----------------------------
def estimate_sampling_stats(t_sec: np.ndarray) -> dict:
    if t_sec is None or len(t_sec) < 2:
        return {"n": 0, "duration": 0.0, "dt_mean": 0.0, "dt_std": 0.0, "fs_hz": 0.0}

    dt = np.diff(t_sec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return {
            "n": int(len(t_sec)),
            "duration": float(t_sec[-1] - t_sec[0]) if len(t_sec) >= 2 else 0.0,
            "dt_mean": 0.0,
            "dt_std": 0.0,
            "fs_hz": 0.0,
        }

    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt, ddof=1)) if len(dt) >= 2 else 0.0
    dt_med = float(np.median(dt))
    fs_hz = float(1.0 / dt_med) if dt_med > 0 else 0.0
    duration = float(t_sec[-1] - t_sec[0]) if len(t_sec) >= 2 else 0.0

    return {"n": int(len(t_sec)), "duration": duration, "dt_mean": dt_mean, "dt_std": dt_std, "fs_hz": fs_hz}

def gravity_remove_ema(x: np.ndarray, y: np.ndarray, z: np.ndarray, t_sec: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t_sec)
    if n < 2:
        return x.copy(), y.copy(), z.copy()

    gx = np.zeros(n, dtype=np.float64)
    gy = np.zeros(n, dtype=np.float64)
    gz = np.zeros(n, dtype=np.float64)
    gx[0], gy[0], gz[0] = x[0], y[0], z[0]

    for i in range(1, n):
        dt = float(t_sec[i] - t_sec[i - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.0
        a = float(np.exp(-dt / tau)) if (tau > 0 and dt > 0) else 0.0
        gx[i] = a * gx[i - 1] + (1 - a) * x[i]
        gy[i] = a * gy[i - 1] + (1 - a) * y[i]
        gz[i] = a * gz[i - 1] + (1 - a) * z[i]

    return x - gx, y - gy, z - gz

def resample_uniform(t_sec: np.ndarray, v: np.ndarray, target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if len(t_sec) < 2 or target_hz <= 0:
        return t_sec, v

    t0 = float(t_sec[0])
    t1 = float(t_sec[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return t_sec, v

    step = 1.0 / float(target_hz)
    n_u = int(np.floor((t1 - t0) / step)) + 1
    if n_u < 2:
        return t_sec, v

    t_u = t0 + step * np.arange(n_u, dtype=np.float64)

    order = np.argsort(t_sec)
    t_sorted = t_sec[order]
    v_sorted = v[order]

    uniq_mask = np.concatenate([[True], np.diff(t_sorted) > 0])
    t_sorted = t_sorted[uniq_mask]
    v_sorted = v_sorted[uniq_mask]
    if len(t_sorted) < 2:
        return t_sec, v

    v_u = np.interp(t_u, t_sorted, v_sorted).astype(np.float64)
    return t_u, v_u

def spectral_entropy_from_signal(v: np.ndarray) -> float:
    if v is None or len(v) < 8:
        return 0.0

    vv = v - float(np.mean(v))
    vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)

    spec = (np.abs(np.fft.rfft(vv)) ** 2)
    if len(spec) > 1:
        spec = spec[1:]

    ssum = float(np.sum(spec))
    if not np.isfinite(ssum) or ssum <= 0:
        return 0.0

    p = spec / ssum
    p = p[(p > 0) & np.isfinite(p)]
    if len(p) == 0:
        return 0.0

    H = -float(np.sum(p * np.log(p)))
    Hmax = float(np.log(len(spec))) if len(spec) > 1 else 1.0
    return float(H / Hmax) if Hmax > 0 else 0.0

def _zscore(v: np.ndarray) -> np.ndarray:
    if v is None or len(v) == 0:
        return v
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0
    if not np.isfinite(s) or s <= 0:
        return v - m
    return (v - m) / s

def _integral_mag2_dt(mag: np.ndarray, t_sec: np.ndarray) -> float:
    if mag is None or t_sec is None or len(mag) < 2 or len(t_sec) < 2:
        return 0.0
    dt = np.diff(t_sec)
    valid = np.isfinite(dt) & (dt > 0) & np.isfinite(mag[:-1]) & np.isfinite(mag[1:])
    if not np.any(valid):
        return 0.0
    m2 = mag * mag
    area = 0.5 * (m2[:-1] + m2[1:]) * dt
    return float(np.sum(area[valid]))


# -----------------------------
# Helpers: GLOBAL sensor features
# -----------------------------
def compute_global_sensor_features(sensor_df_round: pd.DataFrame, prefix: str) -> dict:
    def _empty():
        return {
            f"{prefix}_mag_mean": 0.0,
            f"{prefix}_mag_std": 0.0,
            f"{prefix}_energy": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_jerk_std": 0.0,
            f"{prefix}_mag_median": 0.0,
            f"{prefix}_mag_iqr": 0.0,
            f"{prefix}_axis_corr_xy": 0.0,
            f"{prefix}_axis_corr_xz": 0.0,
            f"{prefix}_axis_corr_yz": 0.0,
            f"{prefix}_mag_p95": 0.0,
            f"{prefix}_mag_p99": 0.0,
            f"{prefix}_mag_mad": 0.0,
            f"{prefix}_mag_trend_slope": 0.0,
            f"{prefix}_mag_spectral_entropy": 0.0,
            f"{prefix}_planarity": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_n_samples": 0,
            f"{prefix}_duration_sec": 0.0,
            f"{prefix}_dt_mean": 0.0,
            f"{prefix}_dt_std": 0.0,
            f"{prefix}_fs_hz": 0.0,
        }

    if sensor_df_round is None or len(sensor_df_round) == 0:
        return _empty()

    g = sensor_df_round.sort_values("timestamp").copy()
    for c in ["timestamp", "x", "y", "z"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["timestamp", "x", "y", "z"])
    if len(g) == 0:
        return _empty()

    ts_ns = g["timestamp"].astype(np.int64).to_numpy()
    t_sec = (ts_ns - ts_ns[0]).astype(np.float64) / 1_000_000_000.0

    samp = estimate_sampling_stats(t_sec)

    x = g["x"].to_numpy(dtype=np.float64)
    y = g["y"].to_numpy(dtype=np.float64)
    z = g["z"].to_numpy(dtype=np.float64)

    if prefix.lower().startswith("acc"):
        x, y, z = gravity_remove_ema(x, y, z, t_sec, tau=IMU_GRAVITY_TAU_SEC)
    else:
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        z = z - float(np.mean(z))

    mag = np.sqrt(x*x + y*y + z*z)
    mag_s = pd.Series(mag)

    mag_mean = float(mag_s.mean())
    mag_std = float(mag_s.std(ddof=1)) if len(mag_s) >= 2 else 0.0
    mag_median = float(mag_s.median())
    mag_iqr = float(mag_s.quantile(0.75) - mag_s.quantile(0.25)) if len(mag_s) > 0 else 0.0

    energy = float(np.mean(mag * mag)) if len(mag) > 0 else 0.0
    rms = float(np.sqrt(energy)) if energy > 0 else 0.0

    jerk_std = 0.0
    if len(mag) >= 3:
        dt = np.diff(t_sec)
        dm = np.diff(mag)
        valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dm)
        if np.any(valid):
            jerk = dm[valid] / dt[valid]
            jerk_std = float(np.std(jerk, ddof=1)) if len(jerk) >= 2 else 0.0

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or len(b) < 3:
            return 0.0
        v = float(np.corrcoef(a, b)[0, 1])
        return 0.0 if np.isnan(v) else v

    axis_corr_xy = _safe_corr(x, y)
    axis_corr_xz = _safe_corr(x, z)
    axis_corr_yz = _safe_corr(y, z)

    mag_p95 = float(mag_s.quantile(0.95)) if len(mag_s) > 0 else 0.0
    mag_p99 = float(mag_s.quantile(0.99)) if len(mag_s) > 0 else 0.0

    mag_mad = float(np.median(np.abs(mag - mag_median))) if len(mag) > 0 else 0.0

    mag_trend_slope = 0.0
    if len(mag) >= 3:
        tt = t_sec - t_sec[0]
        valid = np.isfinite(tt) & np.isfinite(mag)
        if np.sum(valid) >= 3:
            mag_trend_slope = float(np.polyfit(tt[valid], mag[valid], 1)[0])

    mag_spectral_entropy = 0.0
    if len(mag) >= 16 and samp["duration"] > 0:
        t_u, mag_u = resample_uniform(t_sec, mag, target_hz=IMU_RESAMPLE_HZ)
        if len(mag_u) >= IMU_MIN_FFT_SAMPLES:
            mag_spectral_entropy = spectral_entropy_from_signal(mag_u)

    planarity = 0.0
    if len(x) >= 3:
        M = np.vstack([x, y, z]).T
        M = M[np.all(np.isfinite(M), axis=1)]
        if M.shape[0] >= 3:
            C = np.cov(M, rowvar=False)
            try:
                w = np.linalg.eigvalsh(C)
                w = np.sort(np.real(w))[::-1]
                s = float(np.sum(w))
                planarity = float(w[0] / s) if s > 0 and np.isfinite(s) else 0.0
            except np.linalg.LinAlgError:
                planarity = 0.0

    cv = float(mag_std / mag_mean) if mag_mean > 0 else 0.0

    return {
        f"{prefix}_mag_mean": mag_mean,
        f"{prefix}_mag_std": mag_std,
        f"{prefix}_energy": energy,
        f"{prefix}_rms": rms,
        f"{prefix}_jerk_std": jerk_std,
        f"{prefix}_mag_median": mag_median,
        f"{prefix}_mag_iqr": mag_iqr,
        f"{prefix}_axis_corr_xy": axis_corr_xy,
        f"{prefix}_axis_corr_xz": axis_corr_xz,
        f"{prefix}_axis_corr_yz": axis_corr_yz,
        f"{prefix}_mag_p95": mag_p95,
        f"{prefix}_mag_p99": mag_p99,
        f"{prefix}_mag_mad": mag_mad,
        f"{prefix}_mag_trend_slope": mag_trend_slope,
        f"{prefix}_mag_spectral_entropy": mag_spectral_entropy,
        f"{prefix}_planarity": planarity,
        f"{prefix}_cv": cv,
        f"{prefix}_n_samples": int(samp["n"]),
        f"{prefix}_duration_sec": float(samp["duration"]),
        f"{prefix}_dt_mean": float(samp["dt_mean"]),
        f"{prefix}_dt_std": float(samp["dt_std"]),
        f"{prefix}_fs_hz": float(samp["fs_hz"]),
    }


# -----------------------------
# Cross-modal: generic IMU <-> keystrokes
# -----------------------------
def compute_cross_imu_keystroke_features(
    sensor_round_df_clean: pd.DataFrame,
    insert_times_ns: np.ndarray,
    insert_space_times_ns: np.ndarray,
    insert_letter_times_ns: np.ndarray,
    *,
    prefix: str,
    is_acc: bool,
) -> tuple[dict, np.ndarray]:
    out = {
        f"{prefix}_auc_per_char_mean": 0.0,
        f"{prefix}_peak_to_rms_ratio_mean": 0.0,
        f"{prefix}_post_pre_energy_ratio_mean": 0.0,
        f"{prefix}_rise_time_ms_mean": 0.0,
        f"{prefix}_keystroke_xcorr_lag_ms": 0.0,
        f"{prefix}_keystroke_xcorr_peak": 0.0,
        f"{prefix}_peak_space_minus_letter": 0.0,
        f"{prefix}_peak_per_char_mean": 0.0,
        f"{prefix}_jerk_peak_per_char_mean": 0.0,
        f"{prefix}_energy_per_char": 0.0,
        f"{prefix}_peak_lag_mean_ms": 0.0,
        f"corr_dd_{prefix}_energy": 0.0,
        f"fast_slow_{prefix}_ratio": 0.0,
    }

    if sensor_round_df_clean is None or len(sensor_round_df_clean) == 0:
        return out, np.array([], dtype=np.float64)
    if insert_times_ns is None or len(insert_times_ns) == 0:
        return out, np.array([], dtype=np.float64)

    g = sensor_round_df_clean.sort_values("timestamp").copy()
    for c in ["timestamp", "x", "y", "z"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["timestamp", "x", "y", "z"])
    if len(g) == 0:
        return out, np.array([], dtype=np.float64)

    ts = g["timestamp"].astype(np.int64).to_numpy()
    t_sec = (ts - ts[0]).astype(np.float64) / 1_000_000_000.0

    x = g["x"].to_numpy(dtype=np.float64)
    y = g["y"].to_numpy(dtype=np.float64)
    z = g["z"].to_numpy(dtype=np.float64)

    if is_acc:
        x, y, z = gravity_remove_ema(x, y, z, t_sec, tau=IMU_GRAVITY_TAU_SEC)
    else:
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        z = z - float(np.mean(z))

    mag = np.sqrt(x*x + y*y + z*z)

    n_chars = int(len(insert_times_ns))
    E_total = _integral_mag2_dt(mag, t_sec)
    out[f"{prefix}_energy_per_char"] = float(E_total / max(n_chars, 1))

    pre_ns = int(ACC_KEY_PRE_MS * 1_000_000)
    post_ns = int(ACC_KEY_POST_MS * 1_000_000)
    lag_max_ns = int(ACC_LAG_MAX_MS * 1_000_000)

    peaks = []
    jerk_peaks = []
    lags_ms = []
    aucs = []
    peak_to_rms = []
    post_pre_ratio = []
    rise_times_ms = []

    for tk in insert_times_ns:
        a = tk - pre_ns
        b = tk + post_ns
        m = (ts >= a) & (ts <= b)
        if not np.any(m):
            continue

        mag_w = mag[m]
        ts_w = ts[m]
        tsec_w = (ts_w - ts_w[0]).astype(np.float64) / 1_000_000_000.0

        pk = float(np.max(mag_w))
        peaks.append(pk)

        rms_w = float(np.sqrt(np.mean(mag_w * mag_w))) if len(mag_w) > 0 else 0.0
        peak_to_rms.append(float(pk / (rms_w + EPS)))

        aucs.append(_integral_mag2_dt(mag_w, tsec_w))

        jpk = 0.0
        if len(mag_w) >= 3:
            dm = np.diff(mag_w)
            dt = np.diff(tsec_w)
            valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dm)
            if np.any(valid):
                jerk = np.abs(dm[valid] / dt[valid])
                jpk = float(np.max(jerk)) if len(jerk) > 0 else 0.0
        jerk_peaks.append(jpk)

        mlag = (ts >= tk) & (ts <= tk + lag_max_ns)
        if np.any(mlag):
            idxs = np.where(mlag)[0]
            local = mag[idxs]
            j = int(idxs[int(np.argmax(local))])
            lags_ms.append(float((ts[j] - tk) / 1_000_000.0))

        pre_mask = (ts_w >= tk - pre_ns) & (ts_w < tk)
        post_mask = (ts_w >= tk) & (ts_w <= tk + post_ns)

        Epre = 0.0
        if np.any(pre_mask) and np.sum(pre_mask) >= 2:
            tpre = (ts_w[pre_mask] - ts_w[pre_mask][0]).astype(np.float64) / 1e9
            Epre = _integral_mag2_dt(mag_w[pre_mask], tpre)
        Epost = 0.0
        if np.any(post_mask) and np.sum(post_mask) >= 2:
            tpost = (ts_w[post_mask] - ts_w[post_mask][0]).astype(np.float64) / 1e9
            Epost = _integral_mag2_dt(mag_w[post_mask], tpost)
        post_pre_ratio.append(float(Epost / (Epre + EPS)))

        rt = 0.0
        if np.any(post_mask):
            ts_post = ts_w[post_mask]
            mag_post = mag_w[post_mask]
            thr = float(RISE_FRAC * pk)
            idx = np.where(mag_post >= thr)[0]
            if len(idx) > 0:
                rt = float((ts_post[int(idx[0])] - tk) / 1_000_000.0)
        rise_times_ms.append(rt)

    peaks_arr = np.array(peaks, dtype=np.float64)

    out[f"{prefix}_peak_per_char_mean"] = float(np.mean(peaks)) if len(peaks) > 0 else 0.0
    out[f"{prefix}_jerk_peak_per_char_mean"] = float(np.mean(jerk_peaks)) if len(jerk_peaks) > 0 else 0.0
    out[f"{prefix}_peak_lag_mean_ms"] = float(np.mean(lags_ms)) if len(lags_ms) > 0 else 0.0

    out[f"{prefix}_auc_per_char_mean"] = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
    out[f"{prefix}_peak_to_rms_ratio_mean"] = float(np.mean(peak_to_rms)) if len(peak_to_rms) > 0 else 0.0
    out[f"{prefix}_post_pre_energy_ratio_mean"] = float(np.mean(post_pre_ratio)) if len(post_pre_ratio) > 0 else 0.0
    out[f"{prefix}_rise_time_ms_mean"] = float(np.mean(rise_times_ms)) if len(rise_times_ms) > 0 else 0.0

    def _mean_peak_for_times(times_ns: np.ndarray) -> float:
        if times_ns is None or len(times_ns) == 0:
            return 0.0
        vals = []
        for tk in times_ns:
            a = tk - pre_ns
            b = tk + post_ns
            m = (ts >= a) & (ts <= b)
            if np.any(m):
                vals.append(float(np.max(mag[m])))
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    space_mean = _mean_peak_for_times(insert_space_times_ns)
    letter_mean = _mean_peak_for_times(insert_letter_times_ns)
    out[f"{prefix}_peak_space_minus_letter"] = float(space_mean - letter_mean)

    if len(insert_times_ns) >= 2:
        dd_ms = (insert_times_ns[1:] - insert_times_ns[:-1]) / 1_000_000.0
        valid_dd = (dd_ms > MIN_INTERVAL_MS) & (dd_ms <= MAX_INTERVAL_MS) & np.isfinite(dd_ms)

        dd_list = []
        e_list = []
        fast_e = 0.0
        slow_e = 0.0

        for i in range(1, len(insert_times_ns)):
            if not valid_dd[i - 1]:
                continue
            t0 = int(insert_times_ns[i - 1])
            t1 = int(insert_times_ns[i])

            mi = (ts >= t0) & (ts <= t1)
            if not np.any(mi):
                continue

            ts_i = ts[mi]
            mag_i = mag[mi]
            tsec_i = (ts_i - ts_i[0]).astype(np.float64) / 1_000_000_000.0

            e = _integral_mag2_dt(mag_i, tsec_i)
            d = float(dd_ms[i - 1])

            dd_list.append(d)
            e_list.append(e)

            if d < FAST_DD_MS:
                fast_e += e
            else:
                slow_e += e

        if len(dd_list) >= 3:
            c = float(np.corrcoef(np.array(dd_list), np.array(e_list))[0, 1])
            out[f"corr_dd_{prefix}_energy"] = 0.0 if np.isnan(c) else c
        else:
            out[f"corr_dd_{prefix}_energy"] = 0.0

        out[f"fast_slow_{prefix}_ratio"] = float(fast_e / (slow_e + EPS)) if (fast_e > 0 or slow_e > 0) else 0.0

    if len(ts) >= 8 and (ts[-1] > ts[0]):
        t_u, mag_u = resample_uniform(t_sec, mag, target_hz=IMU_RESAMPLE_HZ)
        if len(mag_u) >= max(32, IMU_MIN_FFT_SAMPLES // 2):
            dm = np.diff(mag_u, prepend=mag_u[0])
            proxy = np.abs(dm) * float(IMU_RESAMPLE_HZ)

            ins_sec = (insert_times_ns.astype(np.int64) - ts[0]) / 1_000_000_000.0
            ins_sec = ins_sec[np.isfinite(ins_sec)]
            ev = np.zeros(len(t_u), dtype=np.float64)
            if len(ins_sec) > 0:
                idx = np.searchsorted(t_u, ins_sec)
                idx = np.clip(idx, 0, len(t_u) - 1)
                ev[idx] = 1.0

            ev_z = _zscore(ev)
            pr_z = _zscore(proxy)

            max_lag = int(np.round((XCORR_MAX_LAG_MS / 1000.0) * IMU_RESAMPLE_HZ))
            if max_lag >= 1 and len(ev_z) > (2 * max_lag + 3):
                xc = np.correlate(pr_z, ev_z, mode="full")
                center = len(ev_z) - 1
                lo = center - max_lag
                hi = center + max_lag + 1
                xc_slice = xc[lo:hi]

                denom = float(len(ev_z))
                xc_slice = xc_slice / max(denom, 1.0)

                k = int(np.argmax(np.abs(xc_slice)))
                lag_samp = k - max_lag
                out[f"{prefix}_keystroke_xcorr_lag_ms"] = float(lag_samp * (1000.0 / IMU_RESAMPLE_HZ))
                out[f"{prefix}_keystroke_xcorr_peak"] = float(xc_slice[k])

    return out, peaks_arr


# -----------------------------
# Cross-modal: ACC <-> GYRO coupling
# -----------------------------
def compute_acc_gyro_coupling_features(
    acc_clean: pd.DataFrame,
    gyro_clean: pd.DataFrame,
    insert_times_ns: np.ndarray,
) -> dict:
    out = {
        "corr_acc_gyro_peak_per_char": 0.0,
        "gyro_to_acc_energy_ratio_per_char_mean": 0.0,
    }

    if acc_clean is None or len(acc_clean) == 0:
        return out
    if gyro_clean is None or len(gyro_clean) == 0:
        return out
    if insert_times_ns is None or len(insert_times_ns) == 0:
        return out

    a = acc_clean.sort_values("timestamp").copy()
    for c in ["timestamp", "x", "y", "z"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["timestamp", "x", "y", "z"])
    if len(a) == 0:
        return out
    ats = a["timestamp"].astype(np.int64).to_numpy()
    at_sec = (ats - ats[0]).astype(np.float64) / 1e9
    ax = a["x"].to_numpy(dtype=np.float64)
    ay = a["y"].to_numpy(dtype=np.float64)
    az = a["z"].to_numpy(dtype=np.float64)
    ax, ay, az = gravity_remove_ema(ax, ay, az, at_sec, tau=IMU_GRAVITY_TAU_SEC)
    amag = np.sqrt(ax*ax + ay*ay + az*az)

    g = gyro_clean.sort_values("timestamp").copy()
    for c in ["timestamp", "x", "y", "z"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["timestamp", "x", "y", "z"])
    if len(g) == 0:
        return out
    gts = g["timestamp"].astype(np.int64).to_numpy()
    gt_sec = (gts - gts[0]).astype(np.float64) / 1e9
    gx = g["x"].to_numpy(dtype=np.float64) - float(np.mean(g["x"].to_numpy(dtype=np.float64)))
    gy = g["y"].to_numpy(dtype=np.float64) - float(np.mean(g["y"].to_numpy(dtype=np.float64)))
    gz = g["z"].to_numpy(dtype=np.float64) - float(np.mean(g["z"].to_numpy(dtype=np.float64)))
    gmag = np.sqrt(gx*gx + gy*gy + gz*gz)

    n_chars = int(len(insert_times_ns))
    Eacc = _integral_mag2_dt(amag, at_sec)
    Egyro = _integral_mag2_dt(gmag, gt_sec)
    out["gyro_to_acc_energy_ratio_per_char_mean"] = float(
        (Egyro / max(n_chars, 1)) / ((Eacc / max(n_chars, 1)) + EPS)
    )

    pre_ns = int(ACC_KEY_PRE_MS * 1_000_000)
    post_ns = int(ACC_KEY_POST_MS * 1_000_000)

    acc_peaks = []
    gyro_peaks = []

    for tk in insert_times_ns:
        ma = (ats >= tk - pre_ns) & (ats <= tk + post_ns)
        mg = (gts >= tk - pre_ns) & (gts <= tk + post_ns)
        if np.any(ma) and np.any(mg):
            acc_peaks.append(float(np.max(amag[ma])))
            gyro_peaks.append(float(np.max(gmag[mg])))

    if len(acc_peaks) >= 3 and len(gyro_peaks) >= 3:
        c = float(np.corrcoef(np.array(acc_peaks), np.array(gyro_peaks))[0, 1])
        out["corr_acc_gyro_peak_per_char"] = 0.0 if np.isnan(c) else c

    return out

import os
import argparse
import pandas as pd
import numpy as np

# =========================================================
# SEM NECHAJ ÚPLNE BEZ ZMENY:
# - všetky config konštanty
# - filter_intervals_ms
# - safe_skew
# - safe_kurt
# - series_stats
# - build_active_segments_from_keystrokes
# - assign_round_ids_by_bounds
# - filter_sensor_by_segments
# - load_sensor_csv
# - estimate_sampling_stats
# - gravity_remove_ema
# - resample_uniform
# - spectral_entropy_from_signal
# - _zscore
# - _integral_mag2_dt
# - compute_global_sensor_features
# - compute_cross_imu_keystroke_features
# - compute_acc_gyro_coupling_features
#
# Tieto funkcie MUSIA zostať identické s pôvodným skriptom,
# ak chceš identický výsledný vektor.
# =========================================================

VOWELS = set(list("aeiouy"))


def build_single_vector(
    uid: str,
    keystrokes_path: str,
    acc_path: str,
    gyro_path: str,
    output_path: str,
    round_id: int | None = None,
) -> pd.DataFrame:
    # -----------------------------
    # 1) Load keystrokes
    # -----------------------------
    if not os.path.exists(keystrokes_path):
        raise FileNotFoundError(f"Chýba keystrokes súbor: {keystrokes_path}")

    df = pd.read_csv(keystrokes_path)
    for col in ["TimestampBeforeNs", "TimestampAfterNs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "RoundId" not in df.columns:
        raise ValueError("Keystrokes CSV neobsahuje stĺpec RoundId.")

    available_rounds = sorted(pd.to_numeric(df["RoundId"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(available_rounds) == 0:
        raise ValueError("V keystrokes CSV nie sú žiadne validné RoundId.")

    # Ak round_id nie je zadané, zoberieme posledné / najnovšie round
    selected_round_id = int(round_id) if round_id is not None else int(max(available_rounds))

    if selected_round_id not in available_rounds:
        raise ValueError(
            f"RoundId={selected_round_id} sa v súbore nenachádza. Dostupné RoundId: {available_rounds}"
        )

    df = df[df["RoundId"] == selected_round_id].copy()
    if len(df) == 0:
        raise ValueError(f"Po odfiltrovaní RoundId={selected_round_id} nezostali žiadne riadky.")

    # -----------------------------
    # 2) Build single keystroke vector
    # -----------------------------
    master = []

    segments_by_round: dict[int, list[tuple[int, int]]] = {}
    round_bounds_list: list[dict] = []

    insert_times_by_round: dict[int, np.ndarray] = {}
    insert_space_times_by_round: dict[int, np.ndarray] = {}
    insert_letter_times_by_round: dict[int, np.ndarray] = {}

    for rid, group in df.groupby("RoundId", sort=True):
        veta = group.sort_values("TimestampBeforeNs").reset_index(drop=True)

        if len(veta) > 0:
            start_ns = int(pd.to_numeric(veta["TimestampBeforeNs"], errors="coerce").dropna().min())
            end_ns = int(pd.to_numeric(veta["TimestampAfterNs"], errors="coerce").dropna().max())
            round_bounds_list.append({"RoundId": int(rid), "start_ns": start_ns, "end_ns": end_ns})
        else:
            continue

        segments_by_round[int(rid)] = build_active_segments_from_keystrokes(veta, max_gap_ms=MAX_INTERVAL_MS)

        pd_values_raw = (veta["TimestampAfterNs"] - veta["TimestampBeforeNs"]) / 1_000_000
        ft_values_raw = (veta["TimestampBeforeNs"] - veta["TimestampAfterNs"].shift(1)) / 1_000_000
        dd_values_raw = (veta["TimestampBeforeNs"] - veta["TimestampBeforeNs"].shift(1)) / 1_000_000
        uu_values_raw = (veta["TimestampAfterNs"] - veta["TimestampAfterNs"].shift(1)) / 1_000_000

        ft_values_raw = ft_values_raw.dropna()
        dd_values_raw = dd_values_raw.dropna()
        uu_values_raw = uu_values_raw.dropna()

        pd_before = len(pd_values_raw.dropna())
        ft_before = len(ft_values_raw)
        dd_before = len(dd_values_raw)
        uu_before = len(uu_values_raw)

        pd_values = pd_values_raw.dropna()
        pd_values = pd_values[pd_values > MIN_INTERVAL_MS]

        ft_values = filter_intervals_ms(ft_values_raw, MIN_INTERVAL_MS, MAX_INTERVAL_MS)
        dd_values = filter_intervals_ms(dd_values_raw, MIN_INTERVAL_MS, MAX_INTERVAL_MS)
        uu_values = filter_intervals_ms(uu_values_raw, MIN_INTERVAL_MS, MAX_INTERVAL_MS)

        pd_after = len(pd_values)
        ft_after = len(ft_values)
        dd_after = len(dd_values)
        uu_after = len(uu_values)

        effective_typing_time_sec = float(dd_values.sum()) / 1000.0
        num_events = pd_after
        cps = (num_events / effective_typing_time_sec) if effective_typing_time_sec > 0 else 0.0
        wpm = (cps * 60.0) / 5.0
        total_duration = float(effective_typing_time_sec)

        final_content = veta["InputContent"].iloc[-1] if "InputContent" in veta.columns and len(veta) > 0 else ""
        final_length = len(str(final_content))
        typing_efficiency = (final_length / num_events) if num_events > 0 else 0.0

        action = veta["ActionType"].astype(str) if "ActionType" in veta.columns else pd.Series([], dtype=str)
        action_l = action.str.lower()
        insert_mask = (action_l == "insert")
        delete_mask = (action_l == "delete")

        if "TimestampBeforeNs" in veta.columns and len(action_l) > 0:
            ins_times = veta.loc[insert_mask, "TimestampBeforeNs"]
            ins_times = pd.to_numeric(ins_times, errors="coerce").dropna().astype(np.int64).to_numpy()
            ins_times = np.sort(ins_times)
        else:
            ins_times = np.array([], dtype=np.int64)
        insert_times_by_round[int(rid)] = ins_times

        keychar = veta["KeyChar"].astype(str) if "KeyChar" in veta.columns else pd.Series([""] * len(veta), dtype=str)
        keychar_l = keychar.str.lower()

        if len(ins_times) > 0 and "KeyChar" in veta.columns:
            ins_idx = np.where(insert_mask.to_numpy())[0]
            ins_keychar = keychar.iloc[ins_idx].astype(str).to_numpy()

            ins_before_ns = pd.to_numeric(veta["TimestampBeforeNs"].iloc[ins_idx], errors="coerce").dropna().astype(np.int64).to_numpy()
            mlen = min(len(ins_before_ns), len(ins_keychar))
            ins_before_ns = ins_before_ns[:mlen]
            ins_keychar = ins_keychar[:mlen]

            space_ns = ins_before_ns[ins_keychar == " "]
            kc = np.array([str(x).lower() for x in ins_keychar], dtype=object)
            letter_mask = np.array([(len(x) == 1 and x.isalpha()) for x in kc], dtype=bool)
            letter_ns = ins_before_ns[letter_mask]

            insert_space_times_by_round[int(rid)] = np.sort(space_ns.astype(np.int64)) if len(space_ns) > 0 else np.array([], dtype=np.int64)
            insert_letter_times_by_round[int(rid)] = np.sort(letter_ns.astype(np.int64)) if len(letter_ns) > 0 else np.array([], dtype=np.int64)
        else:
            insert_space_times_by_round[int(rid)] = np.array([], dtype=np.int64)
            insert_letter_times_by_round[int(rid)] = np.array([], dtype=np.int64)

        insert_count = int(insert_mask.sum()) if len(action_l) > 0 else 0
        backspace_count = int(delete_mask.sum()) if len(action_l) > 0 else 0
        error_rate = (backspace_count / insert_count) if insert_count > 0 else 0.0

        corr_times_ms = []
        if len(veta) > 0 and "TimestampBeforeNs" in veta.columns and "TimestampAfterNs" in veta.columns and len(action_l) > 0:
            before_ns = veta["TimestampBeforeNs"].to_numpy()
            after_ns = veta["TimestampAfterNs"].to_numpy()
            insert_idx = np.where(insert_mask.to_numpy())[0]
            delete_idx = np.where(delete_mask.to_numpy())[0]

            if len(insert_idx) > 0 and len(delete_idx) > 0:
                for d in delete_idx:
                    prev_inserts = insert_idx[insert_idx < d]
                    if len(prev_inserts) == 0:
                        continue
                    i = int(prev_inserts[-1])
                    dt_ms = (before_ns[d] - after_ns[i]) / 1_000_000
                    if (dt_ms > MIN_INTERVAL_MS) and (dt_ms <= MAX_INTERVAL_MS):
                        corr_times_ms.append(float(dt_ms))
        mean_corr_time = float(np.mean(corr_times_ms)) if len(corr_times_ms) > 0 else 0.0

        burst_sizes = []
        if len(action_l) > 0:
            current = 0
            for is_insert in insert_mask.to_numpy():
                if is_insert:
                    current += 1
                else:
                    if current > 0:
                        burst_sizes.append(current)
                        current = 0
            if current > 0:
                burst_sizes.append(current)

        mean_burst_size = float(np.mean(burst_sizes)) if len(burst_sizes) > 0 else 0.0
        burst_count = int(len(burst_sizes))
        max_burst_size = int(max(burst_sizes)) if len(burst_sizes) > 0 else 0

        space_mask = (keychar == " ")
        space_pd_series = pd_values_raw[space_mask].dropna()
        space_pd_series = space_pd_series[space_pd_series > MIN_INTERVAL_MS]
        space_pd = float(space_pd_series.mean()) if len(space_pd_series) > 0 else 0.0

        space_ft_series = (veta["TimestampBeforeNs"] - veta["TimestampAfterNs"].shift(1)) / 1_000_000
        space_ft_series = space_ft_series[space_mask].dropna()
        space_ft_series = space_ft_series[(space_ft_series > MIN_INTERVAL_MS) & (space_ft_series <= MAX_INTERVAL_MS)]
        space_ft = float(space_ft_series.mean()) if len(space_ft_series) > 0 else 0.0

        is_alpha_single = keychar_l.str.len().eq(1) & keychar_l.str.match(r"[a-z]", na=False)
        vowel_mask = is_alpha_single & keychar_l.isin(VOWELS)
        consonant_mask = is_alpha_single & (~keychar_l.isin(VOWELS))

        vowel_pd_series = pd_values_raw[vowel_mask].dropna()
        vowel_pd_series = vowel_pd_series[vowel_pd_series > MIN_INTERVAL_MS]
        vowel_pd = float(vowel_pd_series.mean()) if len(vowel_pd_series) > 0 else 0.0

        consonant_pd_series = pd_values_raw[consonant_mask].dropna()
        consonant_pd_series = consonant_pd_series[consonant_pd_series > MIN_INTERVAL_MS]
        consonant_pd = float(consonant_pd_series.mean()) if len(consonant_pd_series) > 0 else 0.0

        is_single = keychar.str.len().eq(1)
        non_alnum_mask = is_single & (~keychar.apply(lambda x: str(x).isalnum()))
        special_pd_series = pd_values_raw[non_alnum_mask].dropna()
        special_pd_series = special_pd_series[special_pd_series > MIN_INTERVAL_MS]
        special_pd = float(special_pd_series.mean()) if len(special_pd_series) > 0 else 0.0

        capital_mask = is_single & keychar.apply(lambda x: str(x).isupper())
        capital_pd_series = pd_values_raw[capital_mask].dropna()
        capital_pd_series = capital_pd_series[capital_pd_series > MIN_INTERVAL_MS]
        capital_pd = float(capital_pd_series.mean()) if len(capital_pd_series) > 0 else 0.0

        long_pause_count = int((ft_values_raw > 1000.0).sum()) if len(ft_values_raw) > 0 else 0

        dd_mean_val = float(dd_values.mean()) if len(dd_values) > 0 else 0.0
        dd_std_val = float(dd_values.std(ddof=1)) if len(dd_values) >= 2 else 0.0
        dd_cv_ratio = (dd_std_val / dd_mean_val) if dd_mean_val > 0 else 0.0

        dd_q25 = float(dd_values.quantile(0.25)) if len(dd_values) > 0 else 0.0
        dd_q75 = float(dd_values.quantile(0.75)) if len(dd_values) > 0 else 0.0
        dd_iqr = float(dd_q75 - dd_q25) if len(dd_values) > 0 else 0.0

        micro_pause_count = int(((dd_values_raw > MICRO_PAUSE_MS) & (dd_values_raw <= MAX_INTERVAL_MS)).sum()) if len(dd_values_raw) > 0 else 0

        dd_vc, dd_cv, dd_cc, dd_ls, dd_sl = [], [], [], [], []
        dd_trend_slope = 0.0

        if len(veta) > 1 and "TimestampBeforeNs" in veta.columns and "ActionType" in veta.columns and "KeyChar" in veta.columns:
            ins_idx = np.where(insert_mask.to_numpy())[0]
            if len(ins_idx) >= 2:
                before_ins = veta["TimestampBeforeNs"].iloc[ins_idx].to_numpy()
                ch_ins = keychar.iloc[ins_idx].astype(str)
                ch_ins_l = ch_ins.str.lower()

                dd_ins = (before_ins[1:] - before_ins[:-1]) / 1_000_000

                dd_ins_cleaned = []
                for k in range(len(dd_ins)):
                    dt = float(dd_ins[k])
                    if not ((dt > MIN_INTERVAL_MS) and (dt <= MAX_INTERVAL_MS)):
                        continue
                    dd_ins_cleaned.append(dt)

                    c1 = str(ch_ins.iloc[k])
                    c2 = str(ch_ins.iloc[k + 1])
                    c1_l = str(ch_ins_l.iloc[k])
                    c2_l = str(ch_ins_l.iloc[k + 1])

                    is_space_1 = (c1 == " ")
                    is_space_2 = (c2 == " ")

                    if (not is_space_1) and is_space_2:
                        dd_ls.append(dt)
                        continue
                    if is_space_1 and (not is_space_2):
                        dd_sl.append(dt)
                        continue

                    is_a1 = (len(c1_l) == 1) and (c1_l.isalpha())
                    is_a2 = (len(c2_l) == 1) and (c2_l.isalpha())
                    if is_a1 and is_a2:
                        is_v1 = c1_l in VOWELS
                        is_v2 = c2_l in VOWELS
                        if is_v1 and (not is_v2):
                            dd_vc.append(dt)
                        elif (not is_v1) and is_v2:
                            dd_cv.append(dt)
                        elif (not is_v1) and (not is_v2):
                            dd_cc.append(dt)

                if len(dd_ins_cleaned) >= 2:
                    x_ = np.arange(len(dd_ins_cleaned), dtype=float)
                    dd_trend_slope = float(np.polyfit(x_, np.array(dd_ins_cleaned, dtype=float), 1)[0])

        dd_vc = pd.Series(dd_vc, dtype=float)
        dd_cv = pd.Series(dd_cv, dtype=float)
        dd_cc = pd.Series(dd_cc, dtype=float)
        dd_ls = pd.Series(dd_ls, dtype=float)
        dd_sl = pd.Series(dd_sl, dtype=float)

        n_inserts = int(insert_count)
        n_dd_valid = int(dd_after)
        n_ft_valid = int(ft_after)
        n_uu_valid = int(uu_after)

        backspace_per_char = float(backspace_count / max(n_inserts, 1))
        burst_count_per_char = float(burst_count / max(n_inserts, 1))
        long_pause_per_char = float(long_pause_count / max(n_inserts, 1))
        micro_pause_per_char = float(micro_pause_count / max(n_inserts, 1))

        dd_valid_ratio = float(n_dd_valid / max(dd_before, 1))
        ft_valid_ratio = float(n_ft_valid / max(ft_before, 1))
        uu_valid_ratio = float(n_uu_valid / max(uu_before, 1))

        features = {
            "UserId": uid,
            "RoundId": int(rid),

            "cps": float(cps),
            "wpm": float(wpm),
            "total_duration": float(total_duration),
            "typing_efficiency": float(typing_efficiency),

            "error_rate": float(error_rate),
            "backspace_count": int(backspace_count),
            "mean_corr_time": float(mean_corr_time),
            "mean_burst_size": float(mean_burst_size),
            "burst_count": int(burst_count),
            "max_burst_size": int(max_burst_size),

            "n_inserts": int(n_inserts),
            "n_dd_valid": int(n_dd_valid),
            "n_ft_valid": int(n_ft_valid),
            "n_uu_valid": int(n_uu_valid),
            "dd_valid_ratio": float(dd_valid_ratio),
            "ft_valid_ratio": float(ft_valid_ratio),
            "uu_valid_ratio": float(uu_valid_ratio),

            "backspace_per_char": float(backspace_per_char),
            "burst_count_per_char": float(burst_count_per_char),
            "long_pause_per_char": float(long_pause_per_char),
            "micro_pause_per_char": float(micro_pause_per_char),

            "space_pd": float(space_pd),
            "space_ft": float(space_ft),
            "vowel_pd": float(vowel_pd),
            "consonant_pd": float(consonant_pd),
            "special_pd": float(special_pd),
            "capital_pd": float(capital_pd),
            "long_pause_count": int(long_pause_count),
            "dd_cv_ratio": float(dd_cv_ratio),
            "dd_iqr": float(dd_iqr),
            "micro_pause_count": int(micro_pause_count),
            "dd_trend_slope": float(dd_trend_slope),

            "dd_vc_mean": float(dd_vc.mean()) if len(dd_vc) > 0 else 0.0,
            "dd_vc_std": float(dd_vc.std(ddof=1)) if len(dd_vc) >= 2 else 0.0,
            "dd_vc_median": float(dd_vc.median()) if len(dd_vc) > 0 else 0.0,
            "dd_vc_count": int(len(dd_vc)),

            "dd_cv_mean": float(dd_cv.mean()) if len(dd_cv) > 0 else 0.0,
            "dd_cv_std": float(dd_cv.std(ddof=1)) if len(dd_cv) >= 2 else 0.0,
            "dd_cv_median": float(dd_cv.median()) if len(dd_cv) > 0 else 0.0,
            "dd_cv_count": int(len(dd_cv)),

            "dd_cc_mean": float(dd_cc.mean()) if len(dd_cc) > 0 else 0.0,
            "dd_cc_std": float(dd_cc.std(ddof=1)) if len(dd_cc) >= 2 else 0.0,
            "dd_cc_median": float(dd_cc.median()) if len(dd_cc) > 0 else 0.0,
            "dd_cc_count": int(len(dd_cc)),

            "dd_ls_mean": float(dd_ls.mean()) if len(dd_ls) > 0 else 0.0,
            "dd_ls_std": float(dd_ls.std(ddof=1)) if len(dd_ls) >= 2 else 0.0,
            "dd_ls_median": float(dd_ls.median()) if len(dd_ls) > 0 else 0.0,
            "dd_ls_count": int(len(dd_ls)),

            "dd_sl_mean": float(dd_sl.mean()) if len(dd_sl) > 0 else 0.0,
            "dd_sl_std": float(dd_sl.std(ddof=1)) if len(dd_sl) >= 2 else 0.0,
            "dd_sl_median": float(dd_sl.median()) if len(dd_sl) > 0 else 0.0,
            "dd_sl_count": int(len(dd_sl)),
        }

        features.update(series_stats("pd", pd_values))
        features.update(series_stats("ft", ft_values))
        features.update(series_stats("dd", dd_values))
        features.update(series_stats("uu", uu_values))

        master.append(features)

    master_df = pd.DataFrame(master)
    if len(master_df) == 0:
        raise ValueError("Nevznikol žiadny keystroke vektor.")

    # -----------------------------
    # 3) Load acc/gyro -> assign RoundId -> FILTER -> compute -> merge
    # -----------------------------
    round_bounds_df = pd.DataFrame(round_bounds_list).sort_values("RoundId").reset_index(drop=True)

    acc_global_df = pd.DataFrame(columns=["RoundId"])
    gyro_global_df = pd.DataFrame(columns=["RoundId"])
    coupling_df = pd.DataFrame(columns=["RoundId"])

    acc_clean_by_round: dict[int, pd.DataFrame] = {}
    gyro_clean_by_round: dict[int, pd.DataFrame] = {}

    if os.path.exists(acc_path):
        acc_df = load_sensor_csv(acc_path)
        acc_df = assign_round_ids_by_bounds(acc_df, round_bounds_df)

        acc_rows = []
        for rid, g in acc_df.groupby("RoundId", sort=True):
            segs = segments_by_round.get(int(rid), [])
            g_clean = filter_sensor_by_segments(g, segs)
            acc_clean_by_round[int(rid)] = g_clean

            acc_cov = float(len(g_clean) / max(len(g), 1))
            row = {"RoundId": int(rid), "acc_cov": acc_cov}
            row.update(compute_global_sensor_features(g_clean, prefix="acc"))

            ins_times = insert_times_by_round.get(int(rid), np.array([], dtype=np.int64))
            ins_space = insert_space_times_by_round.get(int(rid), np.array([], dtype=np.int64))
            ins_letter = insert_letter_times_by_round.get(int(rid), np.array([], dtype=np.int64))

            cross_acc, _ = compute_cross_imu_keystroke_features(
                g_clean, ins_times, ins_space, ins_letter,
                prefix="acc", is_acc=True
            )
            row.update(cross_acc)
            acc_rows.append(row)

        if len(acc_rows) > 0:
            acc_global_df = pd.DataFrame(acc_rows)

    if os.path.exists(gyro_path):
        gyro_df = load_sensor_csv(gyro_path)
        gyro_df = assign_round_ids_by_bounds(gyro_df, round_bounds_df)

        gyro_rows = []
        for rid, g in gyro_df.groupby("RoundId", sort=True):
            segs = segments_by_round.get(int(rid), [])
            g_clean = filter_sensor_by_segments(g, segs)
            gyro_clean_by_round[int(rid)] = g_clean

            gyro_cov = float(len(g_clean) / max(len(g), 1))
            row = {"RoundId": int(rid), "gyro_cov": gyro_cov}
            row.update(compute_global_sensor_features(g_clean, prefix="gyro"))

            ins_times = insert_times_by_round.get(int(rid), np.array([], dtype=np.int64))
            ins_space = insert_space_times_by_round.get(int(rid), np.array([], dtype=np.int64))
            ins_letter = insert_letter_times_by_round.get(int(rid), np.array([], dtype=np.int64))

            cross_gyro, _ = compute_cross_imu_keystroke_features(
                g_clean, ins_times, ins_space, ins_letter,
                prefix="gyro", is_acc=False
            )
            row.update(cross_gyro)
            gyro_rows.append(row)

        if len(gyro_rows) > 0:
            gyro_global_df = pd.DataFrame(gyro_rows)

    coupling_rows = []
    common_rids = sorted(set(acc_clean_by_round.keys()) & set(gyro_clean_by_round.keys()))
    for rid in common_rids:
        ins_times = insert_times_by_round.get(int(rid), np.array([], dtype=np.int64))
        row = {"RoundId": int(rid)}
        row.update(compute_acc_gyro_coupling_features(
            acc_clean_by_round[int(rid)],
            gyro_clean_by_round[int(rid)],
            ins_times
        ))
        coupling_rows.append(row)

    if len(coupling_rows) > 0:
        coupling_df = pd.DataFrame(coupling_rows)

    master_df = master_df.merge(acc_global_df, on="RoundId", how="left")
    master_df = master_df.merge(gyro_global_df, on="RoundId", how="left")
    master_df = master_df.merge(coupling_df, on="RoundId", how="left")

    sensor_cols = [
        c for c in master_df.columns
        if c.startswith("acc_") or c.startswith("gyro_") or c.startswith("corr_")
        or c.startswith("fast_slow_") or c.startswith("gyro_to_acc_") or c.startswith("corr_acc_gyro_")
        or c in ["acc_cov", "gyro_cov"]
    ]
    for c in sensor_cols:
        master_df[c] = pd.to_numeric(master_df[c], errors="coerce").fillna(0.0)

    # Tu očakávame práve 1 riadok
    if len(master_df) != 1:
        raise ValueError(
            f"Očakával som 1 výsledný vektor, ale vzniklo {len(master_df)} riadkov. "
            f"Skontroluj RoundId v keystrokes súbore."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    master_df.to_csv(output_path, index=False)

    return master_df


def main():
    parser = argparse.ArgumentParser(description="Build single authentication feature vector from 3 CSV files.")
    parser.add_argument("--uid", required=True, help="User ID, ktorý sa zapíše do výsledného vektora.")
    parser.add_argument("--keystrokes", required=True, help="Path to keystrokes CSV")
    parser.add_argument("--accelerometer", required=True, help="Path to accelerometer CSV")
    parser.add_argument("--gyroscope", required=True, help="Path to gyroscope CSV")
    parser.add_argument("--output", required=True, help="Path to output vector CSV")
    parser.add_argument("--round-id", type=int, default=None, help="Optional RoundId. Ak nie je zadané, použije sa posledné RoundId.")

    args = parser.parse_args()

    df_out = build_single_vector(
        uid=args.uid,
        keystrokes_path=args.keystrokes,
        acc_path=args.accelerometer,
        gyro_path=args.gyroscope,
        output_path=args.output,
        round_id=args.round_id,
    )

    print(f"✓ Hotovo. Výsledný vektor má tvar: {df_out.shape}")
    print(f"✓ Uložené do: {args.output}")


if __name__ == "__main__":
    main()