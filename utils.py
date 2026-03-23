

import numpy as np
from scipy.optimize import curve_fit


def to_numpy(array_like, dtype=None, copy=False):
    """Convert torch/numpy/array-like inputs to a numpy array."""
    if array_like is None:
        return None
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    arr = np.asarray(array_like, dtype=dtype)
    if copy:
        arr = np.array(arr, dtype=arr.dtype, copy=True)
    return arr

def pq_to_xy(p, q):
    """
    Convert hex-grid (p, q) coordinates to plotting coordinates (x, y).
    Works with scalars or numpy-like arrays.
    """
    p_arr = to_numpy(p, dtype=float)
    q_arr = to_numpy(q, dtype=float)
    x = q_arr - p_arr
    y = (p_arr + q_arr) / np.sqrt(3.0)
    return x, y

def _normalize_windows(windows):
    """
    Normalize windows into a list of (start, end) tuples.

    Accepts:
    - (start, end)
    - [(start, end), ...]
    """
    if windows is None:
        return None

    if isinstance(windows, tuple) and len(windows) == 2:
        return [(float(windows[0]), float(windows[1]))]

    normalized = list(windows)
    if not normalized:
        raise ValueError("`windows` cannot be empty.")
    for w in normalized:
        if len(w) != 2:
            raise ValueError("Each window must have exactly two values: (start, end).")
    return [(float(start), float(end)) for start, end in normalized]

def _mask_from_windows(t, windows):
    """Build a boolean mask over t for one or more [start, end) windows."""
    mask = np.zeros_like(t, dtype=bool)
    for start, end in windows:
        if end <= start:
            raise ValueError(f"Invalid window ({start}, {end}): end must be > start.")
        mask |= (t >= start) & (t < end)
    return mask

def _reduce_over_time(values, reduction):
    reduction = str(reduction).lower()
    if reduction == "mean":
        return np.mean(values, axis=0)
    if reduction == "peak" or reduction == "max":
        return np.max(values, axis=0)
    if reduction == "min":
        return np.min(values, axis=0)
    if reduction == "sum":
        return np.sum(values, axis=0)
    if reduction == "median":
        return np.median(values, axis=0)
    raise ValueError(
        "reduction must be one of {'mean', 'peak', 'max', 'min', 'sum', 'median'}."
    )

def _normalize_time_major(values, t=None):
    """
    Normalize activity history to shape (T, N), optionally aligned to t.
    Supports input shapes:
    - (T, N)
    - (1, T, N)
    - (N, T) if t is provided (auto-transposed)
    - (T,) -> (T, 1)
    """
    v = to_numpy(values)
    if v.ndim == 3:
        if v.shape[0] != 1:
            raise ValueError(
                "3D activity history must have shape (1, T, N). "
                f"Got {v.shape}."
            )
        v = v[0]
    elif v.ndim == 1:
        v = v[:, None]
    elif v.ndim != 2:
        raise ValueError(f"Activity history must be 1D, 2D, or 3D. Got shape {v.shape}.")

    if t is None:
        return v, None

    t_np = to_numpy(t).reshape(-1)
    if v.shape[0] != t_np.size:
        if v.shape[1] == t_np.size:
            v = v.T
        else:
            raise ValueError(
                "Time dimension mismatch between activity history and t: "
                f"history shape {v.shape}, len(t)={t_np.size}."
            )
    return v, t_np

def _normalize_final_values(final_values):
    final_v = to_numpy(final_values)
    if final_v.ndim == 2 and final_v.shape[0] == 1:
        final_v = final_v[0]
    elif final_v.ndim > 1:
        final_v = np.ravel(final_v)
    return final_v

def compute_response_metric(
    values=None,
    t=None,
    *,
    metric="steady",
    use_relu=True,
    steady_window=0.0,
    last_n=10,
    windows=None,
    baseline_windows=None,
    window_reduction="mean",
    baseline_reduction="mean",
    final_values=None,
    return_components=False,
):
    """
    Compute per-neuron response metrics from activity history.

    Parameters
    ----------
    values : array-like, optional
        Activity history. Supported shapes: (T, N), (1, T, N), (N, T), or (T,).
    t : array-like, optional
        Time axis for time-window metrics.
    metric : str
        One of:
        - 'peak': max across time
        - 'mean': mean across time
        - 'final': last timepoint
        - 'steady': mean over last steady_window seconds or last_n samples
        - 'window_delta': reduction(windows) - reduction(baseline_windows)
    use_relu : bool
        If True, apply relu to activity before metric computation.
    steady_window : float
        Time span for 'steady' metric when t is available.
    last_n : int
        Fallback window size for 'steady' metric.
    windows : tuple | sequence of tuples, optional
        Target windows for 'window_delta'.
    baseline_windows : tuple | sequence of tuples, optional
        Baseline windows for 'window_delta'. If None, uses complement of windows.
    window_reduction : str
        Reduction for target window(s): mean/peak/max/min/sum/median.
    baseline_reduction : str
        Reduction for baseline window(s): mean/peak/max/min/sum/median.
    final_values : array-like, optional
        Fallback per-neuron values when values is unavailable.
    return_components : bool
        If True, return intermediate arrays/masks in a dictionary.

    Returns
    -------
    np.ndarray or dict
        Per-neuron scores, or a dict with score/components when
        return_components=True.
    """
    metric = str(metric).lower()
    allowed = {"peak", "mean", "final", "steady", "window_delta"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}. Got {metric!r}.")

    if values is None:
        if final_values is None:
            raise ValueError("Provide either `values` or `final_values`.")
        scores = _normalize_final_values(final_values)
        if use_relu:
            scores = np.maximum(scores, 0.0)
        if return_components:
            return {
                "scores": scores,
                "values": None,
                "t": None,
                "window_values": None,
                "baseline_values": None,
                "window_mask": None,
                "baseline_mask": None,
            }
        return scores

    v, t_np = _normalize_time_major(values, t=t)
    if use_relu:
        v = np.maximum(v, 0.0)

    window_values = None
    baseline_values = None
    window_mask = None
    baseline_mask = None

    if metric == "peak":
        scores = _reduce_over_time(v, "peak")
    elif metric == "mean":
        scores = _reduce_over_time(v, "mean")
    elif metric == "final":
        scores = v[-1]
    elif metric == "steady":
        if steady_window and t_np is not None:
            t_end = t_np[-1]
            mask = t_np >= (t_end - float(steady_window))
            if np.any(mask):
                scores = _reduce_over_time(v[mask], "mean")
            else:
                last_n = max(1, int(last_n))
                scores = _reduce_over_time(v[-last_n:], "mean")
        else:
            last_n = max(1, int(last_n))
            scores = _reduce_over_time(v[-last_n:], "mean")
    else:
        if t_np is None:
            raise ValueError("`t` is required for metric='window_delta'.")
        normalized_windows = _normalize_windows(windows)
        if normalized_windows is None:
            raise ValueError("`windows` must be provided for metric='window_delta'.")
        window_mask = _mask_from_windows(t_np, normalized_windows)
        if not np.any(window_mask):
            raise ValueError("No timepoints selected by `windows`.")

        if baseline_windows is None:
            baseline_mask = ~window_mask
        else:
            normalized_baseline = _normalize_windows(baseline_windows)
            baseline_mask = _mask_from_windows(t_np, normalized_baseline)
        if not np.any(baseline_mask):
            raise ValueError("No timepoints selected by baseline windows.")

        window_values = _reduce_over_time(v[window_mask], window_reduction)
        baseline_values = _reduce_over_time(v[baseline_mask], baseline_reduction)
        scores = window_values - baseline_values

    if return_components:
        return {
            "scores": scores,
            "values": v,
            "t": t_np,
            "window_values": window_values,
            "baseline_values": baseline_values,
            "window_mask": window_mask,
            "baseline_mask": baseline_mask,
        }
    return scores

def _von_mises_deg(theta_deg, baseline, amplitude, kappa, mu_deg, period_deg):
    delta = 2.0 * np.pi * (theta_deg - mu_deg) / period_deg
    return baseline + amplitude * np.exp(kappa * (np.cos(delta) - 1.0))

def fit_von_mises(angles_deg, values, period_deg=180.0, n_fit_points=361):
    """
    Von Mises fit for tuning curves.
    Returns dict with fitted params, y at data angles, and smooth fit curve.
    The fit curve is sampled over the observed angle range.
    """
    x = np.asarray(angles_deg, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 4 or period_deg <= 0:
        return None

    mu_guess = float(x[np.argmax(y)] % period_deg)
    p0 = [float(np.min(y)), float(np.ptp(y)), 1.0, mu_guess]
    bounds = (
        [-np.inf, 0.0, 0.0, 0.0],
        [np.inf, np.inf, 50.0, float(period_deg)],
    )
    try:
        popt, _ = curve_fit(
            lambda theta, baseline, amplitude, kappa, mu_deg: _von_mises_deg(
                theta, baseline, amplitude, kappa, mu_deg, period_deg
            ),
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        return None

    baseline, amplitude, kappa, mu_deg = popt
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), int(n_fit_points))
    y_fit = _von_mises_deg(x, baseline, amplitude, kappa, mu_deg, period_deg)
    y_fit_dense = _von_mises_deg(x_fit, baseline, amplitude, kappa, mu_deg, period_deg)
    return {
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "kappa": float(kappa),
        "mu_deg": float(mu_deg % period_deg),
        "period_deg": float(period_deg),
        "x_fit": x_fit,
        "y_fit": y_fit,
        "y_fit_dense": y_fit_dense,
    }

def fwhm(kappa, period_deg):
    cos_val = np.log(0.5) / kappa + 1.0
    if np.ndim(cos_val) == 0:
        if cos_val < -1.0:
            return period_deg
        return period_deg * np.arccos(cos_val) / (2.0 * np.pi)
    fwhm = np.where(cos_val < -1.0, period_deg, period_deg * np.arccos(np.clip(cos_val, -1.0, 1.0)) / (2.0 * np.pi))
    hwhm = fwhm / 2.0
    return (fwhm, hwhm)


def compute_pq_vector(angle: float, spatial_frequency: float = 0.1):
    """
    Compute the (p, q) wave vector components for a given orientation angle
    and spatial frequency on the hex grid.
    """
    p = -np.cos(np.radians(angle)) + np.sin(np.radians(angle)) / np.sqrt(3)
    q = np.cos(np.radians(angle)) + np.sin(np.radians(angle)) / np.sqrt(3)
    return spatial_frequency * p, spatial_frequency * q


def create_sine_grating(
    cell_ids,
    tm1_coords,
    n_cells,
    angle: float,
    spatial_frequency: float,
    phase: float,
    amplitude: float,
    offset: float,
    center: tuple | None = None,
) -> np.ndarray:
    """
    Create a sine grating stimulus on the hex grid.

    Args:
        cell_ids: ordered list of Tm1 cell IDs
        tm1_coords: dict mapping cell_id -> (p, q)
        n_cells: total number of Tm1 cells in the network
        angle: grating orientation in degrees
        spatial_frequency: spatial frequency of the grating
        phase: phase offset in radians
        amplitude: grating amplitude
        offset: DC offset added to each sample
        center: (p0, q0) reference point; defaults to (0, 0)
    """
    kp, kq = compute_pq_vector(angle, spatial_frequency)
    p0, q0 = center if center is not None else (0.0, 0.0)
    grating = np.zeros(n_cells, dtype=float)
    for idx, cell_id in enumerate(cell_ids):
        if cell_id not in tm1_coords:
            continue
        p, q = tm1_coords[cell_id]
        grating[idx] = amplitude * np.sin(kp * (p - p0) + kq * (q - q0) + phase) + offset

    return grating


def remove_reciprocal_connections(source_indices, target_indices, weights, neuron_types):
    """
    Filter edges to only keep those with Tm1 as the source type.
    """
    sources = to_numpy(source_indices)
    keep_mask = neuron_types[sources] == 'Tm1'
    
    filtered_sources = source_indices[keep_mask]
    filtered_targets = target_indices[keep_mask]
    filtered_weights = weights[keep_mask]
    
    n_removed = len(sources) - len(filtered_sources)
    print(f"Filtered to keep only Tm1 source connections. "
          f"Removed {n_removed} connections. "
          f"Remaining: {len(filtered_sources)} connections")
    
    return filtered_sources, filtered_targets, filtered_weights





