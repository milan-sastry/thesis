import numpy as np
import load_weights as lw
import utils as utils
import matplotlib.pyplot as plt

def compute_steady_state_scores(v_final, use_relu=True):
    """
    Return per-neuron steady-state scores from the final voltage.

    Parameters
    ----------
    v_final : array-like, shape (N,) or (1, N)
        Final voltage from the model.
    use_relu : bool, default True
        If True, clamp negative values to zero.

    Returns
    -------
    scores : np.ndarray, shape (N,)
    """
    v = utils.to_numpy(v_final).reshape(-1)
    if use_relu:
        v = np.maximum(v, 0)
    return v


def compute_flash_scores(
    v_history,
    t,
    flash_windows,
    baseline_window,
    use_relu=True,
    return_components=False,
    use_flash_peak=False,
):
    """
    Compute per-neuron score = peak activity during flash - mean gray baseline.

    Parameters
    ----------
    v_history : array-like
        Activity history. Supported shapes:
        - (T, N)
        - (1, T, N)
        - (N, T) (auto-transposed to time-major)
    t : array-like, shape (T,)
        Time axis matching v_history.
    flash_windows : tuple or sequence of tuples
        Flash interval(s) as (start, end) in the same units as t.
        Window is interpreted as [start, end).
    use_relu : bool, default False
        If True, apply relu to v_history before computing peaks/means.
    return_components : bool, default False
        If True, return (scores, flash_peak, baseline_mean).

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Per-neuron scores.
    flash_peak : np.ndarray, shape (N,), optional
        Returned when return_components=True.
    baseline_mean : np.ndarray, shape (N,), optional
        Returned when return_components=True.
    """
    components = utils.compute_response_metric(
        values=v_history,
        t=t,
        metric="window_delta",
        use_relu=use_relu,
        windows=flash_windows,
        baseline_windows=baseline_window,
        window_reduction="peak" if use_flash_peak else "mean",
        baseline_reduction="mean",
        return_components=True,
    )
    scores = components["scores"]
    flash_peak = components["window_values"]
    baseline_mean = components["baseline_values"]

    if return_components:
        return scores, flash_peak, baseline_mean
    return scores

def _compute_fourier_components(v, t_np, temporal_freq, grating_onset_t, baseline_window, analysis_window):
    """
    Parameters
    ----------
    v : np.ndarray, shape (T,) or (T, N)
        Voltage, already converted to float numpy array.
    t_np : np.ndarray, shape (T,)
        Time axis.
    baseline_window, analysis_window : tuple of (float, float)
        Already resolved (not None).

    Returns
    -------
    dict with keys: baseline_mean, f0, C, t_ref
        baseline_mean : scalar or (N,) — mean during baseline window (0 if window empty)
        f0            : scalar or (N,) — mean of baseline-subtracted analysis window
        C             : scalar or (N,) complex — F1 coefficient: 2 * Y[idx] / nFFT
        t_ref         : float — first timepoint of analysis window (for phase reference)
    """
    baseline_mask = (
        (t_np >= grating_onset_t + baseline_window[0])
        & (t_np < grating_onset_t + baseline_window[1])
    )
    if np.any(baseline_mask):
        baseline_mean = np.mean(v[baseline_mask], axis=0)
        v = v - baseline_mean
    else:
        baseline_mean = np.zeros(v.shape[1:]) if v.ndim > 1 else 0.0

    analysis_mask = (
        (t_np >= grating_onset_t + analysis_window[0])
        & (t_np <= grating_onset_t + analysis_window[1])
    )
    if not np.any(analysis_mask):
        raise ValueError(
            f"analysis_window {analysis_window} relative to grating_onset_t "
            f"{grating_onset_t} selects no timepoints in t."
        )
    v_analysis = v[analysis_mask]
    t_ref = float(t_np[analysis_mask][0])

    f0 = np.mean(v_analysis, axis=0)
    v_analysis = v_analysis - f0
    nFFT = v_analysis.shape[0]
    if nFFT < 2:
        raise ValueError(
            "Not enough timepoints in analysis_window for FFT. "
            "Increase stimulus duration or widen analysis_window."
        )
    sampling_rate = 1.0 / float(t_np[1] - t_np[0])
    Y = np.fft.fft(v_analysis, axis=0)
    f = np.arange(nFFT) * (sampling_rate / nFFT)
    idx = int(np.argmin(np.abs(f - temporal_freq)))
    C = (2.0 / nFFT) * Y[idx]  # complex F1 amplitude

    return dict(baseline_mean=baseline_mean, f0=f0, C=C, t_ref=t_ref)

def compute_fourier_scores(
    v_history,
    t,
    temporal_freq,
    grating_onset_t,
    response_component="f1",
    use_relu=False,
    baseline_window=None,
    analysis_window=None,
):
    """
    Parameters
    ----------
    v_history : array-like, shape (T, N) or (1, T, N)
        Voltage history from the model.
    t : array-like, shape (T,)
        Time axis in arbitrary units where dt = t[1] - t[0] (typically 0.1).
        Covers baseline + grating periods concatenated.
    temporal_freq : float
        Temporal frequency of the drifting grating in cycles per time-unit
        (i.e. omega / (2*pi) when omega is in rad/time-unit).
    grating_onset_t : float
        Absolute t-value where the grating stimulus begins (e.g.
        baseline_steps * dt).  All window offsets are relative to this value.
    response_component : {'f1', 'f0', 'f1_over_f0'}
        Which Fourier component to return as the score.
    use_relu : bool
        If True, apply ReLU to the voltage before Fourier analysis.
    baseline_window : tuple of (float, float), optional
        Time window for baseline subtraction, expressed as offsets relative to
        grating onset.  Negative values refer to before onset.  Defaults to
        (-1, 0).
    analysis_window : tuple of (float, float), optional
        Analysis window for F0/F1 computation, expressed as offsets relative to
        grating onset.  Defaults to (1/temporal_freq, t[-1] - grating_onset_t),
        i.e. skip the first cycle as transient and use the remainder.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Per-neuron Fourier scores.
    """
    allowed = {"f1", "f0", "f1_over_f0"}
    if response_component not in allowed:
        raise ValueError(
            f"response_component must be one of {sorted(allowed)}. "
            f"Got {response_component!r}."
        )

    v, t_np = _normalize_time_major(v_history, t=t)
    if use_relu:
        v = np.maximum(v, 0.0)

    if baseline_window is None:
        baseline_window = (-1, 0.0)
    if analysis_window is None:
        analysis_window = (1.0 / temporal_freq, float(t_np[-1]) - grating_onset_t)

    components = _compute_fourier_components(v, t_np, temporal_freq, grating_onset_t, baseline_window, analysis_window)
    f0, C = components["f0"], components["C"]
    f1 = np.abs(C)

    if response_component == "f1":
        return f1
    elif response_component == "f0":
        return f0
    else:  # f1_over_f0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(f0) > 0, f1 / np.abs(f0), 0.0)


def reconstruct_f1(
    signal,
    t,
    temporal_freq,
    grating_onset_t,
    baseline_window=None,
    analysis_window=None,
):
    """
    Parameters
    ----------
    signal : array-like, shape (T,)
        Single-neuron response time series.
    t : array-like, shape (T,)
        Time axis in arbitrary units (dt = t[1] - t[0], typically 0.1).
    temporal_freq : float
        Temporal frequency of the drifting grating in cycles per time-unit.
    grating_onset_t : float
        Absolute t-value where the grating stimulus begins (e.g.
        baseline_steps * dt).  All window offsets are relative to this value.
    baseline_window : tuple of (float, float), optional
        Offsets relative to grating onset for baseline subtraction.
        Defaults to (-1, 0).
    analysis_window : tuple of (float, float), optional
        Offsets relative to grating onset for F0/F1 estimation.
        Defaults to (1/temporal_freq, t[-1] - grating_onset_t).

    Returns
    -------
    recon : np.ndarray, shape (T,)
        Reconstructed F0 + F1 sinusoid in original signal units, defined
        over the full time axis so it can be overlaid on the raw trace.
    """
    t_np = np.asarray(t, dtype=float)
    v = np.asarray(signal, dtype=float).ravel()

    if baseline_window is None:
        baseline_window = (-1, 0.0)
    if analysis_window is None:
        analysis_window = (1.0 / temporal_freq, float(t_np[-1]) - grating_onset_t)

    components = _compute_fourier_components(v, t_np, temporal_freq, grating_onset_t, baseline_window, analysis_window)
    baseline_mean = components["baseline_mean"]
    f0 = components["f0"]
    C = components["C"]
    t_ref = components["t_ref"]

    recon = (
        baseline_mean
        + f0
        + np.real(C * np.exp(2j * np.pi * temporal_freq * (t_np - t_ref)))
    )
    return recon


def _normalize_time_major(values, t=None):
    """Re-export for local use (imported from utils via compute_response_metric)."""
    from utils import _normalize_time_major as _ntm
    return _ntm(values, t=t)

def compute_scores_by_angle(
    results,
    use_flash=False,
    flash_windows=None,
    baseline_window=None,
    use_fourier=False,
    temporal_freq=None,
    grating_onset_t=None,
    response_component="f1",
    use_relu=True,
    analysis_window=None,
):
    """
    Compute per-angle per-neuron scores from run-group data.

    Exactly one scoring mode must be active:
      - Default (all False): steady-state final voltage (relu-clamped).
      - use_flash=True: flash-peak minus baseline from v_history / t.
      - use_fourier=True: F0/F1 Fourier component at the grating temporal
        frequency (see compute_f1_scores).

    If multiple runs share the same angle, scores are averaged across runs.

    Parameters
    ----------
    grating_onset_t : float, optional
        Absolute t-value where the grating begins (e.g. baseline_steps * dt).
        Required when use_fourier=True; passed through to compute_f1_scores.
    """
    if use_fourier and use_flash:
        raise ValueError("use_fourier and use_flash cannot both be True.")
    if use_fourier and temporal_freq is None:
        raise ValueError("`temporal_freq` must be provided when use_fourier=True.")
    if use_fourier and grating_onset_t is None:
        raise ValueError("`grating_onset_t` must be provided when use_fourier=True.")

    runs_data = results["runs"]
    scores_by_angle_lists = {}
    for run in runs_data:
        angle = float(run["angle"])
        if use_fourier:
            scores = compute_fourier_scores(
                run["v_history"],
                run["t"],
                temporal_freq=temporal_freq,
                grating_onset_t=grating_onset_t,
                response_component=response_component,
                use_relu=use_relu,
                analysis_window=analysis_window,
            )
        elif use_flash:
            scores = compute_flash_scores(run["v_history"], run["t"], flash_windows, baseline_window)
        else:
            scores = compute_steady_state_scores(run["v_final"])
        scores_by_angle_lists.setdefault(angle, []).append(scores)
    scores_by_angle = {}
    for angle, score_list in scores_by_angle_lists.items():
        stacked = np.stack([utils.to_numpy(s).reshape(-1) for s in score_list], axis=0)
        scores_by_angle[angle] = np.mean(stacked, axis=0)
    return scores_by_angle


def average_scores_by_type(
    scores,
    neuron_types=None,
    exclude_types=("Tm1",),
    active_only=False,
    activity_threshold=0.01,
):
    """
    Average per-neuron scores by cell type.

    Parameters
    ----------
    scores : array-like, shape (N,)
        Per-neuron scores (e.g., from compute_flash_scores).
    neuron_types : array-like, optional
        Type label for each neuron. If None, defaults to lw.neuron_types.
    exclude_types : sequence of str, optional
        Cell types to exclude from returned averages. Defaults to ("Tm1",).
    active_only : bool, optional
        If True, compute stats using only neurons with score > activity_threshold.
    activity_threshold : float, optional
        Threshold for active_only filtering.

    Returns
    -------
    dict
        {cell_type: mean_score}
    """
    scores = utils.to_numpy(scores).reshape(-1)
    if neuron_types is None:
        neuron_types = lw.neuron_types
    neuron_types = utils.to_numpy(neuron_types).reshape(-1)
    excluded = {str(cell_type) for cell_type in (exclude_types or ())}

    if scores.shape[0] != neuron_types.shape[0]:
        raise ValueError(
            "scores and neuron_types must have same length. "
            f"Got {scores.shape[0]} and {neuron_types.shape[0]}."
        )

    avg_by_type = {}
    for cell_type in np.unique(neuron_types):
        if str(cell_type) in excluded:
            continue
        vals = scores[neuron_types == cell_type]
        if active_only:
            vals = vals[vals > activity_threshold]
        n = vals.size
        if n == 0:
            mean = np.nan
            sem = np.nan
        else:
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        avg_by_type[str(cell_type)] = {"mean": mean, "sem": sem, "n": int(n)}
    return avg_by_type

def orientation_selectivity_index(responses, angles):
    """
    Compute orientation selectivity index (OSI) for a set of responses and angles.

    Parameters
    ----------
    responses : array-like, shape (N,)
        Response values corresponding to each angle.
    angles : array-like, shape (N,)
        Angles in degrees corresponding to each response.

    Returns
    -------
    osi : float
        Orientation selectivity index, between 0 and 1.
    """
    angles_rad = np.deg2rad(angles)
    vector_sum = np.sum(responses * np.exp(2j * angles_rad))
    osi = np.abs(vector_sum) / np.sum(responses) if np.sum(responses) > 0 else 0.0
    return float(osi)

def tuning_curve(
    results,
    fit=True,
    exclude_types=("Tm1",),
    fit_period_deg=180.0,
    active_only=False,
    activity_threshold=0.01,
    aggregation="type_mean",
    target_type=None,
    neuron_ids=None,
    neuron_types=None,
    use_flash=False,
    flash_windows=None,
    baseline_window=None,
    use_fourier=False,
    temporal_freq=None,
    grating_onset_t=None,
    response_component="both",
    use_relu=True,
    fwhm = True,
    analysis_window=None,
):
    """
    Build tuning curves from run results.

    Parameters
    ----------
    use_fourier : bool
        If True, score each angle using Fourier analysis of the response time
        series (see compute_f1_scores). Intended for moving-grating experiments.
    temporal_freq : float, optional
        Temporal frequency of the drifting grating in cycles per time-unit.
        Required when use_fourier=True.  For a grating created with angular
        frequency omega (rad/time-unit), pass temporal_freq = omega / (2*pi).
    grating_onset_t : float, optional
        Absolute t-value where the grating stimulus begins (e.g.
        baseline_steps * dt).  Required when use_fourier=True; passed through
        to compute_f1_scores.
    response_component : {'f1', 'f0', 'f1_over_f0', 'both'}
        Which Fourier component to use as the score when use_fourier=True.
        Defaults to 'both', which computes both f0 and f1 and stores them as
        separate fields (mean_f0/mean_f1, fit_f0/fit_f1, osi_f0/osi_f1).

    Returns
    -------
    dict
        {key: {"angles", "mean", "sem", "n", "fit"}} where:
        - aggregation='type_mean': key is cell type.
        - aggregation='individual': key is "<cell_type>#<neuron_index>".

    Notes
    -----
    - For aggregation='type_mean', behavior matches the original implementation.
    - For aggregation='individual', `sem` is 0 and `n` is 1 at each angle so
      output stays compatible with existing plotting helpers.
    """
    if neuron_types is None:
        neuron_types = lw.neuron_types
    neuron_types = utils.to_numpy(neuron_types).reshape(-1)

    if aggregation not in {"type_mean", "individual"}:
        raise ValueError(
            "`aggregation` must be one of {'type_mean', 'individual'}. "
            f"Got {aggregation!r}."
        )

    both_mode = use_fourier and response_component == "both"
    _component = response_component if not both_mode else "f0"

    scores_by_angle = compute_scores_by_angle(
        results,
        use_flash=use_flash,
        flash_windows=flash_windows,
        baseline_window=baseline_window,
        use_fourier=use_fourier,
        temporal_freq=temporal_freq,
        grating_onset_t=grating_onset_t,
        response_component=_component,
        use_relu=use_relu,
        analysis_window=analysis_window,
    )
    if both_mode:
        scores_by_angle_f1 = compute_scores_by_angle(
            results,
            use_flash=use_flash,
            flash_windows=flash_windows,
            baseline_window=baseline_window,
            use_fourier=True,
            temporal_freq=temporal_freq,
            grating_onset_t=grating_onset_t,
            response_component="f1",
            use_relu=use_relu,
            analysis_window=analysis_window,
        )

    angles = sorted(scores_by_angle.keys())
    curves = {}

    for angle in angles:
        scores_at_angle = utils.to_numpy(scores_by_angle[angle]).reshape(-1)
        if scores_at_angle.shape[0] != neuron_types.shape[0]:
            raise ValueError(
                "scores and neuron_types must have same length. "
                f"Got {scores_at_angle.shape[0]} and {neuron_types.shape[0]}."
            )

        if aggregation == "type_mean":
            stats_by_type = average_scores_by_type(
                scores_at_angle,
                neuron_types=neuron_types,
                exclude_types=exclude_types,
                active_only=active_only,
                activity_threshold=activity_threshold,
            )
            if both_mode:
                scores_at_angle_f1 = utils.to_numpy(scores_by_angle_f1[angle]).reshape(-1)
                stats_by_type_f1 = average_scores_by_type(
                    scores_at_angle_f1,
                    neuron_types=neuron_types,
                    exclude_types=exclude_types,
                    active_only=active_only,
                    activity_threshold=activity_threshold,
                )
            if target_type is not None:
                stats_by_type = {k: v for k, v in stats_by_type.items() if k == str(target_type)}
                if both_mode:
                    stats_by_type_f1 = {k: v for k, v in stats_by_type_f1.items() if k == str(target_type)}
            for cell_type, stats in stats_by_type.items():
                if cell_type not in curves:
                    curves[cell_type] = {"angles": [], "mean": [], "sem": [], "n": []}
                    if both_mode:
                        curves[cell_type]["mean_f1"] = []
                curves[cell_type]["angles"].append(float(angle))
                curves[cell_type]["mean"].append(float(stats["mean"]))
                curves[cell_type]["sem"].append(float(stats["sem"]))
                curves[cell_type]["n"].append(int(stats["n"]))
                if both_mode and cell_type in stats_by_type_f1:
                    curves[cell_type]["mean_f1"].append(float(stats_by_type_f1[cell_type]["mean"]))
        else:
            if neuron_ids is None:
                if target_type is None:
                    raise ValueError(
                        "For aggregation='individual', provide either `target_type` "
                        "or `neuron_ids`."
                    )
                selected_indices = np.where(neuron_types == target_type)[0]
            else:
                selected_indices = np.asarray(neuron_ids, dtype=int).reshape(-1)

            if selected_indices.size == 0:
                raise ValueError("No neurons selected for aggregation='individual'.")
            if np.any(selected_indices < 0) or np.any(selected_indices >= neuron_types.shape[0]):
                raise ValueError(
                    "`neuron_ids` contains out-of-bounds indices for `neuron_types`."
                )

            if both_mode:
                scores_at_angle_f1 = utils.to_numpy(scores_by_angle_f1[angle]).reshape(-1)

            for idx in selected_indices:
                key = f"{neuron_types[idx]}#{int(idx)}"
                if key not in curves:
                    curves[key] = {"angles": [], "mean": [], "sem": [], "n": []}
                    if both_mode:
                        curves[key]["mean_f1"] = []
                curves[key]["angles"].append(float(angle))
                curves[key]["mean"].append(float(scores_at_angle[idx]))
                curves[key]["sem"].append(0.0)
                curves[key]["n"].append(1)
                if both_mode:
                    curves[key]["mean_f1"].append(float(scores_at_angle_f1[idx]))

    for cell_type, data in curves.items():
        data["angles"] = np.asarray(data["angles"], dtype=float)
        data["mean"] = np.asarray(data["mean"], dtype=float)
        data["sem"] = np.asarray(data["sem"], dtype=float)
        data["n"] = np.asarray(data["n"], dtype=int)
        data["fit"] = (
            utils.fit_von_mises(data["angles"], data["mean"], period_deg=fit_period_deg)
            if fit
            else None
        )
        data["osi"] = orientation_selectivity_index(np.maximum(data["mean"], 0), data["angles"])
        data["fwhm"] = utils.fwhm(kappa=data["fit"]["kappa"], period_deg=fit_period_deg) if fwhm else np.nan

        if both_mode and "mean_f1" in data:
            data["mean_f1"] = np.asarray(data["mean_f1"], dtype=float)
            data["fit_f1"] = (
                utils.fit_von_mises(data["angles"], data["mean_f1"], period_deg=fit_period_deg)
                if fit
                else None
            )
            data["osi_f1"] = orientation_selectivity_index(np.maximum(data["mean_f1"], 0), data["angles"])
            data["fwhm_f1"] = utils.fwhm(kappa=data["fit_f1"]["kappa"], period_deg=fit_period_deg) if fwhm else np.nan
            # Aliases for clarity
            data["mean_f0"] = data["mean"]
            data["fit_f0"] = data["fit"]
            data["osi_f0"] = data["osi"]
            data["fwhm_f0"] = data["fwhm"]

    return curves

def rank_neurons_by_key(curves, cell_type_filter=None, top_n=None, sort_by="osi", ascending=False):
    """
    Rank neurons by a tuning curve metric.

    Parameters
    ----------
    curves : dict
        Output of tuning_curve() with aggregation='individual'.
        Keys are "<cell_type>#<neuron_index>".
    cell_type_filter : str or list of str, optional
        Only include neurons of these cell type(s). If None, includes all.
    top_n : int, optional
        Return only the top N neurons. If None, returns all.
    sort_by : str
        Key to sort by. Must be present in each curve's data dict.
        Common options: "osi", "fwhm", "range" (max-min of mean responses).
        In both-component mode also: "osi_f0", "osi_f1", "range_f0", "range_f1".
    ascending : bool
        If True, sort ascending (e.g. lowest FWHM first). Default False.

    Returns
    -------
    list of dicts, sorted by `sort_by`:
        [{"key": key, "neuron_index": int, "cell_type": str, "osi": float, "fwhm": float}, ...]
    """
    if isinstance(cell_type_filter, str):
        cell_type_filter = {cell_type_filter}
    elif cell_type_filter is not None:
        cell_type_filter = set(cell_type_filter)

    records = []
    for key, data in curves.items():
        cell_type, _, idx_str = key.partition("#")
        if cell_type_filter is not None and cell_type not in cell_type_filter:
            continue
        mean = np.asarray(data["mean"], dtype=float)
        record = {
            "key": key,
            "neuron_index": int(idx_str),
            "cell_type": cell_type,
            "osi": float(data["osi"]),
            "fwhm": float(data["fwhm"]),
            "range": float(np.max(mean) - np.min(mean)),
        }
        record["range_osi"] = record["range"] * record["osi"]
        for extra_key in ("osi_f0", "osi_f1", "fwhm_f0", "fwhm_f1"):
            if extra_key in data:
                record[extra_key] = float(data[extra_key])
        if "mean_f1" in data:
            mean_f1 = np.asarray(data["mean_f1"], dtype=float)
            record["range_f0"] = record["range"]
            record["range_f1"] = float(np.max(mean_f1) - np.min(mean_f1))
            record["range_osi_f0"] = record["range_f0"] * record["osi_f0"]
            record["range_osi_f1"] = record["range_f1"] * record["osi_f1"]
        records.append(record)

    records.sort(key=lambda r: r[sort_by], reverse=not ascending)
    if top_n is not None:
        records = records[:top_n]
    return records


def plot_tuning_curves(curves, types=None, show_sem=True, show_fit=True, ax=None, ylim=None):
    """
    Plot per-type tuning curves with optional SEM and von Mises fits.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    cell_types = list(curves.keys()) if types is None else [t for t in types if t in curves]
    for cell_type in cell_types:
        data = curves[cell_type]
        angles = np.asarray(data["angles"], dtype=float)
        mean = np.asarray(data["mean"], dtype=float)
        sem = np.asarray(data["sem"], dtype=float)
        order = np.argsort(angles)
        x = angles[order]
        y = mean[order]
        both_mode = "mean_f1" in data

        label_f0 = f"{cell_type} (F0)" if both_mode else cell_type
        (line,) = ax.plot(x, y, marker="o", linestyle="None", label=label_f0)
        color = line.get_color()
        if show_sem:
            e = sem[order]
            ax.fill_between(x, y - e, y + e, alpha=0.15, color=color)
        if show_fit and data.get("fit") is not None:
            fit_data = data["fit"]
            x_fit = np.asarray(fit_data.get("x_fit", x), dtype=float)
            y_fit_dense = np.asarray(fit_data.get("y_fit_dense", fit_data["y_fit"]), dtype=float)
            ax.plot(x_fit, y_fit_dense, linewidth=2.0, alpha=0.9, color=color)

        if both_mode:
            y_f1 = np.asarray(data["mean_f1"], dtype=float)[order]
            ax.plot(x, y_f1, marker="s", linestyle="None", color=color, label=f"{cell_type} (F1)")
            if show_fit and data.get("fit_f1") is not None:
                fit_f1 = data["fit_f1"]
                x_fit = np.asarray(fit_f1.get("x_fit", x), dtype=float)
                y_fit_dense = np.asarray(fit_f1.get("y_fit_dense", fit_f1["y_fit"]), dtype=float)
                ax.plot(x_fit, y_fit_dense, linewidth=2.0, linestyle="--", alpha=0.9, color=color)

    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Score")
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
    if ylim is None:
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_title("Tuning Curves by Type")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return ax

"""
curves_arr is a list of dicts, each dict is a set of tuning curves for a given parameter
params_arr is a list that contains the values of the parameter for each set of curves
"""
def plot_curves_by_param(curves_arr, params_arr, types=None, cmap="viridis", filename="curves_by_param.png", ylim=None, show_points=False):
    if len(curves_arr) == 0:
        raise ValueError("`curves_arr` must contain at least one set of curves.")
    if len(curves_arr) != len(params_arr):
        raise ValueError(
            f"`curves_arr` and `params_arr` must have same length. Got {len(curves_arr)} and {len(params_arr)}."
        )

    reference_curves = curves_arr[0]
    cell_types = list(reference_curves.keys()) if types is None else [t for t in types if t in reference_curves]
    if len(cell_types) == 0:
        raise ValueError("No matching cell types found to plot.")

    n = len(cell_types)
    nrows = 1 if n <= 2 else 2
    ncols = int(np.ceil(n / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    params = np.asarray(params_arr, dtype=float)
    cmap_obj = plt.get_cmap(cmap)
    pmin = float(np.min(params))
    pmax = float(np.max(params))
    # Avoid divide-by-zero in color normalization when all params are equal.
    if np.isclose(pmin, pmax):
        pmax = pmin + 1e-12
    norm = plt.Normalize(vmin=pmin, vmax=pmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    for ax, cell_type in zip(axes_flat, cell_types):
        for curves, param in zip(curves_arr, params):
            if cell_type not in curves:
                continue
            data = curves[cell_type]
            angles = np.asarray(data["angles"], dtype=float)
            order = np.argsort(angles)
            x = angles[order]
            color = cmap_obj(norm(float(param)))
            both_mode = "mean_f1" in data

            y = np.asarray(data["mean"], dtype=float)[order]
            fit_data = data.get("fit")
            if fit_data is not None:
                x_fit = np.asarray(fit_data.get("x_fit", x), dtype=float)
                y_fit_dense = np.asarray(fit_data.get("y_fit_dense", fit_data["y_fit"]), dtype=float)
                ax.plot(x_fit, y_fit_dense, linewidth=2.0, alpha=0.95, color=color, label="F0" if both_mode else None)
                if show_points:
                    ax.plot(x, y, marker="o", linestyle="None", alpha=0.7, color=color, markersize=4)
            else:
                ax.plot(x, y, linewidth=1.5, marker="o", alpha=0.9, color=color)

            if both_mode:
                y_f1 = np.asarray(data["mean_f1"], dtype=float)[order]
                fit_f1 = data.get("fit_f1")
                if fit_f1 is not None:
                    x_fit = np.asarray(fit_f1.get("x_fit", x), dtype=float)
                    y_fit_dense = np.asarray(fit_f1.get("y_fit_dense", fit_f1["y_fit"]), dtype=float)
                    ax.plot(x_fit, y_fit_dense, linewidth=2.0, linestyle="--", alpha=0.95, color=color, label="F1")
                    if show_points:
                        ax.plot(x, y_f1, marker="s", linestyle="None", alpha=0.7, color=color, markersize=4)
                else:
                    ax.plot(x, y_f1, linewidth=1.5, linestyle="--", marker="s", alpha=0.9, color=color)

        ax.set_title(cell_type)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        if ylim is None:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid(True, alpha=0.25)
        # Add F0/F1 legend using proxy artists when both components are shown
        _any_both = any("mean_f1" in curves.get(cell_type, {}) for curves in curves_arr)
        if _any_both:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], linewidth=2.0, linestyle="-", color="gray", label="F0"),
                Line2D([0], [0], linewidth=2.0, linestyle="--", color="gray", label="F1"),
            ]
            ax.legend(handles=handles, frameon=False, fontsize=8)

    for ax in axes_flat[len(cell_types):]:
        ax.axis("off")

    fig.supylabel("Score")
    fig.supxlabel("Angle (deg)")
    active_axes = axes_flat[:len(cell_types)].tolist()
    cbar = fig.colorbar(sm, ax=active_axes, fraction=0.03, pad=0.02)
    tick_values = np.unique(params)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:g}" for v in tick_values])
    cbar.set_label("Parameter value")
    fig.suptitle("Tuning Curves by Type Across Parameter Sweep", y=1.02)
    plt.savefig(filename)
    return fig, axes
