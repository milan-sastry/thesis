import numpy as np
import load_weights as lw
from utils import fit_von_mises, to_numpy, compute_response_metric
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
    v = to_numpy(v_final).reshape(-1)
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
    components = compute_response_metric(
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

def compute_scores_by_angle(results, use_flash=False, flash_windows=None, baseline_window=None):
    """
    Compute per-angle per-neuron scores from run-group data.

    If use_flash=False (default), scores are the steady-state final voltage
    (relu-clamped) read from run["v_final"].  If use_flash=True, scores are
    computed as flash-peak minus baseline using run["v_history"] and run["t"].

    If multiple runs share the same angle, scores are averaged across runs so
    downstream tuning-curve code receives one score vector per angle.
    """
    runs_data = results["runs"]
    scores_by_angle_lists = {}
    for run in runs_data:
        angle = float(run["angle"])
        if use_flash:
            scores = compute_flash_scores(run["v_history"], run["t"], flash_windows, baseline_window)
        else:
            scores = compute_steady_state_scores(run["v_final"])
        scores_by_angle_lists.setdefault(angle, []).append(scores)
    scores_by_angle = {}
    for angle, score_list in scores_by_angle_lists.items():
        stacked = np.stack([to_numpy(s).reshape(-1) for s in score_list], axis=0)
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
    scores = to_numpy(scores).reshape(-1)
    if neuron_types is None:
        neuron_types = lw.neuron_types
    neuron_types = to_numpy(neuron_types).reshape(-1)
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
):
    """
    Build tuning curves from run results.

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
    neuron_types = to_numpy(neuron_types).reshape(-1)

    if aggregation not in {"type_mean", "individual"}:
        raise ValueError(
            "`aggregation` must be one of {'type_mean', 'individual'}. "
            f"Got {aggregation!r}."
        )

    scores_by_angle = compute_scores_by_angle(results, use_flash=use_flash, flash_windows=flash_windows, baseline_window=baseline_window)
    angles = sorted(scores_by_angle.keys())
    curves = {}

    for angle in angles:
        scores_at_angle = to_numpy(scores_by_angle[angle]).reshape(-1)
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
            if target_type is not None:
                stats_by_type = {k: v for k, v in stats_by_type.items() if k == str(target_type)}
            for cell_type, stats in stats_by_type.items():
                if cell_type not in curves:
                    curves[cell_type] = {"angles": [], "mean": [], "sem": [], "n": []}
                curves[cell_type]["angles"].append(float(angle))
                curves[cell_type]["mean"].append(float(stats["mean"]))
                curves[cell_type]["sem"].append(float(stats["sem"]))
                curves[cell_type]["n"].append(int(stats["n"]))
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

            for idx in selected_indices:
                key = f"{neuron_types[idx]}#{int(idx)}"
                if key not in curves:
                    curves[key] = {"angles": [], "mean": [], "sem": [], "n": []}
                curves[key]["angles"].append(float(angle))
                curves[key]["mean"].append(float(scores_at_angle[idx]))
                curves[key]["sem"].append(0.0)
                curves[key]["n"].append(1)

    for cell_type, data in curves.items():
        data["angles"] = np.asarray(data["angles"], dtype=float)
        data["mean"] = np.asarray(data["mean"], dtype=float)
        data["sem"] = np.asarray(data["sem"], dtype=float)
        data["n"] = np.asarray(data["n"], dtype=int)
        data["fit"] = (
            fit_von_mises(data["angles"], data["mean"], period_deg=fit_period_deg)
            if fit
            else None
        )
        data["osi"] = orientation_selectivity_index(np.maximum(data["mean"], 0), data["angles"])

    return curves

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
        (line,) = ax.plot(x, y, marker="o", linestyle="None", label=cell_type)
        if show_sem:
            e = sem[order]
            ax.fill_between(x, y - e, y + e, alpha=0.15)
        if show_fit and data.get("fit") is not None:
            fit_data = data["fit"]
            x_fit = np.asarray(fit_data.get("x_fit", x), dtype=float)
            y_fit_dense = np.asarray(fit_data.get("y_fit_dense", fit_data["y_fit"]), dtype=float)
            ax.plot(
                x_fit,
                y_fit_dense,
                linewidth=2.0,
                alpha=0.9,
                color=line.get_color(),
            )

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

            y = np.asarray(data["mean"], dtype=float)[order]
            fit_data = data.get("fit")
            if fit_data is not None:
                x_fit = np.asarray(fit_data.get("x_fit", x), dtype=float)
                y_fit_dense = np.asarray(fit_data.get("y_fit_dense", fit_data["y_fit"]), dtype=float)
                ax.plot(x_fit, y_fit_dense, linewidth=2.0, alpha=0.95, color=color)
                if show_points:
                    ax.plot(x, y, marker="o", linestyle="None", alpha=0.7, color=color, markersize=4)
            else:
                ax.plot(x, y, linewidth=1.5, marker="o", alpha=0.9, color=color)

        ax.set_title(cell_type)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Score")
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        if ylim is None:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(cell_types):]:
        ax.axis("off")

    active_axes = axes_flat[:len(cell_types)].tolist()
    cbar = fig.colorbar(sm, ax=active_axes, fraction=0.03, pad=0.02)
    tick_values = np.unique(params)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:g}" for v in tick_values])
    cbar.set_label("Parameter value")
    fig.suptitle("Tuning Curves by Type Across Parameter Sweep", y=1.02)
    plt.savefig(filename)
    return fig, axes
