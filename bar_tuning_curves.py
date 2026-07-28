"""
Bar orientation tuning curves.

Presents static Gaussian bars at multiple orientations, each preceded by a
mean-luminance-matched gray baseline.  Uses the greedy set-cover algorithm
from analysis.py to pick bar centers that efficiently sample RF centers.

The main script sweeps bar length and width to show how bar geometry affects
orientation selectivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import load_weights as lw
from stimulus import StimulusGenerator
from network import DrosophilaOpticLobeCircuit
from dataset import filter_model_kwargs
from utils import to_numpy, compute_response_metric, fit_von_mises
import analysis as an
import tuning_curves as tc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stim_gen():
    return StimulusGenerator(lw.tm1_coords, lw.neuron_types, lw.row_ids)


def _make_model(model_settings, dt):
    kwargs = filter_model_kwargs(model_settings or {})
    kwargs.setdefault("dt", dt)
    return DrosophilaOpticLobeCircuit(
        lw.neuron_types,
        lw.source_indices,
        lw.target_indices,
        lw.weights,
        **kwargs,
    )


def _score_run(v_hist, t, metric, bar_start_t, bar_end_t):
    """
    Score all neurons in one run.

    Parameters
    ----------
    metric : {"peak", "steady", "mean"}
        peak   — transient peak of (bar window) minus baseline mean
        steady — late-window mean (last 20 % of bar) minus baseline mean
        mean   — full bar-window mean minus baseline mean
    """
    if metric == "peak":
        return to_numpy(compute_response_metric(
            values=v_hist, t=t,
            metric="window_delta", use_relu=True,
            windows=(bar_start_t, bar_end_t),
            baseline_windows=(0.0, bar_start_t),
            window_reduction="peak",
            baseline_reduction="mean",
        )).reshape(-1)

    elif metric == "mean":
        return to_numpy(compute_response_metric(
            values=v_hist, t=t,
            metric="window_delta", use_relu=True,
            windows=(bar_start_t, bar_end_t),
            baseline_windows=(0.0, bar_start_t),
            window_reduction="mean",
            baseline_reduction="mean",
        )).reshape(-1)

    elif metric == "steady":
        steady_start = bar_start_t + 0.8 * (bar_end_t - bar_start_t)
        return to_numpy(compute_response_metric(
            values=v_hist, t=t,
            metric="window_delta", use_relu=True,
            windows=(steady_start, bar_end_t),
            baseline_windows=(0.0, bar_start_t),
            window_reduction="mean",
            baseline_reduction="mean",
        )).reshape(-1)

    else:
        raise ValueError(f"metric must be 'peak', 'steady', or 'mean'; got {metric!r}")


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_center_angle_sweep(
    center,
    angles,
    width,
    length,
    intensity=1.0,
    sigma=0.5,
    on=True,
    baseline_steps=50,
    bar_steps=100,
    dt=0.1,
    model=None,
    model_settings=None,
):
    """
    Run one network simulation per angle for a single bar center.

    The gray baseline intensity equals the spatial mean of the bar stimulus so
    that the overall luminance is constant across the transition.

    Parameters
    ----------
    center : (int, int)
        (p, q) hex-grid coordinate for the bar center.
    angles : array-like of float
        Orientations in degrees to sweep.
    width, length : float
        Bar dimensions in hex-grid units.
    intensity : float
        Peak bar intensity.
    sigma : float
        Gaussian edge softness.
    on : bool
        True for bright bar on dark background.
    baseline_steps : int
        Number of timesteps for the gray baseline phase.
    bar_steps : int
        Number of timesteps to hold the bar stimulus.
    dt : float
        Model integration timestep (seconds per step).
    model : DrosophilaOpticLobeCircuit, optional
        Pre-built model to reuse.  If None, one is created from model_settings.
    model_settings : dict, optional
        Passed to filter_model_kwargs when model is None.

    Returns
    -------
    list of dicts
        Each dict: {"angle", "v_final", "v_history", "t"}.
    """
    stim_gen = _make_stim_gen()
    if model is None:
        model = _make_model(model_settings, dt)

    p_center, q_center = int(center[0]), int(center[1])
    runs = []

    for angle in angles:
        _, bar = stim_gen.create_gaussian_bar(
            width=width,
            length=length,
            p_center=p_center,
            q_center=q_center,
            on=on,
            angle=float(angle),
            offset=0.0,
            intensity=intensity,
            sigma=sigma,
        )
        gray_level = float(np.mean(bar))
        gray = stim_gen.create_mean_gray(intensity=gray_level)
        sequence = stim_gen.sequence_from_blocks(
            [(gray, baseline_steps), (bar, bar_steps)]
        )
        stimulus_t = stim_gen.to_torch(sequence)

        v_final, history = model(stimulus_t, return_history=True)
        runs.append({
            "angle": float(angle),
            "v_final": to_numpy(v_final.squeeze(0), dtype=np.float32),
            "v_history": to_numpy(history["v"].squeeze(0), dtype=np.float32),
            "t": to_numpy(history["t"], dtype=np.float32),
        })

    return runs


def build_bar_tuning_curves(
    selected_centers,
    per_center_coverage,
    angles,
    width,
    length,
    metric="peak",
    baseline_steps=50,
    bar_steps=100,
    dt=0.1,
    model_settings=None,
    intensity=1.0,
    sigma=0.5,
    on=True,
    types=None,
    fit=True,
    verbose=True,
):
    """
    Build per-type orientation tuning curves from a bar stimulus sweep.

    For each center in `selected_centers`, simulates bar stimuli at all
    `angles`, then assigns each run's neuron scores to the cell types credited
    to that center via `per_center_coverage`.

    Parameters
    ----------
    selected_centers : list of (int, int)
        Output of find_optimal_trial_centers.
    per_center_coverage : dict
        Output of find_optimal_trial_centers.
    angles : array-like of float
        Angles (degrees) to sweep.
    width, length : float
        Bar dimensions.
    metric : {"peak", "steady", "mean"}
        Response metric for scoring each angle.
    baseline_steps, bar_steps : int
        Duration of each stimulus phase in timesteps.
    dt : float
        Model timestep.
    model_settings : dict, optional
        Keyword args forwarded to DrosophilaOpticLobeCircuit.
    intensity : float
        Peak bar intensity.
    sigma : float
        Gaussian bar edge sigma.
    on : bool
        Bright-on-dark (True) or dark-on-bright (False).
    types : list of str, optional
        Cell types to collect curves for.  Defaults to the standard six.
    fit : bool
        If True, fit von Mises curves to each type's tuning data.
    verbose : bool
        Print progress.

    Returns
    -------
    curves : dict
        {cell_type: {"angles", "mean", "sem", "n", "fit", "osi",
                     "classical_osi"}}
    """
    if types is None:
        types = ['Dm3v', 'Dm3p', 'Dm3q', 'TmY4', 'TmY9q', 'TmY9q\u22a5']
    types_set = set(types)
    angles = list(angles)

    model = _make_model(model_settings, dt)
    bar_start_t = baseline_steps * dt
    bar_end_t = (baseline_steps + bar_steps) * dt

    # scores_pool[cell_type][angle] accumulates per-neuron scores
    scores_pool = defaultdict(lambda: defaultdict(list))

    for i, center in enumerate(selected_centers):
        if verbose:
            print(f"  center {i+1}/{len(selected_centers)}: {center}")

        runs = run_center_angle_sweep(
            center=center,
            angles=angles,
            width=width,
            length=length,
            intensity=intensity,
            sigma=sigma,
            on=on,
            baseline_steps=baseline_steps,
            bar_steps=bar_steps,
            dt=dt,
            model=model,
        )

        coverage = per_center_coverage.get(center, {})
        if not coverage:
            continue

        for run in runs:
            angle = run["angle"]
            all_scores = _score_run(
                run["v_history"], run["t"], metric, bar_start_t, bar_end_t
            )
            for cell_type, neuron_indices in coverage.items():
                if cell_type not in types_set:
                    continue
                for idx in neuron_indices:
                    scores_pool[cell_type][angle].append(float(all_scores[idx]))

    # Aggregate
    angles_sorted = sorted(angles)
    curves = {}
    for cell_type in types:
        if cell_type not in scores_pool:
            continue
        type_scores = scores_pool[cell_type]
        means, sems, ns, valid_angles = [], [], [], []
        for angle in angles_sorted:
            if angle not in type_scores or len(type_scores[angle]) == 0:
                continue
            vals = np.array(type_scores[angle], dtype=float)
            n = len(vals)
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
            ns.append(n)
            valid_angles.append(float(angle))

        if not means:
            continue

        angle_arr = np.array(valid_angles)
        mean_arr = np.array(means)
        sem_arr = np.array(sems)
        n_arr = np.array(ns, dtype=int)

        curve_fit = None
        if fit:
            try:
                curve_fit = fit_von_mises(angle_arr, mean_arr, period_deg=180.0)
            except Exception:
                pass

        curves[cell_type] = {
            "angles": angle_arr,
            "mean": mean_arr,
            "sem": sem_arr,
            "n": n_arr,
            "fit": curve_fit,
            "osi": tc.orientation_selectivity_index(
                np.maximum(mean_arr, 0), angle_arr
            ),
            "classical_osi": tc.classical_orientation_selectivity_index(
                np.maximum(mean_arr, 0), angle_arr
            ),
        }

    return curves


# ---------------------------------------------------------------------------
# Geometry sweep
# ---------------------------------------------------------------------------

def sweep_bar_geometry(
    selected_centers,
    per_center_coverage,
    angles,
    param_values,
    fixed_width=None,
    fixed_length=None,
    sweep_axis="length",
    metric="peak",
    baseline_steps=50,
    bar_steps=100,
    dt=0.1,
    model_settings=None,
    intensity=1.0,
    sigma=0.5,
    on=True,
    types=None,
    verbose=True,
):
    """
    Sweep bar length or width and collect a tuning curve set for each value.

    Parameters
    ----------
    param_values : array-like of float
        Values for the swept parameter (length or width).
    fixed_width : float
        Bar width when sweep_axis='length'.
    fixed_length : float
        Bar length when sweep_axis='width'.
    sweep_axis : {"length", "width"}
        Which bar dimension to vary.

    Returns
    -------
    curves_arr : list of dicts
        One tuning-curves dict per entry in param_values.
    params_arr : list of float
        param_values as a list (mirrors curves_arr).
    """
    if sweep_axis not in {"length", "width"}:
        raise ValueError("sweep_axis must be 'length' or 'width'")
    if sweep_axis == "length" and fixed_width is None:
        raise ValueError("fixed_width required when sweep_axis='length'")
    if sweep_axis == "width" and fixed_length is None:
        raise ValueError("fixed_length required when sweep_axis='width'")

    curves_arr = []
    params_arr = list(param_values)

    for param in params_arr:
        if sweep_axis == "length":
            width, length = fixed_width, param
        else:
            width, length = param, fixed_length

        label = f"{sweep_axis}={param:g}"
        if verbose:
            print(f"\n--- Bar {label} (width={width:g}, length={length:g}) ---")

        curves = build_bar_tuning_curves(
            selected_centers=selected_centers,
            per_center_coverage=per_center_coverage,
            angles=angles,
            width=width,
            length=length,
            metric=metric,
            baseline_steps=baseline_steps,
            bar_steps=bar_steps,
            dt=dt,
            model_settings=model_settings,
            intensity=intensity,
            sigma=sigma,
            on=on,
            types=types,
            verbose=verbose,
        )
        curves_arr.append(curves)

    return curves_arr, params_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Parameters ──────────────────────────────────────────────────────────
    ANGLES = np.arange(0.0, 180.0, 15.0)   # 0–165°, 12 orientations
    INTENSITY = 1.0
    SIGMA = 0.5
    ON = True
    BASELINE_STEPS = 1                      # 5 s at dt=0.1
    BAR_STEPS = 100                          # 10 s at dt=0.1
    DT = 0.1
    METRIC = "peak"                          # "peak" | "steady" | "mean"
    TARGET_N = 4                            # neurons per type for set-cover
    TYPES = ['Dm3v', 'Dm3p', 'Dm3q', 'TmY4', 'TmY9q', 'TmY9q\u22a5']

    MODEL_SETTINGS = {
        "vrest_init": -0.1,
    }                      # customise as needed

    # ── Find trial centers ───────────────────────────────────────────────────
    print("Computing trial centers via greedy set-cover …")
    selected_centers, per_center_coverage, type_coverage = an.find_optimal_trial_centers(
        target_n=TARGET_N, types=TYPES, verbose=True
    )

    # ── Length sweep  (fixed width = 2) ─────────────────────────────────────
    FIXED_WIDTH = 2.0 * 2/np.sqrt(3)  #
    LENGTH_VALUES = [2.0, 4.0, 8.0, 12.0, 16.0]

    print("\n=== Length sweep ===")
    length_curves_arr, length_params = sweep_bar_geometry(
        selected_centers=selected_centers,
        per_center_coverage=per_center_coverage,
        angles=ANGLES,
        param_values=LENGTH_VALUES * 2/np.sqrt(3), 
        fixed_width=FIXED_WIDTH,
        sweep_axis="length",
        metric=METRIC,
        baseline_steps=BASELINE_STEPS,
        bar_steps=BAR_STEPS,
        dt=DT,
        model_settings=MODEL_SETTINGS,
        intensity=INTENSITY,
        sigma=SIGMA,
        on=ON,
        types=TYPES,
        verbose=True,
    )

    fig_len, _ = tc.plot_curves_by_param(
        length_curves_arr,
        length_params,
        types=TYPES,
        cmap="plasma",
        filename="bar_tuning_length_sweep.png",
    )
    fig_len.suptitle(
        f"Bar length sweep  (width={FIXED_WIDTH}, metric={METRIC})", y=1.02
    )
    # update colorbar label
    for ax in fig_len.axes:
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == "Parameter value":
            ax.set_ylabel("Length")
    plt.savefig("bar_tuning_length_sweep.png", bbox_inches="tight", dpi=150)
    print("Saved bar_tuning_length_sweep.png")
    plt.show()

    # ── Width sweep  (fixed length = 8) ─────────────────────────────────────
    FIXED_LENGTH = 8.0 * 2/np.sqrt(3)  # 8 hex units along the bar's long axis
    WIDTH_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0]

    print("\n=== Width sweep ===")
    width_curves_arr, width_params = sweep_bar_geometry(
        selected_centers=selected_centers,
        per_center_coverage=per_center_coverage,
        angles=ANGLES,
        param_values=WIDTH_VALUES * 2/np.sqrt(3),
        fixed_length=FIXED_LENGTH,
        sweep_axis="width",
        metric=METRIC,
        baseline_steps=BASELINE_STEPS,
        bar_steps=BAR_STEPS,
        dt=DT,
        model_settings=MODEL_SETTINGS,
        intensity=INTENSITY,
        sigma=SIGMA,
        on=ON,
        types=TYPES,
        verbose=True,
    )

    fig_wid, _ = tc.plot_curves_by_param(
        width_curves_arr,
        width_params,
        types=TYPES,
        cmap="viridis",
        filename="bar_tuning_width_sweep.png",
    )
    fig_wid.suptitle(
        f"Bar width sweep  (length={FIXED_LENGTH}, metric={METRIC})", y=1.02
    )
    plt.savefig("bar_tuning_width_sweep.png", bbox_inches="tight", dpi=150)
    print("Saved bar_tuning_width_sweep.png")
    plt.show()
