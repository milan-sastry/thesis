"""
Average bar-stimulus tuning curves for each cell type.

The key idea: a single bar trial centred at (p, q) stimulates every neuron
whose RF centre coincides with that coordinate.  By first running
`find_optimal_trial_centers` we discover the minimum number of distinct
(p, q) placements needed to accumulate `TARGET_N` cells per type, then run
only those trials instead of one per neuron.
"""

import numpy as np
import matplotlib.pyplot as plt
from visualize import visualize_stimulus_with_tm1_inputs
import analysis as an
import grating_visualization as gv
from bars import generate_bar_response
from utils import fit_von_mises

# ---------------------------------------------------------------------------
# Experiment defaults (override via run_average_tuning_curves kwargs)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_N       = 50
DEFAULT_TYPES          = ['Dm3v', 'Dm3p', 'Dm3q', 'TmY4', 'TmY9q', 'TmY9q\u22a5']
DEFAULT_ANGLES         = list(range(0, 180, 15))
DEFAULT_BAR_WIDTH      = 1.5
DEFAULT_BAR_LENGTH     = 10.0
DEFAULT_AMPLITUDE      = 0.9
DEFAULT_MEAN_DURATION  = 100
DEFAULT_BAR_DURATION   = 100
DEFAULT_MEAN_INTENSITY = 0.1
DEFAULT_SIGMA          = 0.5
DEFAULT_FLASH_WINDOWS  = (10.0, 20.0)
DEFAULT_BASELINE       = (0.0, 10.0)


# ---------------------------------------------------------------------------
# Core experiment function
# ---------------------------------------------------------------------------

def run_average_tuning_curves(
    target_n=DEFAULT_TARGET_N,
    types=None,
    angles=None,
    use_all=False,
    bar_width=DEFAULT_BAR_WIDTH,
    bar_length=DEFAULT_BAR_LENGTH,
    amplitude=DEFAULT_AMPLITUDE,
    mean_duration=DEFAULT_MEAN_DURATION,
    bar_duration=DEFAULT_BAR_DURATION,
    mean_intensity=DEFAULT_MEAN_INTENSITY,
    sigma=DEFAULT_SIGMA,
    flash_windows=DEFAULT_FLASH_WINDOWS,
    baseline_window=DEFAULT_BASELINE,
    model_settings=None,
    fit=True,
    fit_period_deg=180.0,
):
    """
    Run the minimum number of bar-stimulus trials needed to build average
    tuning curves for each cell type.

    Parameters
    ----------
    target_n : int
        Number of cells per type to average over (default 50).  Pass
        ``use_all=True`` to use every available cell.
    types : list[str] | None
        Cell types to analyse.  Defaults to the standard six types.
    angles : list[float] | None
        Bar orientations in degrees.  Defaults to 0–165° in 15° steps.
    use_all : bool
        If True, ignore ``target_n`` and cover all cells with valid RF centres.
    bar_width, bar_length : float
        Gaussian-bar dimensions.
    amplitude : float
        Peak bar intensity.
    mean_duration, bar_duration : int
        Number of simulation steps for the baseline and bar phases.
    mean_intensity : float
        Background grey level.
    sigma : float
        Gaussian blur width of the bar edge.
    flash_windows : tuple[float, float]
        (start, end) of the response window used to compute scores.
    baseline_window : tuple[float, float]
        (start, end) of the baseline window.
    model_settings : dict | None
        Extra kwargs forwarded to DrosophilaOpticLobeCircuit (e.g. weight
        scaling, vrest overrides).
    fit : bool
        If True, fit a von Mises curve to each type's average tuning curve.
    fit_period_deg : float
        Period (degrees) of the von Mises fit.

    Returns
    -------
    type_curves : dict[str, dict]
        ``{type: {"angles", "mean", "sem", "n", "fit"}}`` — same schema as
        ``grating_visualization.tuning_curve``, ready for
        ``gv.plot_tuning_curves``.
    neuron_scores : dict[int, dict[float, float]]
        ``{neuron_index: {angle: score}}`` — raw per-neuron scores.
    selected_centers : list[tuple[int, int]]
        The (p, q) centres used.
    per_center_coverage : dict[tuple, dict[str, list[int]]]
        Neurons credited to each centre.
    type_coverage : dict[str, list[int]]
        All covered neuron indices per type.
    """
    if types is None:
        types = list(DEFAULT_TYPES)
    if angles is None:
        angles = list(DEFAULT_ANGLES)
    angles_f = [float(a) for a in angles]

    # ------------------------------------------------------------------
    # Step 1: find the minimum set of bar centres
    # ------------------------------------------------------------------
    selected_centers, per_center_coverage, type_coverage = an.find_optimal_trial_centers(
        target_n=target_n,
        types=types,
        use_all=use_all,
        verbose=True,
    )

    # Build a quick lookup: neuron_index -> (p, q) centre it belongs to
    neuron_to_center = {}
    for center, tmap in per_center_coverage.items():
        for idxs in tmap.values():
            for idx in idxs:
                neuron_to_center[idx] = center

    # ------------------------------------------------------------------
    # Step 2: run one bar-trial sequence per (p, q) centre
    # ------------------------------------------------------------------
    # neuron_scores[neuron_idx][angle] = float score
    neuron_scores: dict = {}

    n_total = len(selected_centers) * len(angles_f)
    run_no = 0

    for p, q in selected_centers:
        covered_here = per_center_coverage[(p, q)]
        covered_indices = sorted(
            {idx for idxs in covered_here.values() for idx in idxs}
        )

        for angle in angles_f:
            run_no += 1
            print(
                f"[{run_no}/{n_total}] centre=({p},{q})  angle={angle:.0f}°",
                flush=True,
            )

            v_final, v_hist, t, bar = generate_bar_response(
                angle=angle,
                width=bar_width,
                length=bar_length,
                amplitude=amplitude,
                mean_duration=mean_duration,
                mean_intensity=mean_intensity,
                bar_duration=bar_duration,
                model_settings=model_settings,
                center=(p, q),
                sigma=sigma,
            )
            # for idx in covered_indices:
            #     visualize_stimulus_with_tm1_inputs(
            #         neuron_index=idx,
            #         stimulus=bar)
            #     plt.show()

            # compute_flash_scores returns shape (N,) — one score per neuron
            scores = gv.compute_flash_scores(
                v_hist, t, flash_windows, baseline_window
            )

            for idx in covered_indices:
                neuron_scores.setdefault(idx, {})[angle] = float(scores[idx])

    # ------------------------------------------------------------------
    # Step 3: average per-neuron scores by type
    # ------------------------------------------------------------------
    type_curves = {}

    for ntype in types:
        indices = type_coverage.get(ntype, [])
        if not indices:
            continue

        angle_means, angle_sems, angle_ns = [], [], []
        for angle in sorted(angles_f):
            vals = [
                neuron_scores[i][angle]
                for i in indices
                if angle in neuron_scores.get(i, {})
            ]
            if vals:
                arr = np.array(vals, dtype=float)
                n = len(arr)
                mean = float(np.mean(arr))
                sem = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            else:
                mean, sem, n = np.nan, np.nan, 0
            angle_means.append(mean)
            angle_sems.append(sem)
            angle_ns.append(n)

        angle_arr = np.array(sorted(angles_f))
        mean_arr = np.array(angle_means)

        type_curves[ntype] = {
            "angles": angle_arr,
            "mean": mean_arr,
            "sem": np.array(angle_sems),
            "n": np.array(angle_ns, dtype=int),
            "fit": (
                fit_von_mises(angle_arr, mean_arr, period_deg=fit_period_deg)
                if fit
                else None
            ),
        }

    return type_curves, neuron_scores, selected_centers, per_center_coverage, type_coverage


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute average bar tuning curves for each cell type."
    )
    parser.add_argument(
        "--target-n", type=int, default=DEFAULT_TARGET_N,
        help="Number of cells per type to average (default: %(default)s).",
    )
    parser.add_argument(
        "--use-all", action="store_true",
        help="Cover all available cells instead of a fixed target.",
    )
    parser.add_argument(
        "--output", default="average_tuning_curves.png",
        help="Output figure filename (default: %(default)s).",
    )
    parser.add_argument(
        "--no-fit", action="store_true",
        help="Skip von Mises fitting.",
    )
    args = parser.parse_args()

    type_curves, neuron_scores, selected_centers, per_center_coverage, type_coverage = (
        run_average_tuning_curves(
            target_n=args.target_n,
            use_all=args.use_all,
            fit=not args.no_fit,
        )
    )

    # ---- report ----
    print("\n=== Per-type average tuning curve summary ===")
    for ntype, data in type_curves.items():
        n_cells = int(data["n"][0]) if len(data["n"]) else 0
        peak_angle = float(data["angles"][np.nanargmax(data["mean"])]) if len(data["mean"]) else float("nan")
        print(f"  {ntype:<12}  n={n_cells}  peak≈{peak_angle:.0f}°")

    # ---- plot ----
    dm3_types = [t for t in type_curves if t.startswith("Dm3")]
    tmy_types  = [t for t in type_curves if not t.startswith("Dm3")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    gv.plot_tuning_curves(type_curves, types=dm3_types, ax=ax1)
    ax1.set_title(
        f"Dm3 Average Tuning Curves  "
        f"(n={'all' if args.use_all else args.target_n}/type, "
        f"{len(selected_centers)} trial centres)"
    )

    gv.plot_tuning_curves(type_curves, types=tmy_types, ax=ax2)
    ax2.set_title(
        f"TmY Average Tuning Curves  "
        f"(n={'all' if args.use_all else args.target_n}/type, "
        f"{len(selected_centers)} trial centres)"
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {args.output}")
    plt.show()
