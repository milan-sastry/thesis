"""
2-D bias grid search over (dm3_bias, tmy_bias) in [-0.2, -0.05].

All Dm3 types share dm3_bias; all TmY types share tmy_bias.
Objective: mean over 6 non-Tm1 types of (median_osi * median_pref_amp).

Usage:
    python optimize_model.py
    python optimize_model.py --n-grid 10   # 10x10 = 100 evals
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import load_weights as lw
import tuning_curves as tc
import utils as _utils
from grating import generate_moving_grating_response
from tuning_histograms import TuningHistograms


# ---------------------------------------------------------------------------
# Constants (mirror single_neuron_curves.py)
# ---------------------------------------------------------------------------

DM3_TYPES  = ["Dm3p", "Dm3q", "Dm3v"]
TMY_TYPES  = ["TmY9q", "TmY9q\u22a5", "TmY4"]
CELL_TYPES = DM3_TYPES + TMY_TYPES

ANGLES          = list(range(0, 180, 30))
DT              = 0.1
BASELINE_STEPS  = 300
GRATING_STEPS   = 300
N_CYCLES        = 3
OFFSET          = 0.5
AMPLITUDE       = 0.5
SPATIAL_FREQ    = 2 * np.pi / (6.5 * 2 / np.sqrt(3))
GRATING_ONSET_T = BASELINE_STEPS * DT
OMEGA           = N_CYCLES * 2 * np.pi / (GRATING_STEPS * DT)
TEMPORAL_FREQ   = OMEGA / (2 * np.pi)

# Search range
BIAS_LOW  = -0.2
BIAS_HIGH = -0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ids_by_type: dict[str, list[int]] = {}

def _get_ids(cell_type: str) -> list[int]:
    if cell_type not in _ids_by_type:
        _ids_by_type[cell_type] = list(
            np.where(_utils.to_numpy(lw.neuron_types) == cell_type)[0]
        )
    return _ids_by_type[cell_type]


def _run_grating_sweep(model_settings: dict) -> dict:
    runs = []
    for angle in ANGLES:
        v_final, v_hist, t = generate_moving_grating_response(
            model_settings=model_settings,
            angle=angle,
            spatial_frequency=SPATIAL_FREQ,
            n_cycles=N_CYCLES,
            offset=OFFSET,
            amplitude=AMPLITUDE,
            dt=DT,
            steps=GRATING_STEPS,
            baseline_steps=BASELINE_STEPS,
        )
        runs.append({"v_final": v_final, "v_history": v_hist, "t": t, "angle": angle})
    return {"runs": runs}


def _eval(dm3_bias: float, tmy_bias: float) -> tuple[float, dict]:
    """
    Run one grating sweep and return (mean_score, per_type_breakdown).

    Score per type = median_osi * median_pref_amp.
    Mean score     = mean of per-type scores across all 6 types.
    """
    vrest = {ct: dm3_bias for ct in DM3_TYPES}
    vrest.update({ct: tmy_bias for ct in TMY_TYPES})

    results = _run_grating_sweep({"vrest_by_type": vrest})

    breakdown: dict[str, dict] = {}
    type_scores: list[float] = []

    for cell_type in CELL_TYPES:
        ids = _get_ids(cell_type)
        if not ids:
            continue

        curves = tc.tuning_curve(
            results,
            fit=False,
            fit_period_deg=360.0,
            active_only=False,
            use_fourier=True,
            aggregation="individual",
            neuron_ids=ids,
            use_flash=False,
            temporal_freq=TEMPORAL_FREQ,
            grating_onset_t=GRATING_ONSET_T,
            response_component="f1",
            use_relu=True,
            fwhm=False,
        )
        hists    = TuningHistograms(curves, component="auto", period_deg=360.0)
        osi_vals = hists.get_values("osi",            valid_only=False)
        amp_vals = hists.get_values("pref_amplitude", valid_only=False)

        mask     = np.isfinite(osi_vals) & np.isfinite(amp_vals)
        osi_vals = osi_vals[mask]
        amp_vals = amp_vals[mask]

        if len(osi_vals) == 0:
            breakdown[cell_type] = {"median_osi": 0.0, "median_amp": 0.0, "score": 0.0}
            type_scores.append(0.0)
            continue

        med_osi = float(np.median(osi_vals))
        med_amp = float(np.median(amp_vals))
        score   = med_osi * med_amp
        breakdown[cell_type] = {
            "median_osi": round(med_osi, 4),
            "median_amp": round(med_amp, 4),
            "score":      round(score,   4),
        }
        type_scores.append(score)

    mean_score = float(np.mean(type_scores)) if type_scores else 0.0
    return mean_score, breakdown


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_bias_grid_search(n_grid: int = 8, output_path: str = "bias_grid_results.json"):
    dm3_values = np.linspace(BIAS_LOW, BIAS_HIGH, n_grid)
    tmy_values = np.linspace(BIAS_LOW, BIAS_HIGH, n_grid)
    n_total    = n_grid * n_grid

    print(f"=== Bias Grid Search ({n_grid}x{n_grid} = {n_total} evals) ===")
    print(f"dm3_bias: {BIAS_LOW} → {BIAS_HIGH}  ({n_grid} pts)")
    print(f"tmy_bias: {BIAS_LOW} → {BIAS_HIGH}  ({n_grid} pts)")
    print(f"Objective: mean over 6 types of (median_osi * median_pref_amp)\n")

    best_score = -np.inf
    best_dm3   = float(dm3_values[0])
    best_tmy   = float(tmy_values[0])
    eval_log: list[dict] = []
    idx = 0

    for dm3 in dm3_values:
        for tmy in tmy_values:
            idx += 1
            t0 = time.time()
            try:
                score, breakdown = _eval(float(dm3), float(tmy))
            except Exception as exc:
                print(f"[{idx:3d}/{n_total}] dm3={dm3:.3f} tmy={tmy:.3f}  ERROR: {exc}")
                continue
            elapsed = time.time() - t0

            marker = ""
            if score > best_score:
                best_score = score
                best_dm3   = float(dm3)
                best_tmy   = float(tmy)
                marker = "  <-- best"

            type_str = "  ".join(
                f"{ct}: osi={d['median_osi']:.3f} amp={d['median_amp']:.3f} s={d['score']:.3f}"
                for ct, d in breakdown.items()
            )
            print(f"[{idx:3d}/{n_total}] dm3={dm3:.3f} tmy={tmy:.3f}  score={score:.4f}  ({elapsed:.0f}s){marker}")
            print(f"          {type_str}")

            eval_log.append({
                "eval":      idx,
                "dm3_bias":  round(float(dm3), 4),
                "tmy_bias":  round(float(tmy), 4),
                "score":     round(score, 6),
                "elapsed_s": round(elapsed, 1),
                **{f"{ct}_{k}": v for ct, d in breakdown.items() for k, v in d.items()},
            })

    print("\n" + "=" * 60)
    print(f"BEST  dm3_bias={best_dm3:.4f}  tmy_bias={best_tmy:.4f}  score={best_score:.4f}")

    Path(output_path).write_text(json.dumps({
        "best_score":    best_score,
        "best_dm3_bias": best_dm3,
        "best_tmy_bias": best_tmy,
        "dm3_values":    dm3_values.tolist(),
        "tmy_values":    tmy_values.tolist(),
        "n_evaluations": idx,
        "eval_log":      eval_log,
    }, indent=2))
    print(f"Results saved to {output_path}")

    print("\n--- Copy-paste into model_settings ---")
    print('"vrest_by_type": {')
    for ct in DM3_TYPES:
        print(f'    "{ct}": {best_dm3:.4f},')
    for ct in TMY_TYPES:
        print(f'    "{ct}": {best_tmy:.4f},')
    print("},")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D bias grid search (dm3 x tmy)")
    parser.add_argument("--n-grid", type=int, default=8,
                        help="Points per axis (default: 8 → 64 evals)")
    parser.add_argument("--output", type=str, default="bias_grid_results.json")
    args = parser.parse_args()

    t_start = time.time()
    run_bias_grid_search(n_grid=args.n_grid, output_path=args.output)
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
