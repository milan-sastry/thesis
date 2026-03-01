from grating import generate_grating_response
from grating_visualization import tuning_curve, plot_tuning_curves
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    runs_data = []
    scale = 1.0
    model_settings = {
        "remove_reciprocal": True,
        "vrest_by_type": {
            "TmY9q": -0.1,
            "TmY9q⊥": -0.1,
            "TmY4": -0.1,
            "Dm3p": -0.1,
            "Dm3q": -0.1,
            "Dm3v": -0.1,
        },
        "scale_by_connection_type": {
            # ('Dm3p', 'Dm3v'): scale,
            # ('Dm3p', 'Dm3q'): scale,
            # ('Dm3p', 'TmY9q'): scale,

            # ('Dm3q', 'Dm3v'): scale,
            # ('Dm3q', 'Dm3p'): scale,
            # ('Dm3q', 'TmY9q⊥'): scale,

            # ('Dm3v', 'Dm3p'): scale,
            # ('Dm3v', 'Dm3q'): scale,
            # ('Dm3v', 'TmY4'): scale,

       
            # ('TmY4', 'TmY4'): scale,
            # ('TmY4', 'Dm3v'): scale,

            # ('TmY9q', 'TmY9q'): scale,
            # ('TmY9q', 'TmY9q⊥'): scale,
            # ('TmY9q', 'Dm3p'): scale,

            # ('TmY9q⊥', 'TmY9q⊥'): scale,
            ('TmY9q⊥', 'Dm3v'): scale,
            ('TmY9q⊥', 'Dm3p'): scale,
            ('TmY9q⊥', 'Dm3q'): scale,
        }
    }

    for angle in range(0, 180, 15):
        print(f"Running angle {angle}°...")
        v_final, v_hist, t = generate_grating_response(
            model_settings=model_settings,
            angle=angle,
            spatial_frequency=1.5,
            mean_intensity=0.1,
            offset=0.5,
            amplitude=0.5,
            mean_duration=100,
            grating_duration=100,
        )
        runs_data.append({
            "angle": angle,
            "v_history": v_hist,
            "t": t,
        })

    results = {"runs": runs_data}
    curves = tuning_curve(
        results,
        flash_windows=(10.0, 20.0),
        baseline_window=(0.0, 10.0),
        fit=True,
        exclude_types=("Tm1",),
    )
    dm3_types = [t for t in curves if t.startswith("Dm3")]
    tmy_types = [t for t in curves if not t.startswith("Dm3")]
    # for curve in curves:
    #     print(curves[curve]["fit"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_tuning_curves(curves, types=dm3_types, ax=ax1, ylim=(0.0, 1.0))
    ax1.set_title("Dm3 Tuning Curves")
    plot_tuning_curves(curves, types=tmy_types, ax=ax2, ylim=(0.0, 0.25))
    ax2.set_title("TmY Tuning Curves")
    plt.tight_layout()
    plt.savefig("tuning_curves_control.png", dpi=150, bbox_inches="tight")
    plt.show()