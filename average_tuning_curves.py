import numpy as np
import matplotlib.pyplot as plt
import tuning_curves as tc
import load_weights as lw
from grating import generate_moving_grating_response
import utils as _utils

if __name__ == "__main__":
    # scale_factors = [0.0, 1.0]
    scale_factors = [1.0]

    dt = 0.1
    baseline_steps = 400
    grating_steps = 400
    n_cycles = 4
    grating_onset_t = baseline_steps * dt
    grating_dur = grating_steps * dt
    omega = n_cycles * 2 * np.pi / (grating_steps * dt)
    temporal_freq = omega / (2 * np.pi)
    offset = 0.5
    amplitude = 0.5

    cell_types = ["Dm3p", "Dm3q", "Dm3v", "TmY4", "TmY9q", "TmY9q⊥"]

    curves_arr = []
    params_arr = []

    for scale_factor in scale_factors:
        scale_by_connection_type = {
            ('Dm3p', 'Dm3v'): scale_factor,
            ('Dm3q', 'Dm3v'): scale_factor,
            ('Dm3p', 'Dm3q'): scale_factor,
            ('Dm3q', 'Dm3p'): scale_factor,
            ('Dm3v', 'Dm3p'): scale_factor,
            ('Dm3v', 'Dm3q'): scale_factor, 
            ('Dm3p', 'Dm3p'): scale_factor,
            ('Dm3q', 'Dm3q'): scale_factor,
            ('Dm3v', 'Dm3v'): scale_factor,
            ('Dm3v', 'TmY4'): scale_factor,
            ('Dm3p', 'TmY9q'): scale_factor,
            ('Dm3q', 'TmY9q⊥'): scale_factor,
            ('Dm3v', 'TmY9q'): scale_factor,
            ('Dm3q', 'TmY9q'): scale_factor,
            ('Dm3v', 'TmY9q⊥'): scale_factor,
            ('Dm3p', 'TmY4'): scale_factor,
            ('Dm3q', 'TmY4'): scale_factor,
            ('Dm3p', 'TmY9q⊥'): scale_factor,
            ('TmY9q', 'Dm3p') : scale_factor,
            ('TmY9q', 'Dm3v') : scale_factor,
            ('TmY9q', 'Dm3q') : scale_factor,
            ('TmY9q', 'TmY9q⊥') : scale_factor,
            ('TmY9q⊥', 'TmY9q') : scale_factor,
            ('TmY9q', 'TmY9q') : scale_factor,
            ('TmY9q⊥', 'TmY9q⊥') : scale_factor,
            ('TmY9q', 'TmY4') : scale_factor,
            ('TmY9q⊥', 'TmY4') : scale_factor,
            ('TmY4', 'TmY4') : scale_factor,
            ('TmY4', 'TmY9q') : scale_factor,
            ('TmY4', 'TmY9q⊥') : scale_factor,
            ('TmY9q⊥', 'Dm3p') : scale_factor,
            ('TmY9q⊥', 'Dm3v') : scale_factor,
            ('TmY9q⊥', 'Dm3q') : scale_factor,
            ('TmY4', 'Dm3p') : scale_factor,
            ('TmY4', 'Dm3v') : scale_factor,
            ('TmY4', 'Dm3q') : scale_factor,
        }
        
        model_settings = {
            "scale_by_connection_type": scale_by_connection_type,
            "vrest_by_type": {
                "Dm3p": -0.175,
                "Dm3q": -0.175,
                "Dm3v": -0.15,
                "TmY9q": -0.1,
                "TmY9q⊥": -0.1,
                "TmY4": -0.1,
            },
        }

        runs_data = []
        for angle in range(0, 360, 30):
            v_final, v_hist, t = generate_moving_grating_response(
                model_settings=model_settings,
                angle=angle,
                spatial_frequency=2 * np.pi / (6.5 * 2 / np.sqrt(3)),
                n_cycles=n_cycles,
                offset=offset,
                amplitude=amplitude,
                dt=dt,
                steps=grating_steps,
                baseline_steps=baseline_steps,
            )
            runs_data.append({"v_final": v_final, "v_history": v_hist, "t": t, "angle": angle})

        results = {"runs": runs_data}

        curves = tc.tuning_curve(
            results,
            fit=True,
            fit_period_deg=360.0,
            active_only=False,
            use_fourier=True,
            aggregation="type_mean",
            use_flash=False,
            temporal_freq=temporal_freq,
            grating_onset_t=grating_onset_t,
            response_component="f1",
            use_relu=True,
            fwhm=True,
        )

        curves_arr.append(curves)
        params_arr.append(scale_factor)
        print(f"scale={scale_factor} done")
    use_plot_tuning_curves = True

    if use_plot_tuning_curves:
        # One panel for Dm3 types, one for TmY types, with SEM shading.
        # Only valid when curves_arr has a single entry (no param sweep).
        curves = curves_arr[0]
        dm3_types  = ["Dm3p", "Dm3q", "Dm3v"]
        tmy_types  = ["TmY4", "TmY9q", "TmY9q\u22a5"]

        fig, (ax_dm3, ax_tmy) = plt.subplots(1, 2, figsize=(12, 4))
        tc.plot_tuning_curves(curves, types=dm3_types, show_sem=True, show_fit=True, ax=ax_dm3)
        ax_dm3.set_title("Dm3 Tuning Curves")

        tc.plot_tuning_curves(curves, types=tmy_types, show_sem=True, show_fit=True, ax=ax_tmy)
        ax_tmy.set_title("TmY Tuning Curves")

        fig.tight_layout()
        fig.savefig("average_tuning_curves.png", bbox_inches="tight")
    else:
        tc.plot_curves_by_param(
            curves_arr,
            params_arr,
            types=cell_types,
            filename="average_tuning_curves.png",
            show_points=True,
        )
    plt.show()
