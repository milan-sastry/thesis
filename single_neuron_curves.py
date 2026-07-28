
import numpy as np
import matplotlib.pyplot as plt
import tuning_curves as tc
import analysis as an
import load_weights as lw
from grating import generate_grating_response, generate_moving_grating_response
from bars import generate_bar_response
from visualize import visualize_stimulus_with_tm1_inputs, visualize_responses
import utils as _utils
from tuning_histograms import TuningHistograms, compare_histograms


if __name__ == "__main__":
    centers = {
        1521: (18.5, 17.0), # Dm3p
        1685: (18,16.5), # Dm3q
        972: (18.5, 17.5), # Dm3v
        2414: (18, 15), # TmY9q
        2498: (18,17),# TmY9q⊥
        2168: (17.5, 16.5), # TmY4
        
        
        1481: (10.5,18), # Dm3p with most total TmY9q⊥ inputs
        2359: (10.5,18), # top Tmy9q output of 1481


        1456: (11,10), # Dm3p with large number of Tmy9q⊥ inputs and high total after normalization
        1559: (21,15), # Dm3p with most unique TmY9q⊥ inputs

        982: (20,14.5), # Dm3v with second most total TmY9q⊥ inputs
        962: (10,12), # Dm3v with most unique TmY9q⊥ inputs

        2452: (13,6), # TmY9q⊥ -- most synapses to Dm3v + Dm3p
        2608: (11,18.5), #Tmy9q⊥ -- many synapses to Dm3p
        2547: (14,12.5), #Tmy9q⊥ -- many incoming synapses from TmY9q
        2518: (0,0), #TmY9q⊥ -- many incoming synapses from TmY9q

        2415: (12,17), #TmY9q -- many outgoing synapses to TmY9q⊥
    }

    # top 3 neurons ranked on range * osi_f1 for each type
    ids_by_type = {
        "Dm3p": [1159, 1347, 1346, 1231, 1412, 1401, 1427,1509],
        "Dm3q": [1969, 2006, 1620, 2020,2033,1830,1922,1967],
        "Dm3v": [1031, 1001, 816, 1021],
        "TmY9q⊥": [2540, 2480, 2596, 2570],
        "TmY9q": [2319, 2324, 2356, 2336],
        "TmY4": [2134, 2194, 2201, 2068]
    }

    same_center = [1216,2037,1050,2154,2319,2448]
    shared_center = (25,16)

    neuron_index = 2449
    use_bar = False
    moving_grating = True
    n_cycles = 4  # integer cycles within grating_steps — ensures clean FFT / unbiased mean
    rank_component = "f1"  # which component to rank OSI on: "f0" or "f1"

    # Representative-neuron selection config
    selection_scale_factor = 1.0  # which scale factor's distribution to select from
    selection_metric = "osi"       # stat key: "osi", "osi_classical", "fwhm_raw", etc.
    selection_n = 6                # how many representative neurons to pick

    target_dm3s, _, _, _ = an.get_postsynaptic_targets(neuron_index=neuron_index, include_types=["Dm3p", "Dm3q", "Dm3v"])
    source_tm1s, _, _, _ = an.get_presynaptic_inputs(neuron_index=neuron_index, include_types=["Tm1"])
    print(source_tm1s)
    



    # p_center, q_center = centers[neuron_index]


    # scale_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
    # scale_factors = [0.0,1.0,2.0,4.0]
    # scale_factors = [0.0,1.0]
    # scale_factors = [-0.1,-0.125, -0.15, -0.175, -0.2, -0.225, -0.25]
    scale_factors= [1.0]


    dt = 0.1
    baseline_steps = 400  # default in generate_moving_grating_response
    grating_steps = 400
    grating_onset_t = baseline_steps * dt
    grating_dur = grating_steps * dt
    # analysis_window = (grating_dur / 5, grating_dur)
    omega = n_cycles * 2 * np.pi / (grating_steps * dt)
    temporal_freq = omega / (2 * np.pi)  # = n_cycles / (grating_steps * dt)
    offset = 0.5
    amplitude = 0.5

    generate_histograms = True  # True: histograms over all neurons per type; False: tuning curve plots for ids_by_type

    curves_arr_by_type = {t: [] for t in ids_by_type}
    params_arr_by_type = {t: [] for t in ids_by_type}
    hist_groups_by_type = {t: [] for t in ids_by_type}  # [(TuningHistograms, label), ...]

    cross_orientation = False
    cross_orientation_angle = 30
    cross_orientation_type = "Dm3q"  # only this type is analyzed when cross_orientation=True

    for scale_factor in scale_factors:
        scale_by_connection_type = {
            # ('Dm3p', 'Dm3v'): scale_factor,
            # ('Dm3q', 'Dm3v'): scale_factor,
            # ('Dm3p', 'Dm3q'): scale_factor,
            # ('Dm3q', 'Dm3p'): scale_factor,
            # ('Dm3v', 'Dm3p'): scale_factor,
            # ('Dm3v', 'Dm3q'): scale_factor, 
            # ('Dm3p', 'Dm3p'): scale_factor,
            # ('Dm3q', 'Dm3q'): scale_factor,
            # ('Dm3v', 'Dm3v'): scale_factor,
            # ('Dm3v', 'TmY4'): scale_factor,
            # ('Dm3p', 'TmY9q'): scale_factor,
            # ('Dm3q', 'TmY9q⊥'): scale_factor,
            # ('Dm3v', 'TmY9q'): scale_factor,
            # ('Dm3q', 'TmY9q'): scale_factor,
            # ('Dm3v', 'TmY9q⊥'): scale_factor,
            # ('Dm3p', 'TmY4'): scale_factor,
            # ('Dm3q', 'TmY4'): scale_factor,
            # ('Dm3p', 'TmY9q⊥'): scale_factor,
            # ('TmY9q', 'TmY9q⊥') : scale_factor,
            # ('TmY9q⊥', 'TmY9q') : scale_factor,
            # ('TmY9q', 'TmY9q') : scale_factor,
            # ('TmY9q⊥', 'TmY9q⊥') : scale_factor,
            # ('TmY9q', 'TmY4') : scale_factor,
            # ('TmY9q⊥', 'TmY4') : scale_factor,
            # ('TmY4', 'TmY4') : scale_factor,
            # ('TmY4', 'TmY9q') : scale_factor,
            # ('TmY4', 'TmY9q⊥') : scale_factor,
            # ('TmY9q⊥', 'Dm3p') : scale_factor,
            # ('TmY9q⊥', 'Dm3v') : scale_factor,
            # ('TmY9q⊥', 'Dm3q') : scale_factor,
            # ('TmY4', 'Dm3p') : scale_factor,
            # ('TmY4', 'Dm3v') : scale_factor,
            # ('TmY4', 'Dm3q') : scale_factor,
            # ('TmY9q', 'Dm3p') : scale_factor,
            # ('TmY9q', 'Dm3v') : scale_factor,
            # ('TmY9q', 'Dm3q') : scale_factor,
        }
        dm_bias = scale_factor
        tmy_bias = -0.1
        dm3_tau = scale_factor
        model_settings = {
            "scale_by_connection_type": scale_by_connection_type,
            # "vrest_by_type": {
            #     "Dm3p": -0.175,
            #     "Dm3q": -0.175,
            #     "Dm3v": -0.15,
            #     "TmY9q": -0.1,
            #     "TmY9q⊥": -0.1,
            #     "TmY4": -0.1,
            # },
            "vrest_init": -0.05,
            # "remove_reciprocal": True,
            # "tau_by_type" : {
            #     "Dm3p": dm3_tau,
            #     "Dm3q": dm3_tau,
            #     "Dm3v": dm3_tau,
            #     },
        }

        runs_data = []
        cros_orientation_data = []
        if cross_orientation:
            v_final_cross, v_hist_cross, t_cross = generate_moving_grating_response(
                model_settings=model_settings,
                angle=cross_orientation_angle,
                spatial_frequency=2*np.pi/(6*2/np.sqrt(3)), # wavelength = 4 ommatidia
                n_cycles=n_cycles,
                offset=offset,
                amplitude=amplitude,
                dt=dt,
                steps=grating_steps,
                baseline_steps=baseline_steps,
                cross_orientation=False)
        for angle in range(0, 360, 30):
            if use_bar:
                v_final, v_hist, t, bar = generate_bar_response(
                    angle=angle,
                    width=1,
                    length=10.0,
                    amplitude=1.0,
                    bar_duration=100,
                    model_settings=model_settings,
                    center=(p_center, q_center),
                    # use_flash=True,
                    # mean_duration=100,
                    # mean_intensity=0.25,
                    sigma=0.5,
                )
                # if scale_factor == 0.0:
                #     visualize_stimulus_with_tm1_inputs(
                #             stimulus=bar,
                #             neuron_index=neuron_index,
                #             title=f"Bar {angle}°, Scale {scale_factor}, Neuron {neuron_index}",
                #         )
                    
            else:
                if moving_grating:
                    if cross_orientation:
                        v_final_theta, v_hist_theta, t_theta = generate_moving_grating_response(
                            model_settings=model_settings,
                            angle=angle,
                            spatial_frequency=2*np.pi/(6.5*2/np.sqrt(3)), # wavelength = 4 ommatidia
                            n_cycles=n_cycles,
                            offset=offset,
                            amplitude=amplitude,
                            dt=dt,
                            steps=grating_steps,
                            baseline_steps=baseline_steps,
                            cross_orientation=False)
                    v_final, v_hist, t = generate_moving_grating_response(
                        model_settings=model_settings,
                        angle=angle,
                        spatial_frequency=2*np.pi/(6.5*2/np.sqrt(3)), # wavelength = 4 ommatidia
                        n_cycles=n_cycles,
                        offset=offset,
                        amplitude=amplitude,
                        dt=dt,
                        steps=grating_steps,
                        baseline_steps=baseline_steps,
                        cross_orientation=cross_orientation,
                        cross_orientation_angle=cross_orientation_angle,
                        # center=(p_center, q_center),
                    )
                else:
                    v_final, v_hist, t = generate_grating_response(
                    angle=angle,
                    spatial_frequency=2*np.pi/6,
                    offset=0.25,
                    amplitude=0.75,
                    grating_duration=100,
                    model_settings=model_settings,
                    center=(p_center, q_center)
                )
            runs_data.append({
                "v_final": v_final,
                "v_history": v_hist,
                "t": t,
                "angle": angle,
            })
            if cross_orientation:
                cros_orientation_data.append({
                    "v_final_cross": v_final_cross,
                    "v_hist_cross": v_hist_cross,
                    "t_cross": t_cross,
                    "v_final_theta": v_final_theta,
                    "v_hist_theta": v_hist_theta,
                    "t_theta": t_theta,
                    "angle": angle,
                })

            # visualize_responses(v_final[None, :], {"v": v_hist[None, :, :], "t": t}, neuron_indices=[1159], show_f1_fit=True, temporal_freq=omega/(2*np.pi), grating_onset_t=grating_onset_t, use_relu=False, exclude_types=(), title=f"{angle}°, Scale {scale_factor}")



        results = {"runs": runs_data}


        for type in ids_by_type.keys():
            if generate_histograms:
                ids = list(np.where(_utils.to_numpy(lw.neuron_types) == type)[0])
            else:
                ids = ids_by_type[type]

            curves = tc.tuning_curve(
                    results,
                    fit=False,
                    fit_period_deg=360.0,
                    active_only=False,
                    # exclude_types=["Tm1", "TmY4", "TmY9q", "TmY9q⊥", "Dm3q", "Dm3v"],
                    use_fourier=True,
                    aggregation="individual",
                    neuron_ids=ids,
                    use_flash=False,
                    temporal_freq=temporal_freq,
                    grating_onset_t=grating_onset_t,
                    response_component="f1",  # always compute and display f0 and f1
                    use_relu=True,
                    fwhm=True,
                    #analysis_window=analysis_window,
                )

            curves_arr_by_type[type].append(curves)
            params_arr_by_type[type].append(scale_factor)
            if generate_histograms:
                hists = TuningHistograms(curves, component="auto", period_deg=360.0)
                label = str(scale_factor) if len(scale_factors) > 1 else None
                hist_groups_by_type[type].append((hists, label))

    for type in ids_by_type.keys():
        if cross_orientation and type != cross_orientation_type:
            continue
        all_curves = curves_arr_by_type[type][-1]

        if generate_histograms:
            for hists, _ in hist_groups_by_type[type]:
                hists.summary()
            compare_histograms(
                hist_groups_by_type[type],
                filename=f"tuning_histograms_{type}.png",
                title=f"Tuning properties — {type}  (n={len(hist_groups_by_type[type][0][0].stats)})",
                fit_kde=False,
            )
            plt.show()

            # Report neurons closest to median and 75th-percentile OSI
            hists_obj = hist_groups_by_type[type][0][0]
            print(f"\n--- {type}: representative neurons by OSI ---")
            for pct, label in [(50, "median"), (75, "75th pct")]:
                target, closest = hists_obj.find_neurons_near_percentile(pct, stat_key="osi", n=3)
                print(f"  {label} OSI = {target:.4f}")
                for key, val, dist in closest:
                    neuron_id = key.split("#")[1] if "#" in key else key
                    print(f"    neuron {neuron_id:>6s}  OSI={val:.4f}  (Δ={dist:.4f})")

            if True:  # set False to use delta-based (interesting) selection instead
                # Representative neurons: 2 near median vector OSI, 2 near median classical OSI,
                # 1 near 75th pct vector OSI, 1 near 75th pct classical OSI (deduplicated)
                def _ids_near(stat_key, pct, n, exclude=()):
                    _, closest = hists_obj.find_neurons_near_percentile(
                        pct, stat_key=stat_key, n=n + len(exclude) + 5
                    )
                    result = []
                    for key, _, _ in closest:
                        nid = int(key.split("#")[1])
                        if nid not in exclude and nid not in result:
                            result.append(nid)
                            if len(result) >= n:
                                break
                    return result

                rep_ids_ordered = []
                seen = set()
                for stat_key, pct, count in [
                    ("osi", 50, 2),
                    ("osi_classical", 50, 2),
                    ("osi", 75, 1),
                    ("osi_classical", 75, 1),
                ]:
                    for nid in _ids_near(stat_key, pct, count, exclude=seen):
                        if nid not in seen:
                            seen.add(nid)
                            rep_ids_ordered.append(nid)
                rep_ids = rep_ids_ordered
            else:
                # Median-based selection: find selection_n neurons closest to the
                # median of selection_metric at selection_scale_factor.
                params = params_arr_by_type[type]
                idx_sel = next((i for i, p in enumerate(params) if p == selection_scale_factor), None)
                rep_ids = []
                if idx_sel is not None:
                    hists_sel = hist_groups_by_type[type][idx_sel][0]
                    target, closest = hists_sel.find_neurons_near_percentile(
                        50, stat_key=selection_metric, n=selection_n
                    )
                    rep_ids = [int(key.split("#")[1]) for key, _, _ in closest]
                    print(f"\n--- {type}: {selection_n} neurons near median {selection_metric} "
                          f"(scale={selection_scale_factor}) ---")
                    print(f"  median {selection_metric} = {target:.4f}")
                    for key, val, dist in closest:
                        nid = int(key.split("#")[1])
                        print(f"  neuron {nid:>6d}  {selection_metric}={val:.4f}  (Δ={dist:.4f})")

            rep_curves = tc.tuning_curve(
                results,
                fit=False,
                fit_period_deg=360.0,
                active_only=False,
                use_fourier=True,
                aggregation="individual",
                neuron_ids=rep_ids,
                use_flash=False,
                temporal_freq=temporal_freq,
                grating_onset_t=grating_onset_t,
                response_component="both",
                use_relu=True,
                fwhm=True,
            )
            tc.plot_representative_curves_with_polar(
                rep_curves,
                title=f"Representative tuning curves — {type} (scale={scale_factor})",
            )
            plt.show()

            # Show both conditions on the same axes for each selected neuron
            rep_curves_arr = [
                {f"{type}#{nid}": c[f"{type}#{nid}"] for nid in rep_ids if f"{type}#{nid}" in c}
                for c in curves_arr_by_type[type]
            ]
            tc.plot_curves_by_param(
                rep_curves_arr,
                params_arr_by_type[type],
                show_points=True,
                cmap='viridis',

            )
            plt.show()
        else:
            tc.plot_curves_by_param(
                    curves_arr_by_type[type],
                    params_arr_by_type[type],
                    filename=f"scaled_individual_{type}.png",
                    show_points=True,
                    # ylim=(0.0, 0.1),
                )
            # Polar plots for specific neurons
            # tc.plot_polar_tuning_curves(
            #     all_curves,
            #     title=f"Direction tuning — top {len(all_curves)} neurons (scale={params_arr_by_type[type][-1]})",
            # )
            plt.show()

    # ------------------------------------------------------------------
    # Reciprocal comparison histograms
    # To run: set generate_histograms=False, then execute this block.
    # ------------------------------------------------------------------
    def _run_angle_sweep(model_settings):
        """Run the moving grating sweep over all angles and return results dict."""
        runs = []
        for angle in range(0, 360, 30):
            v_final, v_hist, t = generate_moving_grating_response(
                model_settings=model_settings,
                angle=angle,
                spatial_frequency=2 * np.pi / (4 * 2 / np.sqrt(3)),
                n_cycles=n_cycles,
                offset=offset,
                amplitude=amplitude,
                dt=dt,
                steps=grating_steps,
                baseline_steps=baseline_steps,
            )
            runs.append({"v_final": v_final, "v_history": v_hist, "t": t, "angle": angle})
        return {"runs": runs}

    def _curves_for_type(results, cell_type):
        ids = list(np.where(_utils.to_numpy(lw.neuron_types) == cell_type)[0])
        return tc.tuning_curve(
            results,
            fit=False, fit_period_deg=360.0, active_only=False,
            use_fourier=True, aggregation="individual", neuron_ids=ids,
            use_flash=False, temporal_freq=temporal_freq,
            grating_onset_t=grating_onset_t, response_component="f1",
            use_relu=True, fwhm=True,
        )

    if False:  # change to True to run reciprocal comparison
        base_settings = model_settings  # uses the last model_settings from the loop above
        groups_per_type = {t: [] for t in ids_by_type}
        for remove_recip, label in [(False, "full"), (True, "only Tm1 inputs")]:
            settings = {**base_settings, "remove_reciprocal": remove_recip}
            results = _run_angle_sweep(settings)
            for cell_type in ids_by_type:
                curves = _curves_for_type(results, cell_type)
                groups_per_type[cell_type].append((TuningHistograms(curves), label))

        for cell_type, groups in groups_per_type.items():
            compare_histograms(
                groups,
                filename=f"tuning_histograms_{cell_type}_reciprocal_comparison.png",
                title=f"Reciprocal comparison — {cell_type}",
            )
            plt.show()

