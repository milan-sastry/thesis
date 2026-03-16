
import numpy as np
import matplotlib.pyplot as plt
import tuning_curves as tc
import analysis as an
from grating import generate_grating_response 
from bars import generate_bar_response
from visualize import visualize_stimulus_with_tm1_inputs, visualize_responses


if __name__ == "__main__":
    curves_arr = []
    params_arr = []

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

    same_center = [1216,2037,1050,2154,2319,2448]
    shared_center = (25,16)

    neuron_index = 1521
    use_bar = False

    target_dm3s, _, _, _ = an.get_postsynaptic_targets(neuron_index=neuron_index, include_types=["Dm3p", "Dm3q", "Dm3v"])
    source_tm1s, _, _, _ = an.get_presynaptic_inputs(neuron_index=neuron_index, include_types=["Tm1"])
    print(source_tm1s)
    



    p_center, q_center = centers[neuron_index]


    # scale_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    scale_factors = [1.0]

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
            # ('Dm3v', 'TmY4'): scale_factor,
            # ('Dm3p', 'TmY9q'): scale_factor,
            # ('Dm3q', 'TmY9q⊥'): scale_factor,
            # ('Dm3v', 'TmY9q'): scale_factor,
            # ('Dm3q', 'TmY9q'): scale_factor,
            # ('Dm3v', 'TmY9q⊥'): scale_factor,
            # ('Dm3p', 'TmY4'): scale_factor,
            # ('Dm3q', 'TmY4'): scale_factor,
            # ('Dm3p', 'TmY9q⊥'): scale_factor,

            # ('TmY9q', 'Dm3p') : scale_factor,
            # ('TmY9q', 'Dm3v') : scale_factor,
            # ('TmY9q', 'Dm3q') : scale_factor,

            # ('TmY9q⊥', 'Dm3p') : scale_factor,
            # ('TmY9q⊥', 'Dm3v') : scale_factor,
            # ('TmY9q⊥', 'Dm3q') : scale_factor,

            # ('TmY4', 'Dm3p') : scale_factor,
            # ('TmY4', 'Dm3v') : scale_factor,
            # ('TmY4', 'Dm3q') : scale_factor,



            # ('TmY9q⊥', 'Dm3v') : scale_factor,
            # ('TmY9q', 'TmY9q⊥') : scale_factor,
        }
        model_settings = {
            "scale_by_connection_type": scale_by_connection_type,
            # "vrest_init": -0.1,
        }

        runs_data = []
        for angle in range(0, 180, 15):
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
                v_final, v_hist, t = generate_grating_response(
                angle=angle,
                spatial_frequency=2*np.pi/4,
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

            visualize_responses(v_final[None, :], {"v": v_hist[None, :, :], "t": t}, neuron_indices=source_tm1s, exclude_types=(),title=f"Bar {angle}°, Scale {scale_factor}")



        results = {"runs": runs_data}
        curves = tc.tuning_curve(
            results,
            fit=True,
            active_only=False,
            aggregation="individual",
            neuron_ids=[neuron_index], #target_dm3s[:7].tolist() +
            use_flash=False,
        )
        # print(curves)
        curves_arr.append(curves)
        params_arr.append(scale_factor)

    tc.plot_curves_by_param(
        curves_arr,
        params_arr,
        filename="flash_bar_Dm3_scaled_individual_1.0.png",
        show_points=True,
        ylim=(-0.1, 1.0),
    )
    plt.show()