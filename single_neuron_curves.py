
import numpy as np
import matplotlib.pyplot as plt
import grating_visualization as gv
import analysis as an
from grating import generate_grating_response 
from bars import generate_bar_response
from visualize import visualize_stimulus_with_tm1_inputs


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

        2415: (12,17), #TmY9q -- many outgoing synapses to TmY9q⊥
    }

    same_center = [1216,2037,1050,2154,2319,2448]
    shared_center = (25,16)

    neuron_index = 1481
    use_bar = True

    p_center, q_center = centers[neuron_index]
    input_indices, input_cell_ids, input_types, input_synapse_counts = an.get_presynaptic_inputs(neuron_index)
    output_indices, output_cell_ids, output_types, output_synapse_counts = an.get_postsynaptic_targets(neuron_index)

    dm3v_indices = output_indices[np.where(output_types == "Dm3v")[0]]
    dm3p_indices = output_indices[np.where(output_types == "Dm3p")[0]]
    print(dm3p_indices)


    scale_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # scale_factors = [1.0]

    for scale_factor in scale_factors:
        scale_factor *= 2
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

            # ('TmY9q', 'Dm3p') : scale_factor,
            # ('TmY9q', 'Dm3v') : scale_factor,
            # ('TmY9q', 'Dm3q') : scale_factor,

            ('TmY9q⊥', 'Dm3p') : scale_factor,
            ('TmY9q⊥', 'Dm3v') : scale_factor,
            ('TmY9q⊥', 'Dm3q') : scale_factor,

            # ('TmY4', 'Dm3p') : scale_factor,
            # ('TmY4', 'Dm3v') : scale_factor,
            # ('TmY4', 'Dm3q') : scale_factor,



            # ('TmY9q⊥', 'Dm3v') : scale_factor,
            ('TmY9q', 'TmY9q⊥') : scale_factor,
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
                    width=3,
                    length=20.0,
                    amplitude=0.9,
                    mean_duration=100,
                    mean_intensity=0.1,
                    bar_duration=100,
                    model_settings=model_settings,
                    center=(p_center, q_center),
                    sigma=0.5,
                )
                # for index in [2597, 2440, 2569, 2435, 2612, 2543, 2515, 2587, 2608, 2589, 2553]:
                #     visualize_stimulus_with_tm1_inputs(
                #         stimulus=bar,
                #         neuron_index=index,
                #         title=f"Bar {angle}°, Scale {scale_factor}, Neuron {index}",
                #     )
                #     plt.show()
                    
            else:
                v_final, v_hist, t = generate_grating_response(
                angle=angle,
                mean_intensity=0.1, 
                spatial_frequency=1.5, 
                offset=0.45, 
                amplitude=0.45, 
                mean_duration=100, 
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
            #visualize_responses(v_final[None, :], {"v": v_hist[None, :, :], "t": t}, neuron_indices=[neuron_index], title=f"Bar {angle}°, Scale {scale_factor}")


        results = {"runs": runs_data}
        curves = gv.tuning_curve(
            results,
            flash_windows=(10, 20.0),
            baseline_window=(0, 10.0),
            fit=True,
            active_only=False,
            aggregation="individual",
            neuron_ids=[1481, 2359],
        )
        curves_arr.append(curves)
        params_arr.append(scale_factor)

    gv.plot_curves_by_param(
        curves_arr,
        params_arr,
        filename="flash_bar_Dm3_scaled_individual_1.0.png",
        show_points=True,
        ylim=(0.0, 0.6),
    )
    plt.show()