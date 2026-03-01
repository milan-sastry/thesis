from stimulus import StimulusGenerator
import load_weights as lw
import numpy as np
from network import DrosophilaOpticLobeCircuit
from dataset import filter_model_kwargs
from utils import to_numpy


def compute_pq_vector(angle: float, spatial_frequency: float = 0.1):
    """
    Compute the p and q components of the wave vector for a given angle and spatial frequency.
    """
    p = -np.cos(np.radians(angle)) + np.sin(np.radians(angle)) / np.sqrt(3)
    q = np.cos(np.radians(angle)) + np.sin(np.radians(angle)) / np.sqrt(3)
    return spatial_frequency * p, spatial_frequency * q

def generate_grating_response(
    angle,
    spatial_frequency,
    amplitude,
    center=None,
    mean_duration=50,
    mean_intensity=0.5,
    grating_duration=50,
    model_settings=None,
    offset=0.5,
):
    """
    Generate a response to a full-field baseline followed by an oriented sine grating stimulus.
    """
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids
    )
    p,q = compute_pq_vector(angle, spatial_frequency=spatial_frequency)

    grating = stimulus_generator.create_sine_grating(
        wave_vector=(p, q),
        phase=np.pi/2,
        amplitude=amplitude,
        offset=offset,
        center=center,
    ) 

    # stimulus_generator.visualize_bar(grating)
    # plt.show()
    mean_gray = stimulus_generator.create_mean_gray(intensity=mean_intensity)
    blocks = [(mean_gray, mean_duration), (grating+mean_gray, grating_duration)]
    sequence = stimulus_generator.sequence_from_blocks(blocks)
    stimulus = stimulus_generator.to_torch(sequence)
    model_kwargs = filter_model_kwargs(model_settings)
    model = DrosophilaOpticLobeCircuit(
        lw.neuron_types,
        lw.source_indices,
        lw.target_indices,
        lw.weights,
        **model_kwargs,
    )
    v_final, history = model(stimulus, return_history=True)
    v_final_np = to_numpy(v_final.squeeze(0), dtype=np.float32)
    v_hist_np = to_numpy(history["v"].squeeze(0), dtype=np.float32)
    t_np = to_numpy(history["t"], dtype=np.float32)


    return v_final_np, v_hist_np, t_np


# if __name__ == "__main__":
#     # Build and plot tuning curves across a parameter sweep without storing runs.
#     scale_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#     curves_arr = []
#     params_arr = []



#     centers = {
#         1521: (18.5, 17.0), # Dm3p
#         1685: (18,16.5), # Dm3q
#         972: (18.5, 17.5), # Dm3v
#         2414: (18, 15), # TmY9q
#         2498: (18,17),# TmY9q⊥
#         2168: (17.5, 16.5), # TmY4
        
        
#         1481: (10.5,18), # Dm3p with most total TmY9q⊥ inputs
#         1456: (11,10), # Dm3v with large number of Tmy9q⊥ inputs and high total after normalization
#         982: (20,14.5), # Dm3v with second most total TmY9q⊥ inputs
#         962: (10,12), # Dm3v with most unique TmY9q⊥ inputs
#         2452: (13,6), # TmY9q⊥ -- most synapses to Dm3v + Dm3p
#     }

#     neuron_index = 1456
#     p_center, q_center = centers[neuron_index]
#     input_indices, input_cell_ids, input_types, input_synapse_counts = get_presynaptic_inputs(neuron_index)
#     output_indices, output_cell_ids, output_types, output_synapse_counts = get_postsynaptic_targets(neuron_index)

#     dm3v_indices = output_indices[np.where(output_types == "Dm3v")[0]]
#     print(dm3v_indices)
#     dm3p_indices = output_indices[np.where(output_types == "Dm3p")[0]]


#     for scale_factor in scale_factors:
#         scale_by_connection_type = {
#             # ('Dm3p', 'Dm3v'): scale_factor,
#             # ('Dm3q', 'Dm3v'): scale_factor,
#             # ('Dm3p', 'Dm3q'): scale_factor,
#             # ('Dm3q', 'Dm3p'): scale_factor,
#             # ('Dm3v', 'Dm3p'): scale_factor,
#             # ('Dm3v', 'Dm3q'): scale_factor, 
#             # ('Dm3p', 'Dm3p'): scale_factor,
#             # ('Dm3q', 'Dm3q'): scale_factor,
#             # ('Dm3v', 'Dm3v'): scale_factor,
#             # ('Dm3v', 'TmY4'): scale_factor,
#             # ('Dm3p', 'TmY9q'): scale_factor,
#             # ('Dm3q', 'TmY9q⊥'): scale_factor,
#             # ('Dm3v', 'TmY9q'): scale_factor,
#             # ('Dm3v', 'TmY9q⊥'): scale_factor,
#             # ('Dm3p', 'TmY4'): scale_factor,
#             # ('Dm3q', 'TmY4'): scale_factor,
#             # ('Dm3p', 'TmY9q⊥'): scale_factor,
#             # ('Dm3q', 'TmY9q'): scale_factor,
#             ('TmY9q⊥', 'Dm3p') : scale_factor,
#             # ('TmY9q⊥', 'Dm3q') : scale_factor,
#             ('TmY9q⊥', 'Dm3v') : scale_factor,
#         }
#         # weights = lw.scale_weights_by_connection_type(lw.W_raw, scale_by_connection_type, lw.neuron_types)
#         # weights = lw.normalize_weights_diagonal(weights).tocoo()
#         # weights = torch.tensor(
#         #     np.array(weights.data, dtype=np.float64, copy=True).tolist(), 
#         #     dtype=torch.float32
#         # )
#         model_settings = {
#             "scale_by_connection_type": scale_by_connection_type,
#             # "vrest_init": -0.1,
#         }
#         runs_data = []
#         for angle in range(0, 180, 15):
#             v_final, v_hist, t = generate_grating_response(
#                 angle=angle,
#                 mean_intensity=0.1, 
#                 spatial_frequency=1.5, 
#                 offset=0.45, 
#                 amplitude=0.45, 
#                 mean_duration=100, 
#                 grating_duration=100, 
#                 model_settings=model_settings, 
#                 center=(p_center, q_center)
#             )
#             runs_data.append({
#                 "v_final": v_final, 
#                 "v_history": v_hist,
#                 "t": t,
#                 "angle": angle,
#             })
#             #visualize_responses(v_final[None, :], {"v": v_hist[None, :, :], "t": t}, neuron_indices=[neuron_index], title=f"Grating {angle}°, Scale {scale_factor}")

#         results = {"runs": runs_data}
#         curves = gv.tuning_curve(
#             results,
#             flash_windows=(10, 20.0),
#             baseline_window=(0, 10.0),
#             fit=True,
#             active_only=False,
#             aggregation="individual",
#             neuron_ids=[neuron_index],
#         )
#         curves_arr.append(curves)
#         params_arr.append(scale_factor)

#     gv.plot_curves_by_param(
#         curves_arr,
#         params_arr,
#         filename="flash_grating_Dm3_scaled_individual_1.0.png",
#         show_points=True,
#         ylim=(-0.1, 1),
#     )
#     # v_final = runs_data[11]["v_final"]
#     # v_history = runs_data[11]["v_history"]
#     # t = runs_data[11]["t"]
#     # visualize_responses(v_final[None, :], {"v": v_history[None, :, :], "t": t}, neuron_indices=[2168])

#     plt.show()

