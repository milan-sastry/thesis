import load_weights as lw
from visualize import visualize_responses, visualize_stimulus_with_tm1_inputs, save_tm1_inputs_grid
from network import DrosophilaOpticLobeCircuit
from stimulus import StimulusGenerator
from grating import compute_pq_vector
from analysis import get_presynaptic_inputs
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    neuron_index = 1521
    presynaptic_inputs = get_presynaptic_inputs(neuron_index)
    print(presynaptic_inputs)
    model = DrosophilaOpticLobeCircuit(
        lw.neuron_types,
        lw.source_indices,
        lw.target_indices,
        lw.weights,
    )
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids
    )

    # rect, bar = stimulus_generator.create_gaussian_bar(angle=45, width=1.5, length=10, sigma=0.25, intensity=1.0, p_center=18.5, q_center=17)
    # plt.show()
    # stimulus_generator.visualize_bar(bar)
    #stimulus = stimulus_generator.create_sine_grating(wave_vector=(p, q), phase=np.pi/2, amplitude=1, offset=0, center=(18.5, 17))
    #visualize_stimulus_with_tm1_inputs(neuron_index=neuron_index, stimulus=stimulus)
    centers = {
        1521: ((18.5, 17.0),60), # Dm3p
        1685: ((18,16.5),120), # Dm3q
        972: ((18.5, 17.5),0), # Dm3v
        2414: ((18, 15),120), # TmY9q
        2498: ((18,17),30),# TmY9q⊥
        2168: ((17.5, 16.5),90), # TmY4
    }
    stimuli = []
    mean_gray = stimulus_generator.create_mean_gray(intensity=0.1)
    for neuron_index, center in centers.items():
        # p, q = compute_pq_vector(center[1], spatial_frequency=1.5)
        # grating = stimulus_generator.create_sine_grating(wave_vector=(p, q), phase=np.pi/2, amplitude=0.5, offset=0.5, center=center[0])
        rect, bar = stimulus_generator.create_gaussian_bar(angle=center[1], width=1.5, length=10, sigma=0.5, intensity=1.0, p_center=center[0][0], q_center=center[0][1])
        stimuli.append(bar + mean_gray)
    save_tm1_inputs_grid(neuron_indices=centers.keys(), stimulus_by_neuron=stimuli, overlay=True, output_file="gaussian_bar_stimuli_grid.png")
    # stimulus_generator.visualize_bar(stimulus)
    # plt.show()
    # mean_gray = stimulus_generator.create_mean_gray(intensity=0.5)
    # blocks = [(mean_gray, 50), (stimulus, 50), (mean_gray, 50)]
    # sequence = stimulus_generator.sequence_from_blocks(blocks)
    # sequence = stimulus_generator.to_torch(sequence)

    # v_final, history = model(sequence, return_history=True)
    # visualize_responses(v_final, history, neuron_indices=[presynaptic_inputs], metric="peak", top_k=20)
    # plt.show()