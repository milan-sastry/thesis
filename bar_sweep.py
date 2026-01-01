import os
import time
import torch
import numpy as np
from network import DrosophilaOpticLobeCircuit
import load_weights as lw
from utils import remove_reciprocal_connections
from visualize import visualize_responses
from stimulus import StimulusGenerator

if __name__ == "__main__":
    normalized = False
    remove_reciprocal = True
    custom_scale = False

    
    if normalized:
        _source_indices = lw.source_indices
        _target_indices = lw.target_indices
        _weights = lw.weights
    else:
        _source_indices = lw.source_indices_raw
        _target_indices = lw.target_indices_raw
        _weights = lw.weights_raw
    if remove_reciprocal:
        _source_indices, _target_indices, _weights = remove_reciprocal_connections(
                _source_indices, _target_indices, _weights, lw.neuron_types
            )
    
    if custom_scale:
        scale_by_connection_type = {
            ('Tm1', 'Dm3p'): 0.6,
            ('Tm1', 'Dm3q'): 0.75,
            ('Tm1', 'Dm3v'): 0.9,
            ('Tm1', 'TmY4'): 0.6,
            ('Tm1', 'TmY9q'): 0.9,
            ('Tm1', 'TmY9q⊥'): 0.75,
        }
    else:
        scale_by_connection_type = None

    model = DrosophilaOpticLobeCircuit(
        lw.neuron_types, _source_indices, _target_indices, _weights,
        remove_reciprocal=False,
        scale_by_connection_type=scale_by_connection_type
    )

    generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids
    )
    
    n_tm1 = model.n_per_type['Tm1']
    increment = 45

    angles = np.arange(0, 180, increment)
    
    print(f"Running bar sweep with {len(angles)} angles...")
    print(f"Angles: {angles}")
    all_outputs = []
    for angle in angles:
        print(f"Processing angle {angle}...")
        if angle == 45:
            width = 0.75
        else:
            width = 1.0
        stimulus = generator.create_bar(
            angle=angle,
            offset=0.0,
            width=width,
            length=15.0,
            intensity=1.0
        )
        fig, ax, scatter = generator.visualize_bar(stimulus)
        fig.savefig(f'bar_sweep_stimuli/bar_{angle}.png')
        tm1_input = generator.to_torch(stimulus, target_size=n_tm1)
        
        with torch.no_grad():
            v_final, history = model(tm1_input, steps=5000, return_history=True)
        
        text_output = visualize_responses(v_final, history, top_k=10, 
                                         show_plot=False, return_text=True)
    
        angle_section = f"angle: {angle}°\n"
        angle_section += text_output
        all_outputs.append(angle_section)
    
    output_file = f'./bar_sweep_results/bar_sweep_results_{"removed_reciprocal" if remove_reciprocal else ""}_{"normalized" if normalized else "raw"}_{increment}deg.txt'
    if os.path.exists(output_file):
        output_file = f'./bar_sweep_results/bar_sweep_results_{"removed_reciprocal" if remove_reciprocal else ""}_{"normalized" if normalized else "raw"}_{increment}deg_{time.time()}.txt'
    with open(output_file, 'w') as f:
        f.write("bar sweep results\n")
        f.write(f"Tested {len(angles)} orientations: {angles}\n")
        f.write("\n\n".join(all_outputs))
    
    print(f"\nAll results saved to {output_file}")

