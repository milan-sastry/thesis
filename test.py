from utils import remove_reciprocal_connections
from visualize import visualize_responses
import load_weights as lw
from network import DrosophilaOpticLobeCircuit
from stimulus import StimulusGenerator
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    normalized = False
    remove_reciprocal = True
    custom_scale = True

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
        remove_reciprocal=remove_reciprocal,
        scale_by_connection_type=scale_by_connection_type
    )
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids
    )
    n_tm1 = model.n_per_type['Tm1']
    angle = 90
    print("angle: ", angle)
    stimulus = stimulus_generator.create_bar(
        angle=angle,
        offset=0.0,
        width=0.5,
        length=15.0,
        intensity=1.0
    )

    tm1_input = stimulus_generator.to_torch(stimulus)
    stimulus_generator.visualize_bar(stimulus)
    plt.show()
    #visualize_type_to_type_connections(source_indices, target_indices, weights, neuron_types, aggregation='synapses')
    #plt.show()

    if tm1_input.shape[1] < n_tm1:
        padding = torch.zeros(1, n_tm1 - tm1_input.shape[1])
        tm1_input = torch.cat([tm1_input, padding], dim=1)

    with torch.no_grad():
        v_final, history = model(tm1_input, steps=5000, return_history=True)

    visualize_responses(v_final, history, top_k=10, show_plot=False)