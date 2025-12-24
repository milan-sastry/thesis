import torch
import numpy as np
import matplotlib.pyplot as plt
from network import DrosophilaOpticLobeCircuit
from load_weights import neuron_types, source_indices, target_indices, weights, tm1_coords


def visualize_responses(v_final, history, neuron_types, top_k=10, exclude_input=True):


    if torch.is_tensor(v_final):
        v_final = v_final.cpu().numpy()
    if torch.is_tensor(history['v']):
        history['v'] = history['v'].cpu().numpy()
    if torch.is_tensor(history['t']):
        history['t'] = history['t'].cpu().numpy()
    

    final_responses = v_final[0]  
    
    if exclude_input:
        non_tm1_mask = neuron_types != 'Tm1'
        final_responses_filtered = final_responses.copy()
        final_responses_filtered[~non_tm1_mask] = -np.inf  
        print(f"\nExcluding {np.sum(~non_tm1_mask)} Tm1 input neurons from analysis")
    else:
        final_responses_filtered = final_responses
    
    # Find top responding neurons
    top_indices = np.argsort(final_responses_filtered)[::-1][:top_k]
    top_responses = final_responses[top_indices]
    top_types = neuron_types[top_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    colors = plt.cm.tab10(np.arange(top_k))
    bars = ax1.barh(range(top_k), top_responses, color=colors, edgecolor='black')
    

    ax1.set_yticks(range(top_k))
    labels = [f"N{idx}: {typ}" for idx, typ in zip(top_indices, top_types)]
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Final Response')
    ax1.set_title(f'Top {top_k} Responding Neurons')
    ax1.invert_yaxis()  # Highest at top
    ax1.grid(True, alpha=0.3, axis='x')
    
    ax2 = axes[1]
    for i, (idx, typ) in enumerate(zip(top_indices, top_types)):
        v_trace = history['v'][0, :, idx]  # (time,)
        ax2.plot(history['t'], v_trace, label=f"N{idx}: {typ}", 
                color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Voltage')
    ax2.set_title('Time Courses of Top Neurons')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

    print("\n" + "="*60)
    print(f"SUMMARY: Top {top_k} Responding Neurons" + (" (excluding Tm1)" if exclude_input else ""))
    print("="*60)
    for i, (idx, typ, resp) in enumerate(zip(top_indices, top_types, top_responses)):
        print(f"{i+1:2d}. Neuron {idx:4d} ({typ:8s}): {resp:.4f}")
    print("="*60)
    
    print("\nResponses by cell type:")
    unique_types = np.unique(neuron_types)
    for cell_type in unique_types:
        if exclude_input and cell_type == 'Tm1':
            continue
        mask = neuron_types == cell_type
        type_responses = final_responses[mask]
        active = np.sum(type_responses > 0.1)  # Threshold for "active"
        print(f"  {cell_type:8s}: {active:4d}/{len(type_responses):4d} active "
              f"(max: {type_responses.max():.3f}, mean: {type_responses.mean():.3f})")


if __name__ == "__main__":
    print("Loading model and creating stimulus...")
    
    # Create model
    model = DrosophilaOpticLobeCircuit(
        neuron_types, source_indices, target_indices, weights
    )
    
    batch_size = 1
    n_tm1 = model.n_per_type['Tm1']
    angle = 90

    stimulus = model.stimulus_generator.create_bar(
        angle=angle,  
        offset=0.0, 
        width=2.0, 
        length=10.0, 
        intensity=1.0
    )
    
    print(f"Stimulus activates {(stimulus > 0).sum()} / {len(stimulus)} Tm1 neurons")
    

    tm1_input = model.stimulus_generator.to_torch(stimulus)
    fig, ax, scatter = model.stimulus_generator.visualize_bar(tm1_input, title='Tm1 Stimulus')
    plt.show()
    
    if tm1_input.shape[1] < n_tm1:
        padding = torch.zeros(1, n_tm1 - tm1_input.shape[1])
        tm1_input = torch.cat([tm1_input, padding], dim=1)

    print("\nRunning simulation...")
    with torch.no_grad():
        v_final, history = model(tm1_input, steps=5000, return_history=True)
    
    print(f"Simulation complete. Final shape: {v_final.shape}")
    
    visualize_responses(v_final, history, neuron_types, top_k=10, exclude_input=True)

