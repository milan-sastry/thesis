from juliacall import Main as jl
import torch
import numpy as np
import matplotlib.pyplot as plt
from load_weights import row_ids, neuron_types

def visualize_type_to_type_connections(source_indices, target_indices, weights, neuron_types,
                                       aggregation='sum', figsize=(10, 8), cmap='viridis'):
    """
    Visualize connections as a type-to-type connection matrix heatmap.
    
    Args:   
        source_indices: torch.Tensor of source neuron indices
        target_indices: torch.Tensor of target neuron indices
        weights: torch.Tensor of connection weights. For synapse counting, use raw weights
                 (weights_raw) where each value represents the number of synapses.
        neuron_types: numpy array of neuron types (indexed by neuron index)
        aggregation: str, how to aggregate connections between types. 
                     Options: 
                     - 'count' (number of connections/edges between neuron pairs)
                     - 'synapses' (total number of synapses, sums weight values; requires raw weights)
                     - 'sum' (sum of normalized weights)
                     - 'mean' (mean of weights per connection)
        figsize: tuple, figure size for the plot
        cmap: str, colormap name for the heatmap
        
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    sources = source_indices.cpu().numpy() if torch.is_tensor(source_indices) else source_indices
    targets = target_indices.cpu().numpy() if torch.is_tensor(target_indices) else target_indices
    w = weights.cpu().numpy() if torch.is_tensor(weights) else weights
    
    unique_types = np.unique(neuron_types)
    unique_types = sorted(unique_types) 
    
    type_to_idx = {type_name: i for i, type_name in enumerate(unique_types)}
    n_types = len(unique_types)
    
    if aggregation in ['count', 'synapses']:
        connection_matrix = np.zeros((n_types, n_types), dtype=np.int64)
    else:
        connection_matrix = np.zeros((n_types, n_types), dtype=np.float64)
        if aggregation == 'mean':
            count_matrix = np.zeros((n_types, n_types), dtype=np.int64)
    
    for i in range(len(sources)):
        src_idx = sources[i]
        tgt_idx = targets[i]
        
        src_type = neuron_types[src_idx]
        tgt_type = neuron_types[tgt_idx]
        
        src_type_idx = type_to_idx[src_type]
        tgt_type_idx = type_to_idx[tgt_type]
        
        if aggregation == 'count':
            connection_matrix[src_type_idx, tgt_type_idx] += 1
        elif aggregation == 'synapses':
            connection_matrix[src_type_idx, tgt_type_idx] += int(w[i])
        elif aggregation == 'sum':
            connection_matrix[src_type_idx, tgt_type_idx] += w[i]
        elif aggregation == 'mean':
            connection_matrix[src_type_idx, tgt_type_idx] += w[i]
            count_matrix[src_type_idx, tgt_type_idx] += 1
    
    if aggregation == 'mean':
        connection_matrix = np.divide(connection_matrix, count_matrix, 
                                     out=np.zeros_like(connection_matrix), 
                                     where=count_matrix!=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(connection_matrix, cmap=cmap, aspect='auto')
    ax.set_xticks(np.arange(len(unique_types)))
    ax.set_yticks(np.arange(len(unique_types)))
    ax.set_xticklabels(unique_types)
    ax.set_yticklabels(unique_types)
        
    for i in range(n_types):
        for j in range(n_types):
            if aggregation in ['count', 'synapses']:
                text = ax.text(j, i, int(connection_matrix[i, j]),
                            ha="center", va="center", color="black", fontsize=8)
            else:
                text = ax.text(j, i, f'{connection_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
    label_suffix = ' Connections' if aggregation == 'count' else (' Synapses' if aggregation == 'synapses' else ' Weight')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(aggregation.capitalize() + label_suffix)
    
    ax.set_xlabel('Postsynaptic Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Presynaptic Type', fontsize=12, fontweight='bold')
    ax.set_title(f'Type-to-Type Connection Matrix ({aggregation.capitalize()})', 
                 fontsize=14, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    
    return fig, ax

def visualize_tm1_inputs(neuron_index):
    """
    Python wrapper for the Julia visualize_tm1_inputs function.
    """
    jl.include("visualize_tm1_inputs.jl")
    output_file = f"./tm1_inputs/tm1_inputs_{neuron_index}.png"
    cell_id = int(row_ids[neuron_index])
    jl.visualize_tm1_inputs(cell_id, output_file=output_file, hexelsize=6)


def visualize_responses(v_final, history, top_k=10, show_plot=True, return_text=False):
    if torch.is_tensor(v_final):
        v_final = v_final.cpu().numpy()
    if torch.is_tensor(history['v']):
        history['v'] = history['v'].cpu().numpy()
    if torch.is_tensor(history['t']):
        history['t'] = history['t'].cpu().numpy()

    final_v = v_final[0]
    r_final = np.maximum(final_v, 0.0)  

    r_history = np.maximum(history['v'], 0.0) 

    output_lines = []
    non_tm1_mask = neuron_types != 'Tm1'
    r_final_filtered = r_final.copy()
    r_final_filtered[~non_tm1_mask] = -np.inf
    text_output = f"\nExcluding {np.sum(~non_tm1_mask)} Tm1 input neurons from analysis"
    if return_text:
        output_lines.append(text_output)
    else:
        print(text_output)

    dm3_trio = ['Dm3p', 'Dm3q', 'Dm3v']
    tmy_trio = ['TmY4', 'TmY9q', 'TmY9q⊥']
    
    # Find top_k neurons in Dm3 trio
    dm3_mask = np.isin(neuron_types, dm3_trio)
    r_final_dm3 = r_final_filtered.copy()
    r_final_dm3[~dm3_mask] = -np.inf
    dm3_top_indices = np.argsort(r_final_dm3)[::-1][:top_k]
    dm3_top_responses = r_final[dm3_top_indices]
    dm3_top_types = neuron_types[dm3_top_indices]
    
    # Find top_k neurons in Tmy trio
    tmy_mask = np.isin(neuron_types, tmy_trio)
    r_final_tmy = r_final_filtered.copy()
    r_final_tmy[~tmy_mask] = -np.inf
    tmy_top_indices = np.argsort(r_final_tmy)[::-1][:top_k]
    tmy_top_responses = r_final[tmy_top_indices]
    tmy_top_types = neuron_types[tmy_top_indices]

    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Dm3 trio - bar plot
        ax1 = axes[0, 0]
        colors_dm3 = plt.cm.tab10(np.arange(top_k))
        ax1.barh(range(top_k), dm3_top_responses, color=colors_dm3, edgecolor='black')
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels([f"N{idx}: {typ}" for idx, typ in zip(dm3_top_indices, dm3_top_types)])
        ax1.set_xlabel('Final Response r = relu(v)')
        ax1.set_title(f'Top {top_k} Responding Neurons - Dm3 Trio')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # Dm3 trio - time courses
        ax2 = axes[0, 1]
        for i, (idx, typ) in enumerate(zip(dm3_top_indices, dm3_top_types)):
            ax2.plot(history['t'], r_history[0, :, idx], label=f"N{idx}: {typ}",
                    color=colors_dm3[i], linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Response r = relu(v)')
        ax2.set_title('Time Courses - Dm3 Trio')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Tmy trio - bar plot
        ax3 = axes[1, 0]
        colors_tmy = plt.cm.tab20(np.arange(top_k))
        ax3.barh(range(top_k), tmy_top_responses, color=colors_tmy, edgecolor='black')
        ax3.set_yticks(range(top_k))
        ax3.set_yticklabels([f"N{idx}: {typ}" for idx, typ in zip(tmy_top_indices, tmy_top_types)])
        ax3.set_xlabel('Final Response r = relu(v)')
        ax3.set_title(f'Top {top_k} Responding Neurons - Tmy Trio')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')

        # Tmy trio - time courses
        ax4 = axes[1, 1]
        for i, (idx, typ) in enumerate(zip(tmy_top_indices, tmy_top_types)):
            ax4.plot(history['t'], r_history[0, :, idx], label=f"N{idx}: {typ}",
                    color=colors_tmy[i], linewidth=2)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Response r = relu(v)')
        ax4.set_title('Time Courses - Tmy Trio')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    summary_text = ""
    summary_text += f"Top {top_k} Responding Neurons - Dm3 Trio" + "\n"
    for i, (idx, typ, resp) in enumerate(zip(dm3_top_indices, dm3_top_types, dm3_top_responses)):
        summary_text += f"{i+1:2d}. Neuron {idx:4d} ({typ:8s}): {resp:.4f}\n"


    summary_text += f"Top {top_k} Responding Neurons - Tmy Trio" + "\n"
    for i, (idx, typ, resp) in enumerate(zip(tmy_top_indices, tmy_top_types, tmy_top_responses)):
        summary_text += f"{i+1:2d}. Neuron {idx:4d} ({typ:8s}): {resp:.4f}\n"

    summary_text += "Responses by cell type:\n"
    for cell_type in np.unique(neuron_types):
        if cell_type == 'Tm1':
            continue
        mask = neuron_types == cell_type
        type_responses = r_final[mask]
        active = np.sum(type_responses > 0.0)
        summary_text += (
            f"  {cell_type:8s}: "
            f"{active:4d}/{len(type_responses):4d} "
            f"({active / len(type_responses) * 100:.2f}%) active "
            f"(max: {type_responses.max():.3f}, mean: {type_responses.mean():.3f})\n"
        )
    
    if return_text:
        return "\n".join(output_lines) + summary_text
    else:
        print(summary_text, end='')

if __name__ == "__main__":
    visualize_tm1_inputs(977)