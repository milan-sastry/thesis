from operator import index
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from load_weights import row_ids, neuron_types, tm1_coords, W_raw
from stimulus import StimulusGenerator
import analysis as an
import matplotlib.patches as patches
from utils import to_numpy, compute_response_metric, pq_to_xy
from tuning_curves import reconstruct_f1
import load_weights as lw

try:
    from juliacall import Main as jl
except ImportError:
    jl = None

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
    sources = to_numpy(source_indices)
    targets = to_numpy(target_indices)
    w = to_numpy(weights)
    
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

def visualize_tm1_inputs_julia(neuron_index):
    """
    Python wrapper for the Julia visualize_tm1_inputs function. Slow and bad.
    """
    if jl is None:
        raise ImportError("juliacall is not installed; Julia-based visualization is unavailable.")
    jl.include("visualize_tm1_inputs.jl")
    output_file = f"./tm1_inputs/tm1_inputs_{neuron_index}.png"
    cell_id = int(row_ids[neuron_index])
    jl.visualize_tm1_inputs(cell_id, output_file=output_file, hexelsize=6)

def visualize_tm1_inputs(neuron_index, rect=None, ax=None, show=True):
    """
    Visualize all Tm1 input cells to a given neuron, colored by number of synapses. Better than julia wrapper.
    """
    tm1_input_indices, tm1_input_cell_ids, synapse_counts, _ = an.extract_tm1_inputs(neuron_index)
    
    if len(tm1_input_indices) == 0:
        print(f"No Tm1 inputs found for neuron {neuron_index}")
        return
    
    stimulus_generator = StimulusGenerator(
        tm1_coords,
        neuron_types,
        row_ids
    )
    
    stimulus = np.zeros(stimulus_generator.n_cells)
    
    for tm1_idx, cell_id, synapse_count in zip(tm1_input_indices, tm1_input_cell_ids, synapse_counts):
        if cell_id in stimulus_generator.cell_ids:
            cell_idx = stimulus_generator.cell_ids.index(cell_id)
            stimulus[cell_idx] = float(synapse_count)
    
    if stimulus.max() > 0:
        stimulus_normalized = stimulus / stimulus.max()
    else:
        stimulus_normalized = stimulus
    
    target_cell_id = int(row_ids[neuron_index])
    target_type = neuron_types[neuron_index]
    min_synapses = int(min(synapse_counts)) if synapse_counts else 0
    max_synapses = int(max(synapse_counts)) if synapse_counts else 0
    title = f"Tm1 inputs to neuron {neuron_index} (cell_id: {target_cell_id}, type: {target_type})\nSynapse counts: {min_synapses} - {max_synapses}"
    fig, ax, scatter = stimulus_generator.visualize_bar(stimulus_normalized, title=title, ax=ax)
    if rect is not None:
        ax.add_patch(copy.copy(rect))
    cbar = fig.colorbar(scatter, ax=ax)
    if max_synapses > 0:
        tick_positions = np.linspace(0, 1, 6)
        tick_labels = [f"{int(pos * max_synapses)}" for pos in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Number of Synapses', rotation=270, labelpad=20)
    
    if show:
        plt.show()

    return fig, ax


def visualize_stimulus_with_tm1_inputs(
    neuron_index,
    stimulus,
    rect=None,
    ax=None,
    show=True,
    title=None,
    stimulus_cmap="Greys_r",
    tm1_input_cmap="viridis",

    base_edge_color="#B0B0B0",
    base_edge_width=0.5,
    input_edge_width_range=(1.0, 3.5),
    annotate_pq=False,
    add_colorbar=True,
    stimulus_norm=None,
    synapse_norm=None,
    clip_stimulus_zero=False,
):
    """
    Overlay a stimulus map and TM1 inputs to neuron_index on the same hex lattice.

    Face color: stimulus value
    Hex outline: TM1 input strength (synapse count)
    """
    _, tm1_input_cell_ids, synapse_counts, _ = an.extract_tm1_inputs(neuron_index)

    stimulus_generator = StimulusGenerator(tm1_coords, neuron_types, row_ids)
    stimulus_np = np.ravel(to_numpy(stimulus))
    if stimulus_np.shape[0] != stimulus_generator.n_cells:
        raise ValueError(f"stimulus must have length {stimulus_generator.n_cells}, got {stimulus_np.shape[0]}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    coords = np.array([stimulus_generator.tm1_coords[cid] for cid in stimulus_generator.cell_ids_with_coords])
    p_coords = coords[:, 0]
    q_coords = coords[:, 1]
    x, y = pq_to_xy(p_coords, q_coords)

    # Fast lookup from cell_id -> stimulus vector index.
    cell_id_to_idx = {int(cid): i for i, cid in enumerate(stimulus_generator.cell_ids)}
    stimulus_values = np.array(
        [float(stimulus_np[cell_id_to_idx[int(cid)]]) for cid in stimulus_generator.cell_ids_with_coords],
        dtype=float,
    )

    if clip_stimulus_zero:
        stimulus_values = np.maximum(stimulus_values, 0.0)
    if stimulus_norm is None:
        vmin = 0.0 if clip_stimulus_zero else float(np.min(stimulus_values))
        vmax = float(np.max(stimulus_values))
    else:
        vmin = max(float(stimulus_norm[0]), 0.0) if clip_stimulus_zero else float(stimulus_norm[0])
        vmax = float(stimulus_norm[1])
    if np.isclose(vmin, vmax):
        norm = mcolors.Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    stim_mapper = plt.cm.ScalarMappable(norm=norm, cmap=stimulus_cmap)

    synapse_map = {int(cid): float(syn) for cid, syn in zip(tm1_input_cell_ids, synapse_counts)}
    if synapse_norm is None:
        if synapse_map:
            syn_values = np.array(list(synapse_map.values()), dtype=float)
            syn_min = float(np.min(syn_values))
            syn_max = float(np.max(syn_values))
        else:
            syn_min = 0.0
            syn_max = 0.0
    else:
        syn_min, syn_max = float(synapse_norm[0]), float(synapse_norm[1])

    syn_span = max(syn_max - syn_min, 1e-6)
    lw_min, lw_max = input_edge_width_range
    lw_span = max(lw_max - lw_min, 0.0)
    syn_mapper = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=syn_min, vmax=syn_max if syn_max > syn_min else syn_min + 1e-6),
        cmap=tm1_input_cmap,
    )

    hex_radius = 0.58
    for cid, cx, cy, stim_val, p_val, q_val in zip(
        stimulus_generator.cell_ids_with_coords, x, y, stimulus_values, p_coords, q_coords
    ):
        face_color = stim_mapper.to_rgba(stim_val)
        edge_color = base_edge_color
        line_width = base_edge_width
        syn = synapse_map.get(int(cid))
        if syn is not None:
            edge_color = syn_mapper.to_rgba(syn)
            line_width = lw_min + ((syn - syn_min) / syn_span) * lw_span

        hex_patch = patches.RegularPolygon(
            (float(cx), float(cy)),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(30),
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=float(line_width),
            zorder=2,
        )
        ax.add_patch(hex_patch)

        if annotate_pq:
            ax.text(
                float(cx),
                float(cy),
                f"({int(p_val)},{int(q_val)}) : {stim_val}",
                ha="center",
                va="center",
                fontsize=4,
                color="black",
                zorder=3,
            )

    if rect is not None:
        ax.add_patch(copy.copy(rect))

    if title is None:
        target_cell_id = int(row_ids[neuron_index])
        target_type = neuron_types[neuron_index]
        title = (
            f"Stimulus + TM1 inputs to neuron {neuron_index}\n"
            f"(cell_id: {target_cell_id}, type: {target_type})"
        )

    ax.set_xlim(float(np.min(x) - 1.5), float(np.max(x) + 1.5))
    ax.set_ylim(float(np.min(y) - 1.5), float(np.max(y) + 1.5))
    ax.set_aspect("equal")
    ax.set_xlabel("Posterior (h)")
    ax.set_ylabel("Dorsal (v)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", color="grey", linewidth=0.5, alpha=0.5)

    if add_colorbar:
        cbar = fig.colorbar(stim_mapper, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Stimulus value", fontsize=11)

    if show:
        plt.show()

    return fig, ax


def save_tm1_inputs_grid(
    neuron_indices,
    rect=None,
    output_file="./tm1_inputs/tm1_inputs_grid.png",
    ncols=3,
    stimulus=None,
    stimulus_by_neuron=None,
    overlay=False,
    overlay_kwargs=None,
):
    """
    Plot a list of neuron maps into one figure and save it.

    Default mode:
      - TM1 input maps only.

    Overlay mode:
      - stimulus + TM1 input overlay maps.
      - Triggered by overlay=True or by providing stimulus/stimulus_by_neuron.
    """
    if len(neuron_indices) == 0:
        raise ValueError("neuron_indices must contain at least one neuron index")

    neuron_indices = list(neuron_indices)
    nrows = int(np.ceil(len(neuron_indices) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 7 * nrows))
    axes = np.array(axes, ndmin=1).flatten()
    overlay_kwargs = overlay_kwargs or {}
    use_overlay = bool(overlay or stimulus is not None or stimulus_by_neuron is not None)

    neuron_stimuli = []
    if use_overlay:
        synapse_global_min = None
        synapse_global_max = None
        for idx, neuron_index in enumerate(neuron_indices):
            neuron_stimulus = stimulus
            if stimulus_by_neuron is not None:
                if callable(stimulus_by_neuron):
                    neuron_stimulus = stimulus_by_neuron(neuron_index)
                elif isinstance(stimulus_by_neuron, dict):
                    neuron_stimulus = stimulus_by_neuron.get(neuron_index)
                else:
                    neuron_stimulus = stimulus_by_neuron[idx]
            if neuron_stimulus is None:
                raise ValueError(
                    f"No stimulus provided for neuron {neuron_index}. "
                    "Provide stimulus or stimulus_by_neuron."
                )
            neuron_stimulus_np = to_numpy(neuron_stimulus)
            neuron_stimuli.append(np.ravel(neuron_stimulus_np))
            _, _, neuron_synapse_counts, _ = an.extract_tm1_inputs(neuron_index)
            if neuron_synapse_counts:
                local_min = float(min(neuron_synapse_counts))
                local_max = float(max(neuron_synapse_counts))
                synapse_global_min = (
                    local_min if synapse_global_min is None else min(synapse_global_min, local_min)
                )
                synapse_global_max = (
                    local_max if synapse_global_max is None else max(synapse_global_max, local_max)
                )
        all_vals = np.concatenate(neuron_stimuli) if neuron_stimuli else np.array([0.0])
        global_vmin = 0.0 if overlay_kwargs.get("clip_stimulus_zero") else float(np.min(all_vals))
        global_norm = (global_vmin, float(np.max(all_vals)))
        if synapse_global_min is None or synapse_global_max is None:
            synapse_global_norm = (0.0, 1.0)
        else:
            synapse_global_norm = (synapse_global_min, synapse_global_max)
    else:
        global_norm = None
        synapse_global_norm = None

    for idx, (ax, neuron_index) in enumerate(zip(axes, neuron_indices)):
        if use_overlay:
            visualize_stimulus_with_tm1_inputs(
                neuron_index=neuron_index,
                stimulus=neuron_stimuli[idx],
                rect=rect,
                ax=ax,
                show=False,
                add_colorbar=False,
                stimulus_norm=global_norm,
                synapse_norm=synapse_global_norm,
                **overlay_kwargs,
            )
        else:
            visualize_tm1_inputs(neuron_index, rect=rect, ax=ax, show=False)

    for ax in axes[len(neuron_indices):]:
        ax.axis("off")

    if use_overlay:
        cmap_name = overlay_kwargs.get("stimulus_cmap", "Greys_r")
        stim_mapper = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=global_norm[0], vmax=global_norm[1] if global_norm[1] > global_norm[0] else global_norm[0] + 1e-6),
            cmap=cmap_name,
        )
        # Reserve explicit axes for colorbars to prevent label overlap.
        fig.subplots_adjust(left=0.10, right=0.88)
        stim_cax = fig.add_axes([0.03, 0.15, 0.020, 0.72])
        syn_cax = fig.add_axes([0.93, 0.15, 0.020, 0.72])

        stim_cbar = fig.colorbar(stim_mapper, cax=stim_cax)
        stim_cbar.set_label("Stimulus value", rotation=270, labelpad=14)
        stim_cbar.ax.yaxis.set_label_position("left")
        stim_cbar.ax.yaxis.set_ticks_position("left")
        stim_cbar.ax.yaxis.set_label_coords(-2.4, 0.5)

        syn_mapper = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(
                vmin=synapse_global_norm[0],
                vmax=(
                    synapse_global_norm[1]
                    if synapse_global_norm[1] > synapse_global_norm[0]
                    else synapse_global_norm[0] + 1e-6
                ),
            ),
            cmap=overlay_kwargs.get("tm1_input_cmap", "viridis"),
        )
        syn_cbar = fig.colorbar(syn_mapper, cax=syn_cax)
        syn_cbar.set_label("TM1 input synapses", rotation=270, labelpad=18)
        syn_cbar.ax.yaxis.set_label_position("right")
        syn_cbar.ax.yaxis.set_label_coords(3.0, 0.5)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=300)
    if use_overlay:
        print(f"Saved combined stimulus+TM1 input grid to {output_file}")
    else:
        print(f"Saved combined TM1 input plot to {output_file}")
    return fig, axes



def _compute_response_scores(
    v_final,
    history,
    metric="steady",
    steady_window=0.0,
    last_n=10,
    use_relu=True,
):
    history = history or {}
    if "v" not in history or history["v"] is None:
        scores = compute_response_metric(
            values=None,
            final_values=v_final,
            metric="final",
            use_relu=use_relu,
        )
        return scores, None, None

    components = compute_response_metric(
        values=history["v"],
        t=history.get("t"),
        metric=metric,
        use_relu=use_relu,
        steady_window=steady_window,
        last_n=last_n,
        return_components=True,
    )
    scores = components["scores"]
    values_time_major = components["values"]
    t_hist = components["t"]

    # Keep the previous return shape expected by plotting code: (1, T, N).
    r_history = values_time_major[None, :, :]
    return scores, r_history, t_hist


def _select_top_indices(scores, mask, top_k):
    valid_indices = np.where(mask)[0]
    if valid_indices.size == 0:
        return np.array([], dtype=np.int64)
    ranked = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    return ranked[:top_k]


def visualize_responses(
    v_final,
    history,
    top_k=10,
    metric="steady",
    steady_window=0.0,
    last_n=10,
    use_relu=True,
    types=None,
    neuron_indices=None,
    exclude_types=("Tm1",),
    title=None,
    show_f1_fit=False,
    temporal_freq=None,
    grating_onset_t=None,
    f1_baseline_window=None,
    f1_analysis_window=None,
):
    v_final = to_numpy(v_final)
    history = history or {}
    if "v" in history:
        history["v"] = to_numpy(history["v"])
    if "t" in history:
        history["t"] = to_numpy(history["t"])

    scores, r_history, t_hist = _compute_response_scores(
        v_final,
        history,
        metric=metric,
        steady_window=steady_window,
        last_n=last_n,
        use_relu=use_relu,
    )

    exclude_mask = np.isin(neuron_types, list(exclude_types))
    include_mask = ~exclude_mask

    selected_indices = None
    if neuron_indices is not None:
        selected_indices = np.asarray(neuron_indices, dtype=np.int64).ravel()
        selected_indices = np.array(list(dict.fromkeys(selected_indices.tolist())), dtype=np.int64)
        valid_mask = (selected_indices >= 0) & (selected_indices < len(neuron_types))
        selected_indices = selected_indices[valid_mask]
        if exclude_types:
            selected_indices = selected_indices[~np.isin(neuron_types[selected_indices], list(exclude_types))]
        if selected_indices.size == 0:
            return

    if selected_indices is None:
        if types is None:
            types_to_plot = [t for t in np.unique(neuron_types) if t not in exclude_types]
        else:
            types_to_plot = [t for t in types if t not in exclude_types]

        top_indices_by_type = {}
        for cell_type in types_to_plot:
            mask = (neuron_types == cell_type) & include_mask
            top_indices_by_type[cell_type] = _select_top_indices(scores, mask, top_k)
    else:
        types_to_plot = [t for t in np.unique(neuron_types[selected_indices])]
        top_indices_by_type = {
            cell_type: selected_indices[neuron_types[selected_indices] == cell_type]
            for cell_type in types_to_plot
        }

    if not types_to_plot:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(types_to_plot) / n_cols))
    fig_height = max(3 * n_rows, 4)
    fig_width = 14
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.array(axes, ndmin=2).reshape(n_rows, n_cols)

    for plot_idx, cell_type in enumerate(types_to_plot):
        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        type_indices = top_indices_by_type[cell_type]
        type_labels = neuron_types[type_indices] if type_indices.size else np.array([])
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(type_indices))))

        ax_tc = axes[row_idx, col_idx]
        if r_history is None or t_hist is None:
            ax_tc.text(0.5, 0.5, "No time course history", ha="center", va="center")
            ax_tc.set_xticks([])
            ax_tc.set_yticks([])
            continue

        if type_indices.size == 0:
            ax_tc.text(0.5, 0.5, "No neurons", ha="center", va="center")
            ax_tc.set_xticks([])
            ax_tc.set_yticks([])
            continue

        if show_f1_fit and temporal_freq is None:
            raise ValueError("`temporal_freq` must be provided when show_f1_fit=True.")
        if show_f1_fit and grating_onset_t is None:
            raise ValueError("`grating_onset_t` must be provided when show_f1_fit=True.")

        for i, (idx, typ) in enumerate(zip(type_indices, type_labels)):
            ax_tc.plot(
                t_hist,
                r_history[0, :, idx],
                label=f"N{idx}: {typ}",
                color=colors[i],
                linewidth=2,
            )
            if show_f1_fit:
                recon = reconstruct_f1(
                    r_history[0, :, idx],
                    t_hist,
                    temporal_freq=temporal_freq,
                    grating_onset_t=grating_onset_t,
                    baseline_window=f1_baseline_window,
                    analysis_window=f1_analysis_window,
                )
                ax_tc.plot(
                    t_hist,
                    recon,
                    color=colors[i],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.8,
                    label=f"N{idx} F1 fit",
                )

        ax_tc.set_xlabel("Time")
        ax_tc.set_ylabel("Response r = relu(v)" if use_relu else "Response v")
        if title:
            ax_tc.set_title(f"Time Courses - {cell_type}, {title}")
        else:
            ax_tc.set_title(f"Time Courses - {cell_type}")
        ax_tc.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax_tc.grid(True, alpha=0.3)

    for empty_idx in range(len(types_to_plot), n_rows * n_cols):
        row_idx = empty_idx // n_cols
        col_idx = empty_idx % n_cols
        axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_tm1_inputs(2574)









