import load_weights as lw
import numpy as np
from collections import defaultdict
from scipy.signal import convolve2d
from utils import compute_response_metric

filter = np.array([[1, 1, 0],
                    [1, 1.1, 1],
                    [0, 1, 1]])

def _normalize_type_filter(types):
    if types is None:
        return None
    if isinstance(types, str):
        return {types}
    return set(types)

def _type_mask(include_types=None, exclude_types=None):
    include_types = _normalize_type_filter(include_types)
    exclude_types = _normalize_type_filter(exclude_types)

    mask = np.ones(lw.neuron_types.shape[0], dtype=bool)
    if include_types:
        mask &= np.isin(lw.neuron_types, list(include_types))
    if exclude_types:
        mask &= ~np.isin(lw.neuron_types, list(exclude_types))
    return mask

def get_postsynaptic_targets(
    neuron_index,
    use_raw=False,
    include_types=None,
    exclude_types=None,
    min_abs_synapses=0.0,
):
    """
    Return all target cells that the given neuron synapses to.

    Parameters
    ----------
    neuron_index : int
        Source neuron index.
    include_types : str | iterable[str] | None
        If provided, only return targets whose type is in this set.
    exclude_types : str | iterable[str] | None
        If provided, remove targets whose type is in this set.
    min_abs_synapses : float
        Minimum absolute synapse count to include.
    """
    if neuron_index < 0 or neuron_index >= lw.neuron_types.shape[0]:
        raise IndexError(f"neuron_index {neuron_index} out of bounds")

    if use_raw:
        row = lw.W_raw[neuron_index, :].toarray().ravel()
    else:
        row = lw.W_norm_diag[neuron_index, :].toarray().ravel()
    connectivity_mask = np.abs(row) > float(min_abs_synapses)
    final_mask = connectivity_mask & _type_mask(include_types, exclude_types)

    target_indices = np.where(final_mask)[0]
    synapse_counts = row[target_indices]
    sort_idx = np.argsort(np.abs(synapse_counts))[::-1]

    target_indices = target_indices[sort_idx]
    synapse_counts = synapse_counts[sort_idx]
    target_cell_ids = lw.row_ids[target_indices]
    target_types = lw.neuron_types[target_indices]

    return target_indices, target_cell_ids, target_types, synapse_counts


def get_presynaptic_inputs(
    neuron_index,
    use_raw=False,
    include_types=None,
    exclude_types=None,
    min_abs_synapses=0.0,
):
    """
    Return all source cells that synapse onto the given neuron.

    Parameters mirror `get_postsynaptic_targets`.
    """
    if neuron_index < 0 or neuron_index >= lw.neuron_types.shape[0]:
        raise IndexError(f"neuron_index {neuron_index} out of bounds")

    if use_raw:
        col = lw.W_raw[:, neuron_index].toarray().ravel()
    else:
        col = lw.W_norm_diag[:, neuron_index].toarray().ravel()
    connectivity_mask = np.abs(col) > float(min_abs_synapses)
    final_mask = connectivity_mask & _type_mask(include_types, exclude_types)

    source_indices = np.where(final_mask)[0]
    synapse_counts = col[source_indices]
    sort_idx = np.argsort(np.abs(synapse_counts))[::-1]

    source_indices = source_indices[sort_idx]
    synapse_counts = synapse_counts[sort_idx]
    source_cell_ids = lw.row_ids[source_indices]
    source_types = lw.neuron_types[source_indices]

    return source_indices, source_cell_ids, source_types, synapse_counts



def extract_tm1_inputs(neuron_index):
    """
    Return all Tm1 source cells that synapse onto the given neuron.
    """
    connections_to_neuron = lw.W_raw[:, neuron_index]
    connections_array = connections_to_neuron.toarray().flatten()
    tm1_mask = lw.neuron_types == 'Tm1'
    tm1_indices = np.where(tm1_mask)[0]
    
    tm1_input_indices = []
    synapse_counts = []
    tm1_input_data = []
    
    for tm1_idx in tm1_indices:
        synapse_count = connections_array[tm1_idx]
        if synapse_count > 0:
            tm1_input_indices.append(tm1_idx)
            synapse_counts.append(synapse_count)
            cell_id = lw.row_ids[tm1_idx]
            coord_row = lw.tm1_coords[lw.tm1_coords[:, 0] == cell_id]
            if len(coord_row) > 0:
                p, q = int(coord_row[0, 1]), int(coord_row[0, 2])
                tm1_input_data.append((p, q, synapse_count))
    
    tm1_input_indices = np.array(tm1_input_indices) if len(tm1_input_indices) > 0 else np.array([], dtype=np.int64)
    tm1_input_cell_ids = lw.row_ids[tm1_input_indices] if len(tm1_input_indices) > 0 else np.array([])
    
    return tm1_input_indices, tm1_input_cell_ids, synapse_counts, tm1_input_data


def get_receptive_field(neuron_index, normalize=True):
    """
    Return the 2D receptive field of the given neuron, normalized by the maximum synapse count.
    """
    _, _, _, tm1_input_data = extract_tm1_inputs(neuron_index)
    
    if len(tm1_input_data) == 0:
        return None, None, None, None, None
    
    p_coords = [item[0] for item in tm1_input_data]
    q_coords = [item[1] for item in tm1_input_data]
    p_min, p_max = min(p_coords), max(p_coords)
    q_min, q_max = min(q_coords), max(q_coords)
    

    p_size = p_max - p_min + 1
    q_size = q_max - q_min + 1
    rf_2d = np.zeros((p_size, q_size), dtype=np.float64)
    
    for p, q, synapse_count in tm1_input_data:
        p_idx = p - p_min
        q_idx = q - q_min
        rf_2d[p_idx, q_idx] = float(synapse_count)
    
    if normalize and rf_2d.max() > 0:
        rf_2d = rf_2d / rf_2d.max()
    
    p_range = (p_min, p_max)
    q_range = (q_min, q_max)
    p_offset = p_min
    q_offset = q_min
    
    return rf_2d, p_offset, q_offset, p_range, q_range


def get_rf_center(neuron_index, filt=filter, centroid=True):
    """                   
    Find the center of a neuron's receptive field.

    Parameters
    ----------
    neuron_index : int
        Index of the neuron.
    filt : np.ndarray
        Convolution filter used when centroid=False (the default).
    centroid : bool, default False
        If False (default), locate the peak of the RF convolved with `filt`.
        If True, return the intensity-weighted centroid of the raw RF.

    Returns
    -------
    center_p, center_q : int or None
        Grid coordinates of the RF centre, or (None, None) if the RF is empty.
    """
    rf_2d, p_offset, q_offset, _, _ = get_receptive_field(neuron_index, normalize=False)
    if rf_2d is None:
        return None, None

    if centroid:
        im = np.asarray(rf_2d, dtype=float)
        total = im.sum()
        if total == 0:
            return None, None
        rows = np.arange(im.shape[0], dtype=float)
        cols = np.arange(im.shape[1], dtype=float)
        row_c = float(np.dot(rows, im.sum(axis=1)) / total)
        col_c = float(np.dot(cols, im.sum(axis=0)) / total)
        center_p = int(round(row_c)) + p_offset
        center_q = int(round(col_c)) + q_offset
    else:
        convolved = convolve2d(rf_2d, filt, mode='same')
        max_idx = np.unravel_index(np.argmax(convolved), convolved.shape)
        center_p = max_idx[0] + p_offset
        center_q = max_idx[1] + q_offset

    return center_p, center_q

def compute_cell_centers(types = ['Dm3v', 'Dm3p', 'Dm3q', 'TmY4', 'TmY9q', 'TmY9q⊥']):
    cell_centers = {}
    for neuron_index in range(lw.neuron_types.size):
        if lw.neuron_types[neuron_index] not in types:
            continue
        center_p, center_q = get_rf_center(neuron_index)
        cell_centers[neuron_index] = (center_p, center_q)
    return cell_centers

def find_optimal_trial_centers(
    target_n=50,
    types=None,
    use_all=False,
    verbose=True,
):
    """
    Find the minimum set of (p, q) bar-centre coordinates needed to collect
    tuning-curve data for `target_n` cells of each cell type.

    Each bar trial centred at (p, q) captures every neuron whose RF centre
    (from get_center_conv) falls on that same grid coordinate, so one
    placement contributes data for multiple neurons—and potentially multiple
    types—simultaneously.

    Algorithm: greedy weighted set cover.  At each step the candidate centre
    that satisfies the most unmet quota across all types is selected.

    Parameters
    ----------
    target_n : int
        Target number of cells per type to accumulate.  Capped at the number
        of cells that actually have a valid RF centre.  Ignored when
        use_all=True.
    types : list[str] | None
        Cell types to cover.  Defaults to the standard six types.
    use_all : bool
        If True, accumulate data for every cell that has a valid RF centre.
    verbose : bool
        Print a human-readable summary of the solution.

    Returns
    -------
    selected_centers : list[tuple[int, int]]
        Ordered list of (p, q) bar-centre coordinates chosen by the algorithm.
    per_center_coverage : dict[tuple[int,int], dict[str, list[int]]]
        For each selected centre, the neuron indices (by type) that are
        *credited* to that centre (i.e. counted towards the quota).
    type_coverage : dict[str, list[int]]
        Flat list of covered neuron indices per type across all centres.
    """
    if types is None:
        types = ['Dm3v', 'Dm3p', 'Dm3q', 'TmY4', 'TmY9q', 'TmY9q\u22a5']
    types_set = set(types)

    # Compute RF centres for all neurons of the target types
    cell_centers = compute_cell_centers(types=types)  # {idx: (p, q)}

    # Build (p, q) -> {type -> set of neuron indices}
    center_type_map = defaultdict(lambda: defaultdict(set))
    for idx, (p, q) in cell_centers.items():
        if p is None or q is None:
            continue
        ntype = lw.neuron_types[idx]
        if ntype in types_set:
            center_type_map[(int(p), int(q))][ntype].add(idx)

    # Count available neurons per type across all valid centres
    available = defaultdict(int)
    for tmap in center_type_map.values():
        for t, idxs in tmap.items():
            available[t] += len(idxs)

    # Set per-type quotas
    if use_all:
        quota = {t: available.get(t, 0) for t in types}
    else:
        quota = {t: min(target_n, available.get(t, 0)) for t in types}

    remaining = {t: quota[t] for t in types}
    covered = {t: set() for t in types}
    selected_centers = []
    per_center_coverage = {}
    candidates = set(center_type_map.keys())

    while candidates and any(r > 0 for r in remaining.values()):
        best_center = None
        best_score = 0

        for center in candidates:
            score = 0
            for t, idxs in center_type_map[center].items():
                if t in remaining and remaining[t] > 0:
                    new_count = len(idxs - covered[t])
                    score += min(new_count, remaining[t])
            if score > best_score:
                best_score = score
                best_center = center

        if best_center is None or best_score == 0:
            if verbose:
                print("Warning: cannot satisfy remaining quotas with available centres.")
            break

        # Credit this centre for the newly covered neurons
        credited = defaultdict(list)
        for t, idxs in center_type_map[best_center].items():
            if t in remaining and remaining[t] > 0:
                for idx in idxs - covered[t]:
                    if remaining[t] > 0:
                        credited[t].append(idx)
                        covered[t].add(idx)
                        remaining[t] -= 1
        per_center_coverage[best_center] = {t: sorted(v) for t, v in credited.items()}

        selected_centers.append(best_center)
        candidates.discard(best_center)

    if verbose:
        n_angles = 12  # default sweep: 0–165° in steps of 15°
        n_sims = len(selected_centers) * n_angles
        print("\n=== Optimal Trial Centres (greedy set-cover) ===")
        print(f"Types  : {types}")
        print(f"Quota  : { {t: quota[t] for t in types} }")
        print(
            f"\nSelected {len(selected_centers)} centre(s) "
            f"-> {n_sims} network simulations "
            f"({n_angles} angles × {len(selected_centers)} centres)\n"
        )
        header = f"{'p':>4}  {'q':>4}  " + "  ".join(f"{t:>10}" for t in types)
        print(header)
        print("-" * len(header))
        for center in selected_centers:
            row = f"{center[0]:>4}  {center[1]:>4}  "
            row += "  ".join(
                f"{len(per_center_coverage[center].get(t, [])):>10}" for t in types
            )
            print(row)
        print("\nFinal coverage:")
        for t in types:
            q = quota[t]
            n = len(covered[t])
            pct = 100.0 * n / q if q > 0 else 0.0
            print(f"  {t:<12}: {n:>4} / {q:>4}  ({pct:.0f}%)")

    type_coverage = {t: sorted(covered[t]) for t in types}
    return selected_centers, per_center_coverage, type_coverage


def top_k_neurons(
    neuron_type,
    partner_type,
    k=10,
    direction="inputs",
    ranking="raw",
):
    """
    Find the top-k neurons of `neuron_type` ranked by their connectivity
    to/from neurons of `cell_type`.

    Parameters
    ----------
    neuron_type : str
        The cell type whose neurons are ranked (e.g. "Dm3p").
    cell_type : str
        The partner cell type to count connections with (e.g. "Tm1").
    k : int
        Number of top neurons to return.
    direction : {"inputs", "outputs"}
        "inputs"  – rank neurons of `neuron_type` by total input received
                    FROM `cell_type` (i.e. W[cell_type -> neuron_type]).
        "outputs" – rank neurons of `neuron_type` by total output sent
                    TO `cell_type` (i.e. W[neuron_type -> cell_type]).
    ranking : {"raw", "normalized", "unique"}
        "raw"        – sum of absolute raw synapse counts (W_raw).
        "normalized" – sum of absolute diagonal-normalized weights (W_norm_diag).
        "unique"     – number of unique partner neurons connected (non-zero entries).

    Returns
    -------
    indices : np.ndarray, shape (<=k,)
        Global neuron indices (into lw.neuron_types) sorted descending by score.
    cell_ids : np.ndarray, shape (<=k,)
        Connectome cell IDs of the top-k neurons.
    scores : np.ndarray, shape (<=k,)
        Score value for each returned neuron.
    """
    neuron_indices = np.where(lw.neuron_types == neuron_type)[0]
    partner_indices = np.where(lw.neuron_types == partner_type)[0]

    if len(neuron_indices) == 0:
        raise ValueError(f"No neurons found of type '{neuron_type}'")
    if len(partner_indices) == 0:
        raise ValueError(f"No neurons found of type '{partner_type}'")

    if ranking == "raw":
        W = lw.W_raw
    elif ranking == "normalized":
        W = lw.W_norm_diag
    elif ranking == "unique":
        W = lw.W_raw
    else:
        raise ValueError(f"ranking must be 'raw', 'normalized', or 'unique'; got '{ranking}'")

    if direction == "inputs":
        # sub[i, j] = weight from partner i to neuron_type neuron j
        sub = W[partner_indices, :][:, neuron_indices]
        axis = 0  # sum/count over partner rows → one score per neuron column
    elif direction == "outputs":
        # sub[i, j] = weight from neuron_type neuron i to partner j
        sub = W[neuron_indices, :][:, partner_indices]
        axis = 1  # sum/count over partner columns → one score per neuron row
    else:
        raise ValueError(f"direction must be 'inputs' or 'outputs'; got '{direction}'")

    if ranking == "unique":
        scores = np.array((sub != 0).sum(axis=axis)).ravel().astype(float)
    else:
        scores = np.array(np.abs(sub).sum(axis=axis)).ravel()

    k = min(k, len(neuron_indices))
    top_local = np.argsort(scores)[::-1][:k]
    top_global = neuron_indices[top_local]

    return top_global, lw.row_ids[top_global], scores[top_local]


def extract_neuron_response(
    v_final,
    history,
    neuron_index,
    metric="steady",
    steady_window=0.0,
    last_n=10,
    use_relu=True,
):
    """
    Extract a single-neuron response from model outputs.

    metric:
        "steady"  -> mean over the last steady_window seconds or last_n steps
        "peak"    -> max over time
        "mean"    -> mean over the full time course
        "final"   -> value at the final time step (or from v_final if history missing)
    """
    history = history or {}
    if "v" in history and history["v"] is not None:
        metric_values = compute_response_metric(
            values=history["v"],
            t=history.get("t"),
            metric=metric,
            use_relu=use_relu,
            steady_window=steady_window,
            last_n=last_n,
        )
        return float(metric_values[neuron_index])

    fallback_values = compute_response_metric(
        values=None,
        final_values=v_final,
        metric="final",
        use_relu=use_relu,
    )
    return float(fallback_values[neuron_index])

