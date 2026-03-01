import pickle
import numpy as np
import torch
import scipy.sparse as sp

_loaded_data = None

def _load_data():
    global _loaded_data
    if _loaded_data is not None:
        return _loaded_data
    with open('./connectome_data/connectome_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    W_norm_diag = data['W_norm_diag']
    W_norm_full = data['W_norm_full']
    W_raw = data['W_raw']
    neuron_types = data['neuron_types']
    tm1_coords = data['tm1_coords']
    row_ids = data['row_ids']
    full_input_total = data['full_input_total']
    
    W_coo = W_norm_diag.tocoo()
    W_coo_full = W_norm_full.tocoo()
    W_raw_coo = W_raw.tocoo()
    
    weights_raw = torch.tensor(
        np.array(W_raw_coo.data, dtype=np.float64, copy=True).tolist(), 
        dtype=torch.float32
    )
    
    source_indices = torch.tensor(
        np.array(W_coo.row, dtype=np.int64, copy=True).tolist(), 
        dtype=torch.long
    )
    target_indices = torch.tensor(
        np.array(W_coo.col, dtype=np.int64, copy=True).tolist(), 
        dtype=torch.long
    )
    weights = torch.tensor(
        np.array(W_coo.data, dtype=np.float64, copy=True).tolist(), 
        dtype=torch.float32
    )
    weights_norm_full = torch.tensor(
        np.array(W_coo_full.data, dtype=np.float64, copy=True).tolist(), 
        dtype=torch.float32
    )
    
    print(f"Loaded {len(weights)} connections between {len(neuron_types)} neurons")

    _loaded_data = {
        'W_norm_diag': W_norm_diag,
        'W_norm_full': W_norm_full,
        'W_raw': W_raw,
        'neuron_types': neuron_types,
        'tm1_coords': tm1_coords,
        'row_ids': row_ids,
        'full_input_total': full_input_total,
        'weights_raw': weights_raw,
        'source_indices': source_indices,
        'target_indices': target_indices,
        'weights': weights,
        'weights_norm_full': weights_norm_full,
    }
    
    # Set all attributes on the module itself so future accesses bypass __getattr__
    import sys
    this_module = sys.modules[__name__]
    for key, value in _loaded_data.items():
        setattr(this_module, key, value)
    
    return _loaded_data

def __getattr__(name):
    """Lazy loading of module attributes. Only loads data when first accessed."""
    data = _load_data()
    if name in data:
        return data[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def scale_weights_by_connection_type(A, scale_by_connection_type, neuron_types):
    """
    Scale raw synapse counts by (source_type, target_type) prior to normalization.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Raw signed synapse count matrix (rows=source neurons, cols=target neurons).
    scale_by_connection_type : dict
        Dict mapping (source_type, target_type) -> scale value, matching the
        structure used by `scale_by_connection_type` in `network.py`.
        Missing keys default to 1.0.

    Returns
    -------
    scipy.sparse.csr_matrix
        Scaled sparse matrix with the same sparsity pattern as `A`.
    """
    if scale_by_connection_type is None:
        return A.tocsr()
    if not isinstance(scale_by_connection_type, dict):
        raise TypeError("scale_by_connection_type must be a dict keyed by (source_type, target_type)")
    if A.shape[0] != neuron_types.size or A.shape[1] != neuron_types.size:
        raise ValueError(
            f"A shape {A.shape} does not match neuron_types length {neuron_types.size}"
        )

    A_coo = A.tocoo(copy=True)
    source_types = neuron_types[A_coo.row]
    target_types = neuron_types[A_coo.col]

    scales = np.empty(A_coo.data.shape[0], dtype=np.float64)
    for i, (src_t, tgt_t) in enumerate(zip(source_types, target_types)):
        key = (src_t, tgt_t)
        scale = float(scale_by_connection_type.get(key, 1.0))
        if scale < 0:
            raise ValueError(
                f"Negative scale {scale} for connection type {key} would flip synapse sign. "
                "Use non-negative scales."
            )
        scales[i] = scale

    A_coo.data = A_coo.data.astype(np.float64, copy=False) * scales
    return A_coo.tocsr()
    
def normalize_weights_diagonal(A):
    A_pos = A.multiply(A > 0)
    A_neg = (-A).multiply(A < 0)

    colE = np.array(A_pos.sum(axis=0)).ravel()
    colI = np.array(A_neg.sum(axis=0)).ravel()
    colE[colE == 0] = 1.0
    colI[colI == 0] = 1.0

    D_Einv = sp.diags(1.0 / colE)
    D_Iinv = sp.diags(1.0 / colI)

    A_pos_norm = A_pos @ D_Einv
    A_neg_norm = A_neg @ D_Iinv

    return A_pos_norm - A_neg_norm