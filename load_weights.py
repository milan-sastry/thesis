import pickle
import numpy as np
import torch

_loaded_data = None

def _load_data():
    global _loaded_data
    if _loaded_data is not None:
        return _loaded_data
    with open('./connectome_data/connectome_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    W_norm = data['W_norm']
    W_raw = data['W_raw']
    neuron_types = data['neuron_types']
    tm1_coords = data['tm1_coords']
    row_ids = data['row_ids']
    
    W_coo = W_norm.tocoo()
    W_raw_coo = W_raw.tocoo()
    
    source_indices_raw = torch.tensor(
        np.array(W_raw_coo.row, dtype=np.int64, copy=True).tolist(), 
        dtype=torch.long
    )
    target_indices_raw = torch.tensor(
        np.array(W_raw_coo.col, dtype=np.int64, copy=True).tolist(), 
        dtype=torch.long
    )
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
    
    print(f"Loaded {len(weights)} connections between {len(neuron_types)} neurons")

    _loaded_data = {
        'W_norm': W_norm,
        'W_raw': W_raw,
        'neuron_types': neuron_types,
        'tm1_coords': tm1_coords,
        'row_ids': row_ids,
        'source_indices_raw': source_indices_raw,
        'target_indices_raw': target_indices_raw,
        'weights_raw': weights_raw,
        'source_indices': source_indices,
        'target_indices': target_indices,
        'weights': weights,
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
