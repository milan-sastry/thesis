import pickle
import numpy as np
import torch

with open('connectome_data.pkl', 'rb') as f:
    data = pickle.load(f)

W_norm = data['W_norm']
neuron_types = data['neuron_types']

W_coo = W_norm.tocoo()

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