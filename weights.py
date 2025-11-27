from juliacall import Main as jl
import numpy as np
import scipy.sparse as sp
import pickle

jl.include("weights.jl")

I_jl, J_jl, V_jl = jl.findnz(jl.weights_sparse)

I = np.array(I_jl, dtype=np.int64) - 1
J = np.array(J_jl, dtype=np.int64) - 1
V = np.array(V_jl, dtype=np.int64)

nrows = int(jl.size(jl.weights_sparse, 1))
ncols = int(jl.size(jl.weights_sparse, 2))
coo = sp.coo_matrix((V, (I, J)), shape=(nrows, ncols))


row_ids = np.array(jl.cells, dtype=np.int64)
col_ids = np.array(jl.cells, dtype=np.int64)

src_ids = row_ids[I]
tgt_ids = col_ids[J]
edges = np.column_stack([src_ids, tgt_ids, V])
positive_edges = edges[edges[:, 2] > 0]
negative_edges = edges[edges[:, 2] < 0]

A = coo.tocsr() 

# excitatory part: positive entries
A_pos = A.multiply(A > 0)

# inhibitory part: negative entries (take magnitude)
A_neg = (-A).multiply(A < 0)

# column sums (1 × ncols)
colE = np.array(A_pos.sum(axis=0)).ravel()
colI = np.array(A_neg.sum(axis=0)).ravel()

# avoid divide-by-zero
colE[colE == 0] = 1.0
colI[colI == 0] = 1.0

# diagonal inverse-sum matrices
D_Einv = sp.diags(1.0 / colE)
D_Iinv = sp.diags(1.0 / colI)

# normalize columns
A_pos_norm = A_pos @ D_Einv
A_neg_norm = A_neg @ D_Iinv

W_norm = A_pos_norm - A_neg_norm

negative_count = W_norm < 0
neuron_types = np.array(list(jl.row_types))

data_to_save = {
    'W_norm': W_norm,
    'neuron_types': neuron_types
}
with open('connectome_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

