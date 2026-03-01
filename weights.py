from juliacall import Main as jl
import numpy as np
import scipy.sparse as sp
import pickle

jl.include("weights.jl")
type_names = ['Tm1', 'Dm3p', 'Dm3q', 'Dm3v', 'TmY4', 'TmY9q', 'TmY9q⊥']

I_jl, J_jl, V_jl = jl.findnz(jl.weights_sparse)

full_input_total = np.array(jl.full_input_total)

I = np.array(I_jl, dtype=np.int64) - 1 # row indices
J = np.array(J_jl, dtype=np.int64) - 1 # column indices
V = np.array(V_jl, dtype=np.int64) # weights

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
W_raw = A
neuron_types = np.array(list(jl.row_types))


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

def normalize_weights_full(A, full_input_total):
    full_input_total = np.asarray(full_input_total, dtype=np.float64).ravel()
    if full_input_total.size != A.shape[1]:
        raise ValueError(
            f"full_input_total length ({full_input_total.size}) "
            f"does not match number of columns in A ({A.shape[1]})"
        )

    col_total = full_input_total.copy()
    col_total[col_total == 0] = 1.0
    D_inv = sp.diags(1.0 / col_total)
    return A @ D_inv


W_norm_diag = normalize_weights_diagonal(W_raw)
W_norm_full = normalize_weights_full(W_raw, full_input_total)

tm1_coords = np.array([(int(coord[0]), coord[1][0], coord[1][1]) 
                       for coord in jl.tm1_coords])
data_to_save = {
    'W_raw': W_raw,
    'W_norm_diag': W_norm_diag,
    'W_norm_full': W_norm_full,
    'neuron_types': neuron_types,
    'tm1_coords': tm1_coords,
    'row_ids': row_ids,
}
with open('./connectome_data/connectome_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

