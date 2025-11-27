using OpticLobe
using SparseArrays
using NamedArrays  # optional, but harmless

types = ["Tm1", "Dm3v", "Dm3p", "Dm3q", "TmY4", "TmYq", "TmY9q⊥"]
excitatory_types = ["Tm1", "TmY4", "TmYq", "TmY9q⊥"]
inhibitory_types = ["Dm3v", "Dm3p", "Dm3q"]

cells = reduce(vcat, type2ids.(types))

# # These are the long numeric cell IDs, in the SAME order used for indexing.
# row_ids = Int64.(cells)
# col_ids = row_ids  # same ordering for square submatrix

row_names = Name.(cells)
col_names = row_names  

weights = W[row_names, col_names]
weights_sparse = parent(weights)  
row_types = ind2type[id2ind.(cells)]   

for (row, name) in zip(cells, row_names)
    type = ind2type[id2ind(row)]
    # print(typeof(name))
    if type in inhibitory_types
        # println("Inhibitory type: ", type)
        weights[name, :] .= -abs.(weights[name, :])
    end
end
weights_sparse = parent(weights)     


