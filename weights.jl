using OpticLobe
using SparseArrays
using NamedArrays  

types = ["Tm1", "Dm3v", "Dm3p", "Dm3q", "TmY4", "TmY9q", "TmY9q⊥"]
excitatory_types = ["Tm1", "TmY4", "TmY9q", "TmY9q⊥"]
inhibitory_types = ["Dm3v", "Dm3p", "Dm3q"]

cells = reduce(vcat, type2ids.(types))


row_names = Name.(cells)
col_names = row_names  

weights = W[row_names, col_names]
weights_sparse = parent(weights)  
row_types = ind2type[id2ind.(cells)]   
tm1_coords = []
full_input_total = []
for (idx, (row, name)) in enumerate(zip(cells, row_names))

    type = ind2type[id2ind(row)]
   # print(typeof(name))
    push!(full_input_total, sum(W[:, name]))
    if type in inhibitory_types
        # println("Inhibitory type: ", type)
        weights[name, :] .= -abs.(weights[name, :])
      
    end
    if type == "Tm1"
        try
            push!(tm1_coords, (row,id2pq[row]))
        catch e
            println("No coordinates ", row, ": ", e)
        end 
    end
end
weights_sparse = parent(weights)    




