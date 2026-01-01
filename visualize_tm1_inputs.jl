

using OpticLobe
using Luxor
using ColorSchemes

function visualize_tm1_inputs(cell_id::Integer; 
                              output_file::Union{String, Nothing}=nothing, 
                              hexelsize::Int=6,
                              colormap=ColorSchemes.hot)
    
    println("Visualizing Tm1 inputs to cell ID: $cell_id")
    input_map = preimage("Tm1", cell_id)
    max_synapses = maximum(input_map)
    total_synapses = sum(input_map)
    num_inputs = count(input_map .> 0)
    println("  Maximum synapses at a single location: $max_synapses")
    println("  Total synapses from Tm1: $total_synapses")
    println("  Number of Tm1 cells providing input: $num_inputs")

    Drawing(1200, 1200, output_file)
    origin()
    heatmap = eyeheat(input_map; cmap=colormap)
    
    rect2hex(heatmap, hexelsize=hexelsize)
    
    finish()
    preview()
    return input_map
end

