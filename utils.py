

def remove_reciprocal_connections(source_indices, target_indices, weights, neuron_types):
    """
    Filter edges to only keep those with Tm1 as the source type.
    """
    sources = source_indices.cpu().numpy()
    keep_mask = neuron_types[sources] == 'Tm1'
    
    filtered_sources = source_indices[keep_mask]
    filtered_targets = target_indices[keep_mask]
    filtered_weights = weights[keep_mask]
    
    n_removed = len(sources) - len(filtered_sources)
    print(f"Filtered to keep only Tm1 source connections. "
          f"Removed {n_removed} connections. "
          f"Remaining: {len(filtered_sources)} connections")
    
    return filtered_sources, filtered_targets, filtered_weights



