
import torch
import torch.nn as nn
import numpy as np
class DrosophilaOpticLobeCircuit(nn.Module):

    def __init__(self, neuron_types, source_indices, target_indices, weights, 
                 dt=0.1, tau_init=1.0, device='cpu', remove_reciprocal=False,
                 vrest_init=0.0, tau_by_type=None, vrest_by_type=None,
                 scale_by_connection_type=None):
        super().__init__()
        self.device = device
        self.dt = dt
        self._source_indices = source_indices.to(device)
        self._target_indices = target_indices.to(device)
        self._weights = weights.to(device)

        self.neuron_types = neuron_types
        self.n_neurons = len(neuron_types)
        self.n_edges = len(weights)
        self.type_names = ['Tm1', 'Dm3p', 'Dm3q', 'Dm3v', 'TmY4', 'TmY9q', 'TmY9q⊥']
        self.type_masks = {}

        # Create masks for each type
        for type_name in self.type_names:
            mask = (self.neuron_types == type_name)
            mask_tensor = torch.tensor(mask.tolist(), dtype=torch.bool, device=self.device)
            setattr(self, f'mask_{type_name}', mask_tensor)
            self.type_masks[type_name] = mask_tensor
        self.n_per_type = {name: self.type_masks[name].sum().item() 
                           for name in self.type_names}
        
        # Fixed parameters per neuron type 
        # tau_by_type: dict mapping type_name -> tau value, or None to use tau_init for all
        # vrest_by_type: dict mapping type_name -> vrest value, or None to use vrest_init for all
        if tau_by_type is None:
            tau_by_type = {}
        if vrest_by_type is None:
            vrest_by_type = {}
        
        self.tau_by_type = {
            name: float(tau_by_type.get(name, tau_init))
            for name in self.type_names
        }
        
        self.vrest_by_type = {
            name: float(vrest_by_type.get(name, vrest_init))
            for name in self.type_names
        }

        # Scale factors per type-to-type connection
        # scale_by_connection_type: dict mapping (source_type, target_type) -> scale value
        if scale_by_connection_type is None:
            scale_by_connection_type = {}
        
        # Initialize scale_by_connection_type with defaults (1.0) for all type pairs
        self.scale_by_connection_type = {}
        for source_type in self.type_names:
            for target_type in self.type_names:
                key = (source_type, target_type)
                self.scale_by_connection_type[key] = float(
                    scale_by_connection_type.get(key, 1.0)
                )
        #print(self.scale_by_connection_type)
        
        # Pre-compute source and target neuron types for each edge (for efficient scaling in forward)
        source_types_array = neuron_types[source_indices.cpu().numpy()]
        target_types_array = neuron_types[target_indices.cpu().numpy()]
        
        self._source_type_indices = torch.tensor(
            [self.type_names.index(typ) if typ in self.type_names else 0 
             for typ in source_types_array],
            device=self.device, dtype=torch.long
        )
        
        self._target_type_indices = torch.tensor(
            [self.type_names.index(typ) if typ in self.type_names else 0 
             for typ in target_types_array],
            device=self.device, dtype=torch.long
        )
        
        # Pre-compute connection type keys for each edge
        self._connection_type_keys = [
            (source_types_array[i], target_types_array[i])
            for i in range(len(source_types_array))
        ]
        
    def get_tau_vector(self):
        """Get tau vector with values per neuron type."""
        tau = torch.zeros(self.n_neurons, device=self.device)
        for type_name in self.type_names:
            tau[self.type_masks[type_name]] = self.tau_by_type[type_name]
        return tau

    def get_vrest_vector(self):
        """Get resting voltage vector Vrest_i with values per neuron type."""
        vrest = torch.zeros(self.n_neurons, device=self.device)
        for type_name in self.type_names:
            vrest[self.type_masks[type_name]] = self.vrest_by_type[type_name]
        return vrest
    
    def target_sum(self, edge_values: torch.Tensor) -> torch.Tensor:
        """
        Scatter sum operation (borrowed from flyvis).
        edge_values: [batch, n_edges]
        returns:     [batch, n_neurons]
        """
        result = torch.zeros(
            edge_values.shape[0], self.n_neurons, 
            device=self.device, dtype=edge_values.dtype
        )
        result.scatter_add_(
            1, 
            self._target_indices.expand(edge_values.shape[0], -1),
            edge_values
        )
        return result

    def activation(self, v):
        """
        Activation function f(V_j):
            r = ReLU(v)
        """
        return torch.relu(v)

    def forward(self, tm1_input, v_init=None, steps=100, return_history=False):
        if v_init is None:
            v = torch.zeros(1, self.n_neurons, device=self.device)
        else:
            v = v_init.clone()
        
        tm1_mask = self.type_masks['Tm1']
        n_tm1 = self.n_per_type['Tm1']
        assert tm1_input.shape[1] == n_tm1
        
        tau = self.get_tau_vector()
        vrest = self.get_vrest_vector()
        
        # Scale synapses by type-to-type connection
        edge_scales = torch.tensor(
            [self.scale_by_connection_type[key] for key in self._connection_type_keys],
            device=self.device, dtype=torch.float32
        )
        scaled_weights = self._weights * edge_scales
        
        if return_history:
            history = {'v': [v.clone()], 't': [0]}
        
        for step in range(steps):
            v[:, tm1_mask] = tm1_input
            
            # Activation function: f(V_j) = ReLU(V_j)
            r = self.activation(v)
            
            # Synaptic input: s_ij = w_ij * f(V_j)
            source_activities = r[:, self._source_indices]      
            edge_currents = source_activities * scaled_weights  
            synaptic_input = self.target_sum(edge_currents)     
            
            # External input e_i (zero for non-Tm1 neurons, tm1_input for Tm1)
            external_input = torch.zeros_like(v)
            external_input[:, tm1_mask] = tm1_input
            
            # Dynamics: τ_i * dV_i/dt = -V_i + Σ_j s_ij + Vrest_i + e_i
            # Rearranged: dV_i/dt = (-V_i + Σ_j s_ij + Vrest_i + e_i) / τ_i
            dv = (-v + synaptic_input + vrest.unsqueeze(0) + external_input) / tau.unsqueeze(0)
            dv[:, tm1_mask] = 0 
            
            v = v + self.dt * dv
            
            if return_history:
                history['v'].append(v.clone())
                history['t'].append((step + 1) * self.dt)
        
        if return_history:
            history['v'] = torch.stack(history['v'], dim=1)
            history['t'] = torch.tensor(history['t'])
            return v, history
        return v
    
    def steady_state(self, t_pre, dt, value=0.5):
        """
        Compute steady state after constant input (from flyvis).
        """
        if t_pre is None or t_pre <= 0:
            return None
        
        n_steps = int(t_pre / dt)
        n_tm1 = self.n_per_type['Tm1']
        
        tm1_input = torch.ones(1, n_tm1, device=self.device) * value
        
        with torch.no_grad():
            v = None
            for _ in range(n_steps):
                v = self.forward(tm1_input, v_init=v, steps=1, return_history=False)
        
        return v
    
