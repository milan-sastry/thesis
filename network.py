from load_weights import W_norm, neuron_types, source_indices, target_indices, weights
import torch
import torch.nn as nn
import numpy as np
from scipy import sparse


class DrosophilaOpticLobeCircuit(nn.Module):

    
    def __init__(self, neuron_types, source_indices, target_indices, weights, 
             dt=0.1, tau_init=20.0, device='cpu'):

        super().__init__()
        print("init")
        self.device = device
        self.dt = dt
        
        self._source_indices = source_indices
        self._target_indices = target_indices
        self._weights = weights
        
        self.neuron_types = neuron_types
        self.n_neurons = len(neuron_types)
        self.n_edges = len(weights)
        
        # Type organization
        self.type_names = ['Tm1', 'Dm3p', 'Dm3q', 'Dm3v', 'TmY4', 'TmY9q', 'TmY9q⊥']
        self.type_masks = {}

        for type_name in self.type_names:
            print(type_name)
            mask = (self.neuron_types == type_name)
            # Store directly as attribute instead of buffer
            mask_tensor = torch.tensor(mask.tolist(), dtype=torch.bool, device=self.device)
            setattr(self, f'mask_{type_name}', mask_tensor)
            self.type_masks[type_name] = mask_tensor

        self.n_per_type = {name: self.type_masks[name].sum().item() 
                                for name in self.type_names}
        
        # Parameters
        self.tau_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(tau_init))
            for name in self.type_names
        })
        
        self.bias = nn.Parameter(torch.zeros(self.n_neurons))
        
        # Separate E/I scaling
        self.scale_excitatory = nn.Parameter(torch.tensor(1.0))
        self.scale_inhibitory = nn.Parameter(torch.tensor(1.0))
        
        self.clamp_config = {
            'scale_excitatory': 'non_negative',
            'scale_inhibitory': 'non_negative',
            # 'tau_params': (1.0, 100.0),
        }
    
    def clamp(self):
        """Clamp parameters to valid ranges (borrowed from flyvis)."""
        # Clamp scales to be non-negative
        if self.scale_excitatory.requires_grad:
            self.scale_excitatory.data.clamp_(0)
        if self.scale_inhibitory.requires_grad:
            self.scale_inhibitory.data.clamp_(0)
        
        # Optionally clamp tau values
        for tau_param in self.tau_params.values():
            if tau_param.requires_grad:
                tau_param.data.clamp_(1.0, 100.0)  # reasonable range
    
    def get_tau_vector(self):
        """Create vector of time constants."""
        tau = torch.zeros(self.n_neurons, device=self.device)
        for type_name in self.type_names:
            tau[self.type_masks[type_name]] = self.tau_params[type_name]
        return tau
    
    def target_sum(self, edge_values: torch.Tensor) -> torch.Tensor:
        """
        Scatter sum operation (borrowed from flyvis).
        Aggregates edge inputs to target neurons.
        
        Args:
            edge_values: (batch, n_edges) values at each edge
            
        Returns:
            (batch, n_neurons) aggregated values at each neuron
        """
        result = torch.zeros(
            edge_values.shape[0], self.n_neurons, 
            device=self.device, dtype=edge_values.dtype
        )
        result.scatter_add_(
            1,  # neuron dimension
            self._target_indices.expand(edge_values.shape[0], -1),
            edge_values
        )
        return result
    
    def activation(self, x):
        """Activation function."""
        return torch.relu(x)
    
    def forward(self, tm1_input, v_init=None, steps=100, return_history=False):
        """
        Forward pass with efficient scatter/gather ops.
        """
        self.clamp() 
        print("here")
        batch_size = tm1_input.shape[0]
        
        # Initialize state
        if v_init is None:
            v = torch.zeros(batch_size, self.n_neurons, device=self.device)
        else:
            v = v_init.clone()
        
        tm1_mask = self.type_masks['Tm1']
        n_tm1 = self.n_per_type['Tm1']
        assert tm1_input.shape[1] == n_tm1
        
        tau = self.get_tau_vector()
        
        # Pre-compute scaled weights for each edge
        edge_scales = torch.ones(self.n_edges, device=self.device)
        edge_scales[self._weights > 0] = self.scale_excitatory
        edge_scales[self._weights < 0] = self.scale_inhibitory
        scaled_weights = self._weights * edge_scales
        
        if return_history:
            history = {'v': [v.clone()], 't': [0]}
        
        for step in range(steps):
            # Clamp Tm1
            v[:, tm1_mask] = tm1_input
            
            # Get activations
            r = self.activation(v)  # (batch, n_neurons)
            
            # Gather source activities, multiply by weights, scatter to targets
            source_activities = r[:, self._source_indices]  # (batch, n_edges)
            edge_currents = source_activities * scaled_weights  # (batch, n_edges)
            synaptic_input = self.target_sum(edge_currents)  # (batch, n_neurons)
            
            # Total input
            total_input = synaptic_input + self.bias
            
            # Dynamics
            dv = (-v + total_input) / tau
            dv[:, tm1_mask] = 0
            
            # Integration
            v = v + self.dt * dv
            
            if return_history:
                history['v'].append(v.clone())
                history['t'].append((step + 1) * self.dt)
        
        if return_history:
            history['v'] = torch.stack(history['v'], dim=1)
            history['t'] = torch.tensor(history['t'])
            return v, history
        return v
    
    def steady_state(self, t_pre, dt, batch_size, value=0.5):
        """
        Compute steady state after constant input (like flyvis).
        """
        if t_pre is None or t_pre <= 0:
            return None
        
        n_steps = int(t_pre / dt)
        n_tm1 = self.n_per_type['Tm1']
        
        # Constant input to Tm1
        tm1_input = torch.ones(batch_size, n_tm1, device=self.device) * value
        
        # Run without storing history
        with torch.no_grad():
            v = None
            for _ in range(n_steps):
                v = self.forward(tm1_input, v_init=v, steps=1, return_history=False)
        
        return v
    
if __name__ == "__main__":
    W_connectome_sparse = W_norm
    print("here")
    model = DrosophilaOpticLobeCircuit(
        neuron_types, source_indices, target_indices, weights
    )
    print("made the model!")
    batch_size = 1
    n_tm1 = model.n_per_type['Tm1']
    tm1_input = torch.rand(1, n_tm1)

    output = model(tm1_input, steps=50)
    print(len(output))