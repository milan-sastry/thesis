import torch
import torch.nn as nn
import numpy as np
from scipy import sparse

class DrosophilaOpticLobeCircuit(nn.Module):
    """
    Non-spiking rate-based model of Dm3-TmY circuit from Drosophila optic lobe.
    Uses connectome-derived weights (no learning).
    Optimized for sparse connectivity matrices.
    """
    
    def __init__(self, 
                 W_connectome_sparse,  # scipy sparse matrix (n_neurons, n_neurons)
                 neuron_types,          # list/array of type labels for each neuron
                 dt=0.1,               # Time step (ms)
                 tau_init=20.0,        # Initial time constant (ms)
                 device='cpu'):
        
        super().__init__()
        
        self.device = device
        self.dt = dt
        
        # Convert sparse matrix to PyTorch sparse tensor
        W_coo = W_connectome_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([W_coo.row, W_coo.col]))
        values = torch.FloatTensor(W_coo.data)
        shape = W_coo.shape
        
        self.W = torch.sparse_coo_tensor(indices, values, shape, device=device)
        
        # Store neuron type information
        self.neuron_types = np.array(neuron_types)
        self.n_neurons = len(neuron_types)
        
        # Create masks/indices for each cell type
        self.type_names = ['Tm1', 'Dm3p', 'Dm3q', 'Dm3v', 'TmY4', 'TmY9q', 'TmY9q_perp']
        self.type_indices = {}
        self.type_masks = {}
        
        for type_name in self.type_names:
            mask = (self.neuron_types == type_name)
            self.type_indices[type_name] = np.where(mask)[0]
            self.type_masks[type_name] = torch.tensor(mask, dtype=torch.bool, device=device)
        
        # Count neurons per type
        self.n_per_type = {name: len(self.type_indices[name]) for name in self.type_names}
        
        # Learnable parameters: time constants (one per cell type)
        self.tau_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(tau_init))
            for name in self.type_names
        })
        
        # Biases (one per neuron)
        self.bias = nn.Parameter(torch.zeros(self.n_neurons, device=device))
        
        # Global weight scaling factor
        self.weight_scale = nn.Parameter(torch.tensor(1.0))
        
    def get_tau_vector(self):
        """Create vector of time constants for all neurons."""
        tau = torch.zeros(self.n_neurons, device=self.device)
        for type_name in self.type_names:
            tau[self.type_masks[type_name]] = self.tau_params[type_name]
        return tau
    
    def activation(self, x):
        """Activation function: ReLU (non-negative firing rates)."""
        return torch.relu(x)
    
    def forward(self, tm1_input, v_init=None, steps=100, return_history=False):
        """
        Simulate circuit dynamics.
        
        Args:
            tm1_input: (batch, n_tm1) - External input to Tm1 cells
            v_init: (batch, n_neurons) - Initial state (if None, starts at 0)
            steps: Number of time steps to simulate
            return_history: Whether to return full trajectory
            
        Returns:
            v_final: (batch, n_neurons) - Final state
            history: (optional) Dictionary with trajectory
        """
        
        batch_size = tm1_input.shape[0]
        
        # Initialize state
        if v_init is None:
            v = torch.zeros(batch_size, self.n_neurons, device=self.device)
        else:
            v = v_init.clone()
        
        # Set Tm1 input (clamp Tm1 cells to input values)
        tm1_mask = self.type_masks['Tm1']
        n_tm1 = self.n_per_type['Tm1']
        assert tm1_input.shape[1] == n_tm1, f"Expected {n_tm1} Tm1 inputs, got {tm1_input.shape[1]}"
        
        # Get parameters
        tau = self.get_tau_vector()  # (n_neurons,)
        
        # Storage for trajectories (optional)
        if return_history:
            history = {'v': [v.clone()], 't': [0]}
        
        for step in range(steps):
            # Clamp Tm1 to external input
            v[:, tm1_mask] = tm1_input
            
            # Current activations (firing rates)
            r = self.activation(v)  # (batch, n_neurons)
            
            # Compute synaptic input via sparse matrix multiplication
            # For batch processing: loop over batch
            synaptic_input = torch.zeros_like(v)
            for b in range(batch_size):
                # Sparse @ dense = dense
                # W is (n_neurons, n_neurons), r[b] is (n_neurons,)
                synaptic_input[b] = torch.sparse.mm(
                    self.W, 
                    r[b].unsqueeze(1)
                ).squeeze(1)
            
            # Apply global weight scaling
            synaptic_input = synaptic_input * self.weight_scale
            
            # Total input
            total_input = synaptic_input + self.bias
            
            # Dynamics: dv/dt = (-v + total_input) / tau
            # Don't update Tm1 (input layer)
            dv = (-v + total_input) / tau
            dv[:, tm1_mask] = 0  # Clamp Tm1
            
            # Euler integration
            v = v + self.dt * dv
            
            # Store history
            if return_history:
                history['v'].append(v.clone())
                history['t'].append((step + 1) * self.dt)
        
        if return_history:
            # Stack history into tensors
            history['v'] = torch.stack(history['v'], dim=1)  # (batch, time, n_neurons)
            history['t'] = torch.tensor(history['t'])
            return v, history
        else:
            return v


# ============================================================================
# Faster version using dense matrix multiplication
# ============================================================================

class DrosophilaOpticLobeCircuitFast(DrosophilaOpticLobeCircuit):
    """
    Faster version that converts sparse to dense for batch processing.
    Use this if you have enough memory and batch_size > 1.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert to dense for faster batch matrix multiplication
        self.W_dense = self.W.to_dense()
    
    def forward(self, tm1_input, v_init=None, steps=100, return_history=False):
        """Optimized forward pass using dense matrix ops."""
        
        batch_size = tm1_input.shape[0]
        
        # Initialize state
        if v_init is None:
            v = torch.zeros(batch_size, self.n_neurons, device=self.device)
        else:
            v = v_init.clone()
        
        # Set Tm1 input
        tm1_mask = self.type_masks['Tm1']
        n_tm1 = self.n_per_type['Tm1']
        assert tm1_input.shape[1] == n_tm1
        
        # Get parameters
        tau = self.get_tau_vector()  # (n_neurons,)
        
        # Storage for trajectories
        if return_history:
            history = {'v': [v.clone()], 't': [0]}
        
        for step in range(steps):
            # Clamp Tm1 to external input
            v[:, tm1_mask] = tm1_input
            
            # Current activations
            r = self.activation(v)  # (batch, n_neurons)
            
            # Synaptic input: (batch, n_neurons) @ (n_neurons, n_neurons)^T
            synaptic_input = r @ self.W_dense.T  # (batch, n_neurons)
            
            # Apply global weight scaling
            synaptic_input = synaptic_input * self.weight_scale
            
            # Total input
            total_input = synaptic_input + self.bias
            
            # Dynamics
            dv = (-v + total_input) / tau
            dv[:, tm1_mask] = 0  # Clamp Tm1
            
            # Euler integration
            v = v + self.dt * dv
            
            if return_history:
                history['v'].append(v.clone())
                history['t'].append((step + 1) * self.dt)
        
        if return_history:
            history['v'] = torch.stack(history['v'], dim=1)
            history['t'] = torch.tensor(history['t'])
            return v, history
        else:
            return v


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    from scipy.sparse import csr_matrix
    
    # Create dummy data matching your structure
    n_neurons = 2451
    density = 0.01
    
    # Random connectivity with both positive and negative values
    W_connectome = sparse.random(n_neurons, n_neurons, density=density, format='csr')
    W_connectome.data = W_connectome.data * 2 - 1  # Scale to [-1, 1]
    
    # Example neuron type labels
    neuron_types = np.array(['Tm1'] * 745 + 
                           ['Dm3p'] * 498 + 
                           ['Dm3q'] * 454 + 
                           ['Dm3v'] * 321 +
                           ['TmY4'] * 211 + 
                           ['TmY9q'] * 172 + 
                           ['TmY9q_perp'] * 100)
    # Pad to 2451
    neuron_types = np.concatenate([neuron_types, ['other'] * (n_neurons - len(neuron_types))])
    
    # Initialize model
    model = DrosophilaOpticLobeCircuitFast(
        W_connectome, 
        neuron_types,
        dt=0.1,
        tau_init=20.0,
        device='cpu'
    )
    
    # Create input
    batch_size = 5
    n_tm1 = model.n_per_type['Tm1']
    tm1_input = torch.randn(batch_size, n_tm1) * 0.5 + 1.0
    
    # Run simulation
    v_final, history = model(tm1_input, steps=200, return_history=True)
    
    print(f"Connectivity: {W_connectome.nnz} synapses / {n_neurons**2} possible")
    print(f"Sparsity: {100 * W_connectome.nnz / n_neurons**2:.2f}%")
    print(f"\nNeurons per type:")
    for name in model.type_names:
        print(f"  {name}: {model.n_per_type[name]}")
    
    print(f"\nFinal activity statistics:")
    print(f"  Mean: {v_final.mean().item():.3f}")
    print(f"  Std:  {v_final.std().item():.3f}")
    print(f"  Max:  {v_final.max().item():.3f}")
    print(f"  Min:  {v_final.min().item():.3f}")
    
    # Access activity of specific cell types over time
    dm3v_activity = history['v'][:, :, model.type_masks['Dm3v']]  # (batch, time, n_dm3v)
    print(f"\nDm3v activity over time:")
    print(f"  Initial: {dm3v_activity[:, 0, :].mean().item():.3f}")
    print(f"  Final:   {dm3v_activity[:, -1, :].mean().item():.3f}")
    
    # Plot example (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot activity of different cell types over time
        for type_name in ['Dm3p', 'Dm3q', 'Dm3v', 'TmY4', 'TmY9q', 'TmY9q_perp']:
            mask = model.type_masks[type_name]
            activity = history['v'][0, :, mask].mean(dim=1).cpu().numpy()  # Average over neurons
            axes[0].plot(history['t'].cpu().numpy(), activity, label=type_name)
        
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Mean Activity')
        axes[0].set_title('Population Activity Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot individual neurons from one type
        dm3v_neurons = history['v'][0, :, model.type_masks['Dm3v']][:, :10]  # First 10 neurons
        for i in range(dm3v_neurons.shape[1]):
            axes[1].plot(history['t'].cpu().numpy(), dm3v_neurons[:, i].cpu().numpy(), 
                        alpha=0.5, linewidth=0.5)
        
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Activity')
        axes[1].set_title('Individual Dm3v Neurons')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('circuit_dynamics.png', dpi=150)
        print("\nPlot saved to circuit_dynamics.png")
        
    except ImportError:
        print("\nMatplotlib not available for plotting")