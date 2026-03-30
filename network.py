
import torch
import torch.nn as nn
import numpy as np
from utils import remove_reciprocal_connections, to_numpy
class DrosophilaOpticLobeCircuit(nn.Module):

    def __init__(self, neuron_types, source_indices, target_indices, weights,
                 dt=0.1, tau_init=1.0, device='cpu', remove_reciprocal=False,
                 vrest_init=0.0, tau_by_type=None, vrest_by_type=None, default_scale=1.0,
                 scale_by_connection_type=None, tm1_tau_hp=12.3, tm1_tau_lp=2.3,
                 fwhm_cen=8.12, fwhm_sur=27.14, A_rel=0.04, tm1_coords=None, tm1_row_ids=None):
        super().__init__()
        self.device = device
        self.dt = dt
        self.tm1_tau_hp = tm1_tau_hp
        self.tm1_tau_lp = tm1_tau_lp
        self.tm1_sigma_c = (fwhm_cen / 5.5) / (2*np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma, roughly 5.0 degrees per ommatidia
        self.tm1_sigma_s = (fwhm_sur / 5.5) / (2*np.sqrt(2 * np.log(2)))
        self.A_rel = A_rel
        if remove_reciprocal:
            self._source_indices, self._target_indices, self._weights = remove_reciprocal_connections(
                source_indices, target_indices, weights, neuron_types
            )
        else:
            self._source_indices = source_indices.to(device)
            self._target_indices = target_indices.to(device)
            self._weights = weights.to(device)
        self.neuron_types = neuron_types
        self.n_neurons = len(neuron_types)
        self.n_edges = len(self._weights)
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
                    scale_by_connection_type.get(key, default_scale)
                )
        
        # Pre-compute source and target neuron types for each edge (for efficient scaling in forward)
        source_types_array = neuron_types[to_numpy(self._source_indices)]
        target_types_array = neuron_types[to_numpy(self._target_indices)]
        
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

        # Pre-compute spatial DoG filter matrix for Tm1 cells
        if tm1_coords is not None:
            raw = np.array(tm1_coords)  # preserve original dtype (int64 for cell IDs)
            n_tm1_local = self.n_per_type['Tm1']

            if raw.ndim == 2 and raw.shape[1] == 3:
                # Raw format: each row is (cell_id, p, q); need tm1_row_ids to map to Tm1-local indices
                if tm1_row_ids is None:
                    raise ValueError("tm1_row_ids must be provided when tm1_coords has 3 columns (cell_id, p, q)")
                # Keep cell IDs as int64 to avoid float32 precision loss on large IDs
                cell_ids = raw[:, 0].astype(np.int64)
                coord_dict = {int(cid): (float(raw[i, 1]), float(raw[i, 2]))
                              for i, cid in enumerate(cell_ids)}
                p_arr = np.array([coord_dict[int(cid)][0] if int(cid) in coord_dict else np.nan
                                  for cid in tm1_row_ids], dtype=np.float32)
                q_arr = np.array([coord_dict[int(cid)][1] if int(cid) in coord_dict else np.nan
                                  for cid in tm1_row_ids], dtype=np.float32)
            else:
                # Full format: [n_tm1, 2] with (p, q) for each cell in network order
                pq_arr = raw.astype(np.float32)
                p_arr, q_arr = pq_arr[:, 0], pq_arr[:, 1]

            # Only compute DoG for cells that have coordinates; missing cells get identity rows
            has_coord = ~np.isnan(p_arr)
            valid_idx = np.where(has_coord)[0]

            x_v = q_arr[valid_idx] - p_arr[valid_idx]
            y_v = (p_arr[valid_idx] + q_arr[valid_idx]) / np.sqrt(3)
            xy_v = np.stack([x_v, y_v], axis=1)  # [n_valid, 2]

            diff = xy_v[:, None, :] - xy_v[None, :, :]  # [n_valid, n_valid, 2]
            r2 = (diff ** 2).sum(axis=2)              # [n_valid, n_valid]
            H_sub = np.exp(-r2 / (2 * self.tm1_sigma_c ** 2)) - self.A_rel * np.exp(-r2 / (2 * self.tm1_sigma_s ** 2))

            # Embed sub-matrix into a full identity; missing cells pass through unchanged
            H = np.eye(n_tm1_local, dtype=np.float32)
            H[np.ix_(valid_idx, valid_idx)] = H_sub

            self.tm1_H = torch.tensor(H, dtype=torch.float32, device=device)
        else:
            self.tm1_H = None

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

    def forward(self, tm1_input, v_init=None, steps=None, return_history=False, steady_state_tol=1e-5):
        if v_init is None:
            v = torch.zeros(1, self.n_neurons, device=self.device)
        else:
            v = v_init.clone()
        
        tm1_mask = self.type_masks['Tm1']
        n_tm1 = self.n_per_type['Tm1']
        tm1_input = tm1_input.to(self.device)

        if tm1_input.ndim == 1:
            if steps is None:
                raise ValueError("steps must be provided for single-frame input")
            if steps <= 0:
                raise ValueError("steps must be a positive integer")
            tm1_input = tm1_input.unsqueeze(0).unsqueeze(0).repeat(1, steps, 1)
        elif tm1_input.ndim == 2:
            tm1_input = tm1_input.unsqueeze(0)
            steps = tm1_input.shape[1]
        else:
            raise ValueError("tm1_input must be 1D or 2D tensor")

        if tm1_input.shape[0] != 1:
            raise ValueError("tm1_input batch dimension must be 1")
        if tm1_input.shape[2] != n_tm1:
            raise ValueError(f"tm1_input last dimension must be n_tm1={n_tm1}")

        if tm1_input.shape[1] != steps:
            raise ValueError("tm1_input time dimension must match steps")
        
        tau = self.get_tau_vector()
        vrest = self.get_vrest_vector()
        
        edge_scales = torch.tensor(
            [self.scale_by_connection_type[key] for key in self._connection_type_keys],
            device=self.device, dtype=torch.float32
        )
        scaled_weights = self._weights * edge_scales
        
        if return_history:
            history = {'v': [v.clone()], 't': [0]}

        tm1_f = torch.zeros(1, n_tm1, device=self.device)
        tm1_v = torch.zeros(1, n_tm1, device=self.device)
        for step in range(steps):

            x = tm1_input[:, step, :]

            # stage 0: spatial DoG filter (center-surround)
            if self.tm1_H is not None:
                x = (x @ self.tm1_H.T)

            # stage 1: high pass filter
            hp_out = x - tm1_f
            tm1_f = tm1_f + self.dt * (x - tm1_f) / self.tm1_tau_hp

            # stage 2: rectification
            rect_out = torch.relu(hp_out)

            # stage 3: low pass filter
            v[:, tm1_mask] = tm1_v
            tm1_v = tm1_v + self.dt * (rect_out - tm1_v) / self.tm1_tau_lp
           

            r = self.activation(v)
            
            source_activities = r[:, self._source_indices]      
            edge_currents = source_activities * scaled_weights  
            synaptic_input = self.target_sum(edge_currents)  
            
            external_input = torch.zeros_like(v)
            external_input[:, tm1_mask] = tm1_v
            
            # Dynamics: τ_i * dV_i/dt = -V_i + Σ_j s_ij + Vrest_i + e_i
            # Rearranged: dV_i/dt = (-V_i + Σ_j s_ij + Vrest_i + e_i) / τ_i
            dv = (-v + synaptic_input + vrest.unsqueeze(0) + external_input) / tau.unsqueeze(0)
            dv[:, tm1_mask] = 0

            v = v + self.dt * dv

            if return_history:
                history['v'].append(v.clone())
                history['t'].append((step + 1) * self.dt)

            # if steady_state_tol is not None and torch.max(torch.abs(dv)).item() < steady_state_tol:
            #     break

        
        if return_history:
            history['v'] = torch.stack(history['v'], dim=1)
            history['t'] = torch.tensor(history['t'])
            return v, history
        return v