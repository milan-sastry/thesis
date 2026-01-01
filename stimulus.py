import numpy as np
import matplotlib.pyplot as plt
import torch


class StimulusGenerator:
    def __init__(self, tm1_pq_coords, 
                 neuron_types: np.ndarray,
                 row_ids: np.ndarray):
        """
        Args:
            tm1_pq_coords: numpy array of shape (n, 3) with columns [cell_id, p, q]
            neuron_types:  array of neuron types in network order
            row_ids: array of cell IDs in network order
        """


        self.tm1_coords = {int(row[0]): (int(row[1]), int(row[2])) 
                              for row in tm1_pq_coords}
    
        
        tm1_mask = neuron_types == 'Tm1'
        tm1_indices_in_neuron_types = np.where(tm1_mask)[0]
        network_tm1_row_ids = row_ids[tm1_indices_in_neuron_types]
            
        self.cell_ids = list(network_tm1_row_ids)
        self.cell_ids_with_coords = [cid for cid in network_tm1_row_ids if cid in self.tm1_coords]
        self.n_cells = len(self.cell_ids)
        
        coords = np.array([self.tm1_coords[cid] for cid in self.cell_ids_with_coords])
        self.p_center = np.mean(coords[:, 0])
        self.q_center = np.mean(coords[:, 1])
        

    
    def create_bar(self, 
                   angle: float = 90.0,
                   offset: float = 0.0,
                   width: float = 2.0,
                   length: float = 10.0,
                   intensity: float = 1.0) -> np.ndarray:
        """
        
        Args:
            angle: Orientation angle in degrees. 
            offset: Distance to offset bar from center along the perpendicular direction
            width: Half-width of bar in lattice units (kinda doesn't work)
            length: Half-length of bar in lattice units
            intensity: Activation value for cells inside the bar
        Returns:
            stimulus: Array of shape (n_tm1_cells,) with activation values
        """
        theta = np.radians(angle)   
        parallel_h = np.cos(theta)  
        parallel_v = np.sin(theta)  
        
        perp_h = -np.sin(theta)
        perp_v = np.cos(theta)
        
        stimulus = np.zeros(self.n_cells)
        
        for idx, cell_id in enumerate(self.cell_ids):
            if cell_id not in self.tm1_coords:
                continue
            p, q = self.tm1_coords[cell_id]
            h = (q - self.q_center) - (p - self.p_center) 
            v = (p - self.p_center) + (q - self.q_center)  
            
            parallel_dist = h * parallel_h + v * parallel_v
            perp_dist = h * perp_h + v * perp_v
            
            if abs(perp_dist - offset) <= width and abs(parallel_dist) <= length:
                stimulus[idx] = intensity
                #print(cell_id)
        return stimulus
    
    def visualize_bar(self, stimulus: np.ndarray, title: str = "Bar Stimulus", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure
        
        coords = np.array([self.tm1_coords[cid] for cid in self.cell_ids_with_coords])
        stimulus_for_viz = np.array([stimulus[self.cell_ids.index(cid)] if cid in self.cell_ids else 0 
                                     for cid in self.cell_ids_with_coords])
        p_coords = coords[:, 0]
        q_coords = coords[:, 1]
        
        x = q_coords - p_coords  
        y = p_coords + q_coords  
        
        scatter = ax.scatter(x, y, c=stimulus_for_viz, cmap='hot', 
                           s=200, edgecolors='grey', linewidth=0.5,
                           vmin=0, vmax=1, marker='H')
        
        ax.set_aspect('auto')
        ax.set_xlabel('Posterior (h)')
        ax.set_ylabel('Dorsal (v)')
        ax.set_title(title)
        ax.grid(False)
        return fig, ax, scatter
    
    def to_torch(self, stimulus: np.ndarray, 
             device: str = 'cpu', target_size: int = None) -> torch.Tensor:
        tensor = torch.from_numpy(stimulus).float()
        
        # pad if target size is larger
        if target_size is not None and target_size > len(stimulus):
            padding = torch.zeros(target_size - len(stimulus))
            tensor = torch.cat([tensor, padding])
        
        tensor = tensor.unsqueeze(0).expand(1, -1)
        return tensor.to(device)
