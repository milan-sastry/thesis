import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Literal
from load_weights import tm1_coords

class HexBarStimulus:

    
    def __init__(self, tm1_pq_coords: Dict[int, Tuple[int, int]]):
        self.tm1_coords = tm1_pq_coords
        self.cell_ids = list(tm1_pq_coords.keys())
        self.n_cells = len(self.cell_ids)
        
        coords = np.array([tm1_pq_coords[cid] for cid in self.cell_ids])
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
            angle: Orientation angle in degrees. 0° = posterior (h-axis), 90° = dorsal (v-axis)
                   Measured counter-clockwise from the posterior direction.
            offset: Distance to offset bar from center along the perpendicular direction
            width: Half-width of bar in lattice units
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
            p, q = self.tm1_coords[cell_id]
            

            h = (q - self.q_center) - (p - self.p_center) 
            v = (p - self.p_center) + (q - self.q_center)  
            

            parallel_dist = h * parallel_h + v * parallel_v
            perp_dist = h * perp_h + v * perp_v
            
            if abs(perp_dist - offset) <= width and abs(parallel_dist) <= length:
                stimulus[idx] = intensity
        
        return stimulus
    
    def visualize_bar(self, stimulus: np.ndarray, title: str = "Bar Stimulus", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        

        coords = np.array([self.tm1_coords[cid] for cid in self.cell_ids])
        p_coords = coords[:, 0]
        q_coords = coords[:, 1]
        
        x = q_coords - p_coords  
        y = p_coords + q_coords  
        
        scatter = ax.scatter(x, y, c=stimulus, cmap='hot', 
                           s=100, edgecolors='gray', linewidth=0.5,
                           vmin=0, vmax=1)
        
        ax.set_aspect('equal')
        ax.set_xlabel('Posterior (h) →')
        ax.set_ylabel('Dorsal (v) →')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax, scatter
    
    def to_torch(self, stimulus: np.ndarray, 
             device: str = 'cpu', target_size: int = None) -> torch.Tensor:
        tensor = torch.from_numpy(stimulus).float()
        
        # Pad if target size is larger
        if target_size is not None and target_size > len(stimulus):
            padding = torch.zeros(target_size - len(stimulus))
            tensor = torch.cat([tensor, padding])
        
        tensor = tensor.unsqueeze(0).expand(1, -1)
        return tensor.to(device)
