import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from scipy.special import erf
import matplotlib.ticker as ticker
from utils import pq_to_xy, create_sine_grating as _create_sine_grating

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

    def create_gaussian_bar(
        self,
        width,
        length,
        p_center,
        q_center,
        on=True,
        angle=0.0,
        offset=0.0,
        intensity: float = 1.0,
        sigma=0.5,
        length_offset=0.0,
    ) -> np.ndarray:
        """
        Creates a bar stimulus using a precise Gaussian integration (erf).
        The intensity at each neuron is the integral of a Gaussian PSF across the bar's width.
        offset shifts along the bar width axis, length_offset shifts along the bar length axis.
        """
        rect_x, rect_y = pq_to_xy(p_center, q_center)

        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rect_x += offset * cos_theta
        rect_y += offset * sin_theta
        rect_x += length_offset * (-sin_theta)
        rect_y += length_offset * cos_theta

        rect = patches.Rectangle(
            (rect_x - width/2, rect_y - length/2),
            width,
            length,
            angle=angle,
            rotation_point='center',
            linewidth=2,
            edgecolor='cyan',
            facecolor='none',
            linestyle='--'
        )

        stimulus = np.zeros(self.n_cells, dtype=float) if on else np.ones(self.n_cells, dtype=float) * intensity
        
        for idx, cell_id in enumerate(self.cell_ids):
            if cell_id not in self.tm1_coords:
                continue
            
            p, q = self.tm1_coords[cell_id]
            x_raw, y_raw = pq_to_xy(p, q)
            
            dx = x_raw - rect_x
            dy = y_raw - rect_y
            
            x_local = dx * cos_theta + dy * sin_theta
            y_local = -dx * sin_theta + dy * cos_theta
            
            if abs(y_local) <= length / 2:
                
                z1 = (-width/2 - x_local) / (sigma * np.sqrt(2))
                z2 = (width/2 - x_local) / (sigma * np.sqrt(2))
                
                activation = 0.5 * (erf(z2) - erf(z1))
                
                if on:
                    stimulus[idx] = activation * intensity
                else:
                    stimulus[idx] = (1.0 - activation) * intensity
                    
        return rect, stimulus

    def visualize_bar(self, stimulus: np.ndarray, title: str = "Stimulus", rect: patches.Rectangle = None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure
        
        coords = np.array([self.tm1_coords[cid] for cid in self.cell_ids_with_coords])
        stimulus_for_viz = np.array([stimulus[self.cell_ids.index(cid)] if cid in self.cell_ids else 0 
                                     for cid in self.cell_ids_with_coords])
        p_coords = coords[:, 0]
        q_coords = coords[:, 1]
        
        x, y = pq_to_xy(p_coords, q_coords)
        
        scatter = ax.scatter(x, y, c=stimulus_for_viz, cmap='Greys_r', 
                           s=100, edgecolors='grey', linewidth=0.5,
                           vmin=0, vmax=1, marker='H')

        cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Stimulus Value", fontsize=12)
        
        # for px, py, p_val, q_val in zip(x, y, p_coords, q_coords):
        #     ax.text(px, py, f'({int(p_val)},{int(q_val)})', 
        #           ha='center', va='center', fontsize=4, 
        #           fontweight='bold', color='red')
        
        if rect is not None:
            ax.add_patch(rect)
        
        ax.set_aspect('equal')
        ax.set_xlabel('Posterior (h)')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(4/np.sqrt(3)))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1/np.sqrt(3)))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlim(-21, 15)
        ax.set_ylim(3, 40)
        ax.set_ylabel('Dorsal (v)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', color='grey', which='both', linewidth=0.5, alpha=0.5)


        return fig, ax, scatter

    def visualize_sequence(self, stimulus):
        unique_frames = []
        previous_frame = None
        for frame in stimulus:
            if previous_frame is None or not np.array_equal(frame, previous_frame):
                unique_frames.append(frame)
                previous_frame = frame

        for frame in unique_frames:
            self.visualize_bar(frame)
            plt.show()
        plt.show()
        return

    def create_sine_grating(
        self,
        angle: float,
        spatial_frequency: float,
        phase: float,
        amplitude: float,
        offset: float,
        center: tuple[float, float] | None = None,
    ) -> np.ndarray:
        return _create_sine_grating(
            self.cell_ids,
            self.tm1_coords,
            self.n_cells,
            angle,
            spatial_frequency,
            phase,
            amplitude,
            offset,
            center,
        )

    def create_mean_gray(self, intensity: float = 0.0) -> np.ndarray:
        return np.ones(self.n_cells, dtype=float) * intensity

    def create_white_noise(self, intensity: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng()
        noise = rng.standard_normal(self.n_cells) * intensity
        return noise

    def create_white_noise_sequence(
        self,
        steps: int,
        intensity: float = 1.0,
        mean: float = 0.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Creates a time-varying white noise stimulus.
        """
        if steps <= 0:
            return np.zeros((0, self.n_cells), dtype=float)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((steps, self.n_cells)) * intensity + mean
        return noise


    def sequence_from_blocks(self, blocks: list[tuple[np.ndarray, int]]) -> np.ndarray:
        """
        Build a time series by concatenating (stimulus, duration) blocks.
        Returns array of shape (steps, n_cells).
        """
        sequence = []
        for stimulus, duration in blocks:
            if duration <= 0:
                continue
            if stimulus.shape != (self.n_cells,):
                raise ValueError("Each stimulus must have shape (n_cells,)")
            sequence.append(np.repeat(stimulus[None, :], duration, axis=0))
        return np.vstack(sequence) if sequence else np.zeros((0, self.n_cells), dtype=float)
    
    def create_moving_grating_sequence(
        self,
        angle,
        spatial_frequency=0.1,
        amplitude=1.0,
        offset=0.0,
        phi0=0.0,
        temporal_freq=2.0,
        dt=0.1,
        steps=200,
    ):
        frames = []
        for t_idx in range(steps):
            t = t_idx * dt
            phi_t = phi0 - 2 * np.pi * temporal_freq * t
            frame = self.create_sine_grating(
                angle=angle,
                spatial_frequency=spatial_frequency,
                phase=phi_t,
                amplitude=amplitude,
                offset=offset,
            )
            frames.append(frame)
        return np.stack(frames, axis=0)

    def to_torch(
        self,
        stimulus: np.ndarray,
        device: str = 'cpu',
        target_size: int = None,
    ) -> torch.Tensor:
        """
        Convert a stimulus to torch without adding a batch dimension.
        Returns 1D (n_cells) for single-frame input or 2D (steps, n_cells)
        for time-varying input.
        """
        tensor = torch.from_numpy(stimulus).float()

        if tensor.ndim == 1:
            n_cells = tensor.shape[0]
            if target_size is not None and target_size > n_cells:
                padding = torch.zeros(target_size - n_cells)
                tensor = torch.cat([tensor, padding])
        elif tensor.ndim == 2:
            steps, n_cells = tensor.shape
            if target_size is not None and target_size > n_cells:
                padding = torch.zeros(steps, target_size - n_cells)
                tensor = torch.cat([tensor, padding], dim=1)
        else:
            raise ValueError("stimulus must be 1D (n_cells) or 2D (steps, n_cells)")

        return tensor.to(device)
    