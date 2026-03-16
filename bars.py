import load_weights as lw
import numpy as np
from stimulus import StimulusGenerator
import matplotlib.pyplot as plt
import analysis as an
from network import DrosophilaOpticLobeCircuit
import tuning_curves as tc
from dataset import filter_model_kwargs
from visualize import visualize_responses
from utils import to_numpy, pq_to_xy

def moving_bar(
    angle,
    width,
    length,
    intensity,
    p_center,
    q_center,
    speed,
    steps,
    on=True,
    sigma=0.5,
    start_offset=0.0,
):
    """
    Creates a moving Gaussian bar sequence by shifting bar offset each step.

    Args:
        speed: Offset increment per frame in local bar coordinates.
        steps: Number of frames in the sequence.
    Returns:
        np.ndarray of shape (steps, n_cells).
    """
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids,
    )

    frames = []
    for step in range(steps):
        offset = start_offset + speed * step
        _, stimulus = stimulus_generator.create_gaussian_bar(
            width=width,
            length=length,
            p_center=p_center,
            q_center=q_center,
            on=on,
            angle=angle,
            offset=offset,
            intensity=intensity,
            sigma=sigma,
        )
        frames.append(stimulus)

    if not frames:
        return np.zeros((0, stimulus_generator.n_cells), dtype=float)
    return np.stack(frames, axis=0)


def extending_bar(
    angle,
    base_width,
    base_length,
    final_length,
    extend_axis,
    intensity,
    p_center,
    q_center,
    extension_duration,
    dt,
    on=True,
    sigma=0.5,
    hold_final_duration=0.0,
):
    """
    Create an extending Gaussian bar sequence while keeping center fixed.

    Args:
        base_width: Starting width of the bar.
        base_length: Starting length of the bar.
        final_length: Final extent on the axis being extended.
            - extend_axis='lengthwise' -> final bar length
            - extend_axis='widthwise' -> final bar width
        extend_axis: 'lengthwise' or 'widthwise'.
        extension_duration: Duration (seconds) of the extension phase.
        dt: Seconds per frame.
        hold_final_duration: Optional final hold time (seconds).
    Returns:
        np.ndarray of shape (steps, n_cells).
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if extension_duration < 0:
        raise ValueError("extension_duration must be >= 0")
    if hold_final_duration < 0:
        raise ValueError("hold_final_duration must be >= 0")

    axis = str(extend_axis).strip().lower()
    if axis not in {"lengthwise", "widthwise"}:
        raise ValueError("extend_axis must be 'lengthwise' or 'widthwise'")

    start_extent = base_length if axis == "lengthwise" else base_width
    if final_length <= 0 or start_extent <= 0:
        raise ValueError("base and final extents must be > 0")

    extension_steps = max(1, int(np.round(extension_duration / dt)))
    hold_steps = int(np.round(hold_final_duration / dt))
    extent_values = np.linspace(start_extent, final_length, extension_steps)

    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids,
    )

    frames = []
    for extent in extent_values:
        if axis == "lengthwise":
            width = base_width
            length = extent
        else:
            width = extent
            length = base_length

        _, stimulus = stimulus_generator.create_gaussian_bar(
            width=width,
            length=length,
            p_center=p_center,
            q_center=q_center,
            on=on,
            angle=angle,
            offset=0,
            length_offset=0.0,
            intensity=intensity,
            sigma=sigma,
        )
        frames.append(stimulus)

    if hold_steps > 0 and frames:
        final_frame = frames[-1]
        for _ in range(hold_steps):
            frames.append(final_frame.copy())

    if not frames:
        return np.zeros((0, stimulus_generator.n_cells), dtype=float)
    return np.stack(frames, axis=0)

def rotating_bar(
    p_center,
    q_center,
    width=1.0,
    length=10.0,
    intensity=1.0,
    angles=None,
    angle_start=0.0,
    angle_stop=165.0,
    angle_step=15.0,
    frames_per_angle=50,
    on=True,
    sigma=0.5,
    include_gray=False,
    gray_duration=0,
    gray_intensity=0.0,
):
    """
    Create a stimulus sequence of a bar rotating in place around a given center.

    Angles can be specified explicitly via `angles`, or generated from
    `angle_start` / `angle_stop` / `angle_step` (endpoint excluded).

    Args:
        p_center, q_center: Hex-grid center of rotation.
        width, length: Bar dimensions.
        intensity: Peak bar intensity.
        angles: Explicit array-like of angles (degrees). Overrides start/stop/step.
        angle_start: First angle in degrees (inclusive).
        angle_stop: Last angle in degrees (exclusive).
        angle_step: Increment between angles.
        frames_per_angle: How many timesteps each orientation is held.
        on: True for bright bar on dark background; False for dark on bright.
        sigma: Gaussian smoothing width of the bar edge.
        include_gray: If True, insert a mean-gray gap between each orientation.
        gray_duration: Number of frames for each gray gap.
        gray_intensity: Intensity of the gray gap stimulus.

    Returns:
        np.ndarray of shape (total_frames, n_cells).
    """
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids,
    )

    if angles is None:
        angles = np.arange(angle_start, angle_stop, angle_step)
    else:
        angles = np.asarray(angles, dtype=float)

    gray = stimulus_generator.create_mean_gray(intensity=gray_intensity)

    blocks = []
    for angle in angles:
        _, bar_frame = stimulus_generator.create_gaussian_bar(
            width=width,
            length=length,
            p_center=p_center,
            q_center=q_center,
            on=on,
            angle=float(angle),
            offset=0.0,
            intensity=intensity,
            sigma=sigma,
        )
        blocks.append((bar_frame, frames_per_angle))
        if include_gray and gray_duration > 0:
            blocks.append((gray, gray_duration))

    return stimulus_generator.sequence_from_blocks(blocks)

def find_cells_in_bar(
    angle,
    width,
    length,
    p_center,
    q_center,
    types=['Dm3v', 'Dm3p', 'Dm3q', 'TmY4', 'TmY9q', 'TmY9q⊥'],
    threshold=0.0,
):
    if width <= 0 or length <= 0:
        raise ValueError("width and length must be > 0")


    tm1_table = np.asarray(lw.tm1_coords)
    tm1_p = tm1_table[:, 1].astype(float)
    tm1_q = tm1_table[:, 2].astype(float)
    tm1_ids = tm1_table[:, 0]

    center_x, center_y = pq_to_xy(p_center, q_center)
    x, y = pq_to_xy(tm1_p, tm1_q)

    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    dx = x - center_x
    dy = y - center_y
    x_local = dx * cos_theta + dy * sin_theta
    y_local = -dx * sin_theta + dy * cos_theta

    tm1_in_bar = (np.abs(x_local) <= (width / 2.0)) & (np.abs(y_local) <= (length / 2.0))
    if not np.any(tm1_in_bar):
        return []

    tm1_source_mask = np.isin(lw.row_ids, tm1_ids[tm1_in_bar]) & (lw.neuron_types == 'Tm1')
    tm1_source_indices = np.where(tm1_source_mask)[0]
    if tm1_source_indices.size == 0:
        return []

    target_mask = np.isin(lw.neuron_types, list(types))
    target_indices = np.where(target_mask)[0]
    if target_indices.size == 0:
        return []

    tm1_to_targets = lw.W_raw[tm1_source_indices][:, target_indices]
    has_nonzero_tm1_input = np.asarray(tm1_to_targets.getnnz(axis=0) > threshold).ravel()
    return target_indices[has_nonzero_tm1_input].tolist()





def generate_bar_response(
    angle,
    width,
    length,
    amplitude,
    center,
    bar_duration=50,
    model_settings=None,
    sigma=0.5,
    on=True,
    use_flash=False,
    mean_duration=50,
    mean_intensity=0.1,
):
    """
    Generate a response to an oriented Gaussian bar stimulus.

    If use_flash=True, prepends a mean-gray baseline phase and adds the mean
    to the bar, replicating the old flash experiment.  Otherwise the bar is
    presented directly until steady state.
    """
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids,
    )
    p_center, q_center = center
    _, bar = stimulus_generator.create_gaussian_bar(
        width=width,
        length=length,
        p_center=p_center,
        q_center=q_center,
        on=on,
        angle=angle,
        offset=0.0,
        intensity=amplitude,
        sigma=sigma,
    )

    if use_flash:
        mean_gray = stimulus_generator.create_mean_gray(intensity=mean_intensity)
        blocks = [(mean_gray, mean_duration), (bar + mean_gray, bar_duration)]
    else:
        blocks = [(bar, bar_duration)]
    sequence = stimulus_generator.sequence_from_blocks(blocks)
    stimulus = stimulus_generator.to_torch(sequence)

    model_kwargs = filter_model_kwargs(model_settings)
    model = DrosophilaOpticLobeCircuit(
        lw.neuron_types,
        lw.source_indices,
        lw.target_indices,
        lw.weights,
        **model_kwargs,
    )
    # stimulus_generator.visualize_sequence(sequence)
    # plt.show()
    v_final, history = model(stimulus, return_history=True)
    v_final_np = to_numpy(v_final.squeeze(0), dtype=np.float32)
    v_hist_np = to_numpy(history["v"].squeeze(0), dtype=np.float32)
    t_np = to_numpy(history["t"], dtype=np.float32)


    return v_final_np, v_hist_np, t_np, bar


# if __name__ == "__main__":
    # scale_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # curves_arr = []
    # params_arr = []

    # centers = {
    #     1521: (18.5, 17.0), # Dm3p
    #     1685: (18,16.5), # Dm3q
    #     972: (18.5, 17.5), # Dm3v
    #     2414: (18, 15), # TmY9q
    #     2498: (18,17),# TmY9q⊥
    #     2168: (17.5, 16.5), # TmY4
        
        
    #     1481: (10.5,18), # Dm3p with most total TmY9q⊥ inputs
    #     1456: (11,10), # Dm3p with large number of Tmy9q⊥ inputs and high total after normalization
    #     1559: (21,15), # Dm3p with most unique TmY9q⊥ inputs
    #     982: (20,14.5), # Dm3v with second most total TmY9q⊥ inputs
    #     962: (10,12), # Dm3v with most unique TmY9q⊥ inputs
    #     2452: (13,6), # TmY9q⊥ -- most synapses to Dm3v + Dm3p
    #     2608: (11,18.5), #Tmy9q⊥ -- many synapses to Dm3p
    #     2547: (14,12.5), #Tmy9q⊥ -- many incoming synapses from TmY9q


    # }

    # neuron_index = 2547
    # p_center, q_center = centers[neuron_index]
    # input_indices, input_cell_ids, input_types, input_synapse_counts = an.get_presynaptic_inputs(neuron_index)
    # output_indices, output_cell_ids, output_types, output_synapse_counts = an.get_postsynaptic_targets(neuron_index)

    # dm3v_indices = output_indices[np.where(output_types == "Dm3v")[0]]
    # dm3p_indices = output_indices[np.where(output_types == "Dm3p")[0]]
    # print(dm3p_indices)



    # for scale_factor in scale_factors:
    #     scale_by_connection_type = {
    #         # ('Dm3p', 'Dm3v'): scale_factor,
    #         # ('Dm3q', 'Dm3v'): scale_factor,
    #         # ('Dm3p', 'Dm3q'): scale_factor,
    #         # ('Dm3q', 'Dm3p'): scale_factor,
    #         # ('Dm3v', 'Dm3p'): scale_factor,
    #         # ('Dm3v', 'Dm3q'): scale_factor, 
    #         # ('Dm3p', 'Dm3p'): scale_factor,
    #         # ('Dm3q', 'Dm3q'): scale_factor,
    #         # ('Dm3v', 'Dm3v'): scale_factor,
    #         # ('Dm3v', 'TmY4'): scale_factor,
    #         # ('Dm3p', 'TmY9q'): scale_factor,
    #         # ('Dm3q', 'TmY9q⊥'): scale_factor,
    #         # ('Dm3v', 'TmY9q'): scale_factor,
    #         # ('Dm3q', 'TmY9q'): scale_factor,
    #         # ('Dm3v', 'TmY9q⊥'): scale_factor,
    #         # ('Dm3p', 'TmY4'): scale_factor,
    #         # ('Dm3q', 'TmY4'): scale_factor,
    #         # ('Dm3p', 'TmY9q⊥'): scale_factor,
    #         # ('TmY9q⊥', 'Dm3p') : scale_factor,
    #         # ('TmY9q⊥', 'Dm3v') : scale_factor,
    #         ('TmY9q', 'TmY9q⊥') : scale_factor,
    #     }
    #     model_settings = {
    #         "scale_by_connection_type": scale_by_connection_type,
    #         # "vrest_init": -0.1,
    #     }

    #     runs_data = []
    #     for angle in range(0, 180, 15):
    #         v_final, v_hist, t = generate_bar_response(
    #             angle=angle,
    #             width=1.5,
    #             length=20.0,
    #             amplitude=0.9,
    #             mean_duration=100,
    #             mean_intensity=0.1,
    #             bar_duration=100,
    #             model_settings=model_settings,
    #             center=(p_center, q_center),
    #             sigma=0.5,
    #         )
    #         runs_data.append({
    #             "v_final": v_final,
    #             "v_history": v_hist,
    #             "t": t,
    #             "angle": angle,
    #         })
    #         #visualize_responses(v_final[None, :], {"v": v_hist[None, :, :], "t": t}, neuron_indices=[neuron_index], title=f"Bar {angle}°, Scale {scale_factor}")


    #     results = {"runs": runs_data}
    #     curves = tc.tuning_curve(
    #         results,
    #         flash_windows=(10, 20.0),
    #         baseline_window=(0, 10.0),
    #         fit=True,
    #         active_only=False,
    #         aggregation="individual",
    #         neuron_ids=[neuron_index],
    #     )
    #     curves_arr.append(curves)
    #     params_arr.append(scale_factor)

    # tc.plot_curves_by_param(
    #     curves_arr,
    #     params_arr,
    #     filename="flash_bar_Dm3_scaled_individual_1.0.png",
    #     show_points=True,
    #     ylim=(0.0, 0.6),
    # )
    # plt.show()