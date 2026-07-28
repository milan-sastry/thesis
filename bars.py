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


if __name__ == "__main__":
    # --- Lengthwise-extending bar over a central TmY4 cell ---
    # Bar is oriented at 90° (TmY4 preferred orientation) and extends symmetrically
    # from a short stub to a long bar. We track nearby TmY4 cells over time.

    # Parameters
    ANGLE = 90.0          # TmY4 preferred orientation
    BAR_WIDTH = 1.5
    BASE_LENGTH = 1.0     # starting length (short stub)
    FINAL_LENGTH = 16.0   # final length after full extension
    INTENSITY = 0.9
    SIGMA = 0.5
    DT = 0.1              # model timestep (s)
    GRAY_DURATION = 50    # baseline frames before bar appears
    EXTENSION_STEPS = 150 # frames for the extension phase
    HOLD_STEPS = 50       # frames to hold fully-extended bar

    # Pick the center TmY4 cell and find its RF center
    center_tmy4_idx = 2168
    p_center, q_center = an.get_rf_center(center_tmy4_idx)
    print(f"TmY4 index {center_tmy4_idx}: RF center = ({p_center}, {q_center})")

    # ---- Build stimulus ------------------------------------------------
    stim_gen = StimulusGenerator(lw.tm1_coords, lw.neuron_types, lw.row_ids)

    # Gray baseline
    gray = stim_gen.create_mean_gray(intensity=0.0)
    gray_frames = np.repeat(gray[None, :], GRAY_DURATION, axis=0)

    # Extending bar (center fixed, lengthwise growth)
    extension_duration = EXTENSION_STEPS * DT   # seconds
    hold_duration = HOLD_STEPS * DT
    ext_frames = extending_bar(
        angle=ANGLE,
        base_width=BAR_WIDTH,
        base_length=BASE_LENGTH,
        final_length=FINAL_LENGTH,
        extend_axis="lengthwise",
        intensity=INTENSITY,
        p_center=p_center,
        q_center=q_center,
        extension_duration=extension_duration,
        dt=DT,
        sigma=SIGMA,
        hold_final_duration=hold_duration,
    )

    stimulus = np.vstack([gray_frames, ext_frames])
    stimulus_t = stim_gen.to_torch(stimulus)

    # ---- Run model -------------------------------------------------------
    model = DrosophilaOpticLobeCircuit(
        lw.neuron_types,
        lw.source_indices,
        lw.target_indices,
        lw.weights,
        dt=DT,
    )
    _, history = model(stimulus_t, return_history=True)
    v_hist = to_numpy(history["v"].squeeze(0), dtype=np.float32)  # (steps, n_neurons)
    t_arr  = to_numpy(history["t"], dtype=np.float32)             # (steps,)

    # ---- Find TmY4 cells along the bar axis ------------------------------
    # Collect all TmY4 cells whose Tm1 inputs place them inside the final bar
    tmy4_in_bar = find_cells_in_bar(
        angle=ANGLE,
        width=BAR_WIDTH + 1.0,   # a bit generous in width
        length=FINAL_LENGTH,
        p_center=p_center,
        q_center=q_center,
        types=["TmY4"],
    )
    print(f"TmY4 cells inside final bar: {len(tmy4_in_bar)}")

    if not tmy4_in_bar:
        raise RuntimeError("No TmY4 cells found inside the bar — check center or bar dimensions.")

    # Compute each cell's signed distance along the bar axis from the center
    center_x, center_y = pq_to_xy(p_center, q_center)
    theta = np.radians(ANGLE)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # axis_dist > 0 → "above" center along bar length axis, < 0 → "below"
    def axis_distance(neuron_idx):
        cp, cq = an.get_rf_center(neuron_idx)
        if cp is None:
            return None
        cx, cy = pq_to_xy(cp, cq)
        dx, dy = cx - center_x, cy - center_y
        # In create_gaussian_bar: x_local = dx*cos + dy*sin  (width axis)
        #                         y_local = -dx*sin + dy*cos  (length axis)
        return float(-dx * sin_t + dy * cos_t)

    # Attach distances and sort by absolute distance (center-most first)
    cells_with_dist = []
    for idx in tmy4_in_bar:
        d = axis_distance(idx)
        if d is not None:
            cells_with_dist.append((idx, d))

    cells_with_dist.sort(key=lambda x: abs(x[1]))

    # Keep cells within ±FINAL_LENGTH/2 and limit to ≤12 for clarity
    cells_with_dist = [(i, d) for i, d in cells_with_dist if abs(d) <= FINAL_LENGTH / 2]
    cells_with_dist = cells_with_dist[:12]

    print("TmY4 cells selected (sorted by |axis dist| from center):")
    for idx, d in cells_with_dist:
        print(f"  global idx {idx:5d}  cell_id {lw.row_ids[idx]:6d}  axis_dist={d:+.2f}")

    # ---- Plot: activity vs time, colored by distance from center ---------
    cmap = plt.cm.coolwarm
    max_abs_d = max(abs(d) for _, d in cells_with_dist) if cells_with_dist else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_act, ax_len = axes

    # Top panel: activity time courses
    for idx, d in cells_with_dist:
        color = cmap(0.5 + 0.5 * d / max_abs_d)  # extremes → red/blue, centre → white
        ax_act.plot(t_arr, v_hist[:, idx], color=color, lw=1.2, label=f"d={d:+.1f}")

    # Shade the three stimulus phases
    t_bar_on  = GRAY_DURATION * DT
    t_bar_end = t_bar_on + extension_duration
    ax_act.axvspan(0,          t_bar_on,  alpha=0.08, color="grey",  label="baseline")
    ax_act.axvspan(t_bar_on,   t_bar_end, alpha=0.08, color="green", label="extension")
    ax_act.axvspan(t_bar_end,  t_arr[-1], alpha=0.08, color="blue",  label="hold")
    ax_act.set_xlim(t_arr[0], t_arr[-1])
    ax_act.set_ylabel("Voltage (a.u.)")
    ax_act.set_title(
        f"TmY4 activity near bar centre during lengthwise extension\n"
        f"angle={ANGLE}°, center=({p_center},{q_center}), "
        f"length {BASE_LENGTH}→{FINAL_LENGTH}"
    )
    ax_act.legend(loc="upper left", fontsize=7, ncol=2, title="Axis dist from centre")

    # Bottom panel: bar length over time
    n_ext = ext_frames.shape[0]
    t_ext = t_bar_on + np.arange(n_ext) * DT
    lengths = np.concatenate([
        np.linspace(BASE_LENGTH, FINAL_LENGTH, EXTENSION_STEPS),
        np.full(HOLD_STEPS, FINAL_LENGTH),
    ])
    ax_len.plot(t_ext, lengths[:n_ext], color="black", lw=1.5)
    ax_len.set_xlim(t_arr[0], t_arr[-1])
    ax_len.set_xlabel("Time (s)")
    ax_len.set_ylabel("Bar length")
    ax_len.set_title("Bar length over time")

    plt.tight_layout()
    import os
    os.makedirs("extending_bar", exist_ok=True)
    out_path = "extending_bar/extending_bar_TmY4_activity.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()
