from stimulus import StimulusGenerator
import load_weights as lw
import numpy as np
from network import DrosophilaOpticLobeCircuit
from dataset import filter_model_kwargs
from utils import to_numpy
import matplotlib.pyplot as plt

def generate_grating_response(
    angle,
    spatial_frequency,
    amplitude,
    center=None,
    grating_duration=50,
    model_settings=None,
    offset=0.5,
    use_flash=False,
    mean_duration=50,
    mean_intensity=0.5,
):
    """
    Generate a response to an oriented sine grating stimulus.

    If use_flash=True, prepends a mean-gray baseline phase and adds the mean
    to the grating, replicating the old flash experiment.  Otherwise the
    grating is presented directly until steady state.
    """
    stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids
    )
    grating = stimulus_generator.create_sine_grating(
        angle=angle,
        spatial_frequency=spatial_frequency,
        phase=np.pi/2,
        amplitude=amplitude,
        offset=offset,
        center=center,
    )

    stimulus_generator.visualize_bar(grating)
    plt.show()
    if use_flash:
        mean_gray = stimulus_generator.create_mean_gray(intensity=mean_intensity)
        blocks = [(mean_gray, mean_duration), (grating + mean_gray, grating_duration)]
    else:
        blocks = [(grating, grating_duration)]
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
    v_final, history = model(stimulus, return_history=True)
    v_final_np = to_numpy(v_final.squeeze(0), dtype=np.float32)
    v_hist_np = to_numpy(history["v"].squeeze(0), dtype=np.float32)
    t_np = to_numpy(history["t"], dtype=np.float32)


    return v_final_np, v_hist_np, t_np

def generate_moving_grating_response(
    angle,
    spatial_frequency,
    amplitude=0.5,
    center=None,
    offset=0.5,
    phase0=np.pi/2,
    omega=0.5,
    n_cycles=None,
    dt=0.1,
    steps=200,
    baseline_steps=50,
    model_settings=None,
    ):
        """
        Parameters
        ----------
        baseline_steps : int
            Number of dt-length steps to present the mean-gray baseline before
            the grating onset (default 50, i.e. 5 s at dt=0.1).
        amplitude : float
            Half-amplitude of the sinusoidal modulation. Must satisfy
            amplitude <= offset and amplitude <= (1 - offset) to keep the
            stimulus within [0, 1].
        n_cycles : int, optional
            If given, overrides `omega` so that exactly `n_cycles` complete
            temporal cycles fit within `steps` steps.  This is required for
            clean FFT / F1 analysis (avoids spectral leakage and mean bias).
            omega is set to n_cycles * 2π / (steps * dt).
        """
        if n_cycles is not None:
            omega = n_cycles * 2 * np.pi / (steps * dt)
        stimulus_generator = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids
        )
        grating = stimulus_generator.create_moving_grating_sequence(
            angle=angle,
            center=center,
            spatial_frequency=spatial_frequency,
            amplitude=amplitude,
            offset=offset,
            phi0=phase0,
            omega=omega,
            dt=dt,
            steps=steps,
        )

        mean_gray = stimulus_generator.create_mean_gray(intensity=offset)
        baseline = stimulus_generator.sequence_from_blocks([(mean_gray, baseline_steps)])
        sequence = np.concatenate([baseline, grating], axis=0)
        # stimulus_generator.visualize_sequence(sequence)
        # plt.show()

        stimulus = stimulus_generator.to_torch(sequence)
        model_kwargs = filter_model_kwargs(model_settings)
        tm1_row_ids = lw.row_ids[lw.neuron_types == 'Tm1']
        model = DrosophilaOpticLobeCircuit(
            lw.neuron_types,
            lw.source_indices,
            lw.target_indices,
            lw.weights,
            tm1_coords=lw.tm1_coords,
            tm1_row_ids=tm1_row_ids,
            **model_kwargs,
        )
        v_final, history = model(stimulus, return_history=True)
        v_final_np = to_numpy(v_final.squeeze(0), dtype=np.float32)
        v_hist_np = to_numpy(history["v"].squeeze(0), dtype=np.float32)
        t_np = to_numpy(history["t"], dtype=np.float32)
        return v_final_np, v_hist_np, t_np

if __name__ == "__main__":
    stimGen = StimulusGenerator(
        lw.tm1_coords,
        lw.neuron_types,
        lw.row_ids)
    sequence = stimGen.create_moving_grating_sequence(
        angle=60,
        spatial_frequency=np.pi/2,
        amplitude=0.5,
        offset=0.5,
        phi0=np.pi/2,
        omega=1,
        dt=0.1,
        steps=200,
    )
    stimGen.visualize_sequence(sequence)
    