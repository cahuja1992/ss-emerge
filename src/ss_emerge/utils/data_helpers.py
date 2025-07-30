import numpy as np
from scipy.signal import welch
import torch
import torch.nn.functional as F
import random

def calculate_de(eeg_segment, sfreq, bands):
    """
    Calculates Differential Entropy (DE) for an EEG segment across specified frequency bands.
    eeg_segment: (channels, time_points)
    sfreq: Sampling frequency
    bands: Dictionary of frequency bands, e.g., {'delta': (1, 4)}
    
    Returns:
        np.ndarray: DE features of shape (num_bands, num_channels).
    """
    de_features = []
    for band_name, (low, high) in bands.items():
        # Compute Power Spectral Density (PSD) using Welch's method
        # n_per_seg = 2 * sfreq ensures a 2-second window for PSD calculation, common practice
        # CORRECTED: Changed 'sfreq' to 'fs' for the welch function
        freqs, psd = welch(eeg_segment, fs=sfreq, nperseg=min(2*sfreq, eeg_segment.shape[-1]), average='mean', axis=-1)
        
        idx_band = np.where((freqs >= low) & (freqs <= high))[0]
        
        if len(idx_band) == 0:
            # If no frequencies in band, append zeros for that band across all channels
            de_features.append(np.zeros(eeg_segment.shape[0]))
            continue
            
        band_power = np.trapz(psd[:, idx_band], freqs[idx_band], axis=-1)
        
        # Replace zero or negative band_power with a small positive number to avoid log errors
        band_power[band_power <= 0] = np.finfo(float).eps 
        
        # Differential Entropy formula: 0.5 * log(2 * pi * e * band_power)
        de = 0.5 * np.log(2 * np.pi * np.exp(1) * band_power)
        de_features.append(de)
        
    return np.array(de_features)


def group_sample_subjects(bs, ss, Q):
    """
    Samples and reorders subject indices for Meiosis augmentation,
    applying dataset-specific adjustments (like SEED's subject uniqueness check).
    Adapted from SSL_training(SEED).py.

    Args:
        bs (int): Batch size (number of video clips in the current mini-batch).
                  Used in the original code for context, but not for sampling logic itself.
        ss (int): Total available subjects/trials for a single video clip
                  (e.g., 45 for SEED, 40 for DEAP).
        Q (int): Number of samples per augmented view (half of the paired samples used for recombination).

    Returns:
        list: A shuffled list of subject indices (`rand_subs_stre`) after applying
              dataset-specific adjustments.
    """
    rand_subs_stre = random.sample(range(0, ss), ss)
    
    if ss == 45: # Assuming SEED-like data, where subjects have 3 trials each
        loop_limit = int((ss - 1) / 2)
        
        change_indices = []
        change_subs_to_swap = []
        
        for i in range(loop_limit):
            if (rand_subs_stre[i] // 3) == (rand_subs_stre[i + Q] // 3):
                change_indices.append(i)
                change_subs_to_swap.append(rand_subs_stre[i])
        
        if len(change_indices) == 1:
            problem_idx = change_indices[0]
            if (rand_subs_stre[problem_idx] // 3) == (rand_subs_stre[-1] // 3):
                if problem_idx == (loop_limit - 1):
                    temp = rand_subs_stre[problem_idx]
                    rand_subs_stre[problem_idx] = rand_subs_stre[problem_idx - 1]
                    rand_subs_stre[problem_idx - 1] = temp
                else:
                    temp = rand_subs_stre[problem_idx]
                    rand_subs_stre[problem_idx] = rand_subs_stre[problem_idx + 1]
                    rand_subs_stre[problem_idx + 1] = temp
            else:
                temp = rand_subs_stre[problem_idx]
                rand_subs_stre[problem_idx] = rand_subs_stre[-1]
                rand_subs_stre[-1] = temp
        elif len(change_indices) >= 1:
            first_problem_idx = change_indices.pop(0)
            change_indices.append(first_problem_idx)
            for c_idx, original_val in zip(change_indices, change_subs_to_swap):
                rand_subs_stre[c_idx] = original_val

    return rand_subs_stre

def get_group_representations(embeddings, Q):
    """
    Aggregates individual trial embeddings into a single group-level representation using MaxPool1D.
    Adapted from SSL_training(SEED).py.

    Args:
        embeddings (torch.Tensor): Tensor of embeddings from augmented signals.
                                   Expected shape: (num_total_augmented_samples, embedding_dim).
        Q (int): Number of samples (individual trial embeddings) that constitute one "view" within a group.

    Returns:
        torch.Tensor: Group-level representations after MaxPool1D.
                      Shape: (num_video_clips_in_batch * 2, embedding_dim).
    """
    reshaped_embeddings = embeddings.reshape(-1, Q, embeddings.shape[-1])
    group_reps = torch.max(reshaped_embeddings, dim=1)[0]
    return group_reps