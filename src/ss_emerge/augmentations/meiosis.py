import numpy as np
import random
import torch # Not directly used in Meiosis, but useful for overall context
from ss_emerge.utils.data_helpers import group_sample_subjects # Import the subject sampler

def Meiosis(signal, Q, rand_subs_stre, split):
    """
    Performs meiosis-inspired data augmentation for a single video's EEG signals.
    Adapted from kanhaoning/self-supervised-group-meiosis-contrastive-learning-for-eeg-based-emotion-recognition/SSL_training(SEED).py
    and SSL_training(DEAP).py.

    Args:
        signal (np.ndarray): Input EEG signals for one video clip,
                             shape (num_subjects, F, channels, time_points).
                             (e.g., (45, 5, 62, 200) for SEED-like, or (40, 1, 32, 128) for DEAP-like).
        Q (int): This Q parameter is used in the `rand_subs_stre` indexing (`i+Q`).
                 It's the "group size" used to pick the second subject for recombination.
        rand_subs_stre (list): A list of shuffled subject indices from the current video's subjects
                                from which to pick pairs for recombination. Its length should be
                                at least `num_recombination_pairs + Q`.
        split (int): The temporal midpoint (index) for splitting and recombining signals
                     along the `time_points` (last) dimension.

    Returns:
        np.ndarray: Augmented signals. The number of output samples depends on
                    `num_recombination_pairs` (2 * `num_recombination_pairs` + 1).
                    For `num_subjects = 45` (SEED-like), this is 45 samples.
    """
    num_subjects = signal.shape[0]
    
    # Determine num_recombination_pairs based on original code's fixed loops.
    if num_subjects == 45: # SEED: 15 subjects * 3 trials each = 45 samples per video group
        num_recombination_pairs = 22 # `range(0, 22)` in SSL_training(SEED).py
    elif num_subjects == 40: # DEAP: 40 trials per subject, if `ss` is 40.
        num_recombination_pairs = 16 # `range(0, 16)` in SSL_training(DEAP).py
    else:
        # Fallback if other `num_subjects` are encountered (e.g., for different datasets).
        # This part of the original code is heuristic. We'll stick to fixed numbers for now.
        raise ValueError(f"Unsupported num_subjects_per_video: {num_subjects}. Expected 45 (SEED) or 40 (DEAP).")

    new_signal1 = []
    new_signal2 = []

    for i in range(num_recombination_pairs):
        si = rand_subs_stre[i]
        sj = rand_subs_stre[i + Q] # Q=2 means take rand_subs_stre[i] and rand_subs_stre[i+2]
        
        # Concatenate along the time_points dimension (axis=-1)
        xi = np.concatenate([signal[si, :, :, :split], signal[sj, :, :, split:]], axis=-1)
        xj = np.concatenate([signal[sj, :, :, :split], signal[si, :, :, split:]], axis=-1)
        
        new_signal1.append(xi)
        new_signal2.append(xj)
    
    new_signal = new_signal1 + new_signal2
    
    # The original code for SEED also appends `signal[rand_subs_stre[-1]]`
    # This extra sample makes the total output size (2*num_recombination_pairs + 1)
    if rand_subs_stre and len(rand_subs_stre) > 0:
        new_signal.append(signal[rand_subs_stre[-1]])
    
    return np.array(new_signal)


def group_samples_for_contrastive_learning(signal_batch, Q):
    """
    Orchestrates the Meiosis augmentation for a batch of video clips and selects specific samples
    to form the input `groups` tensor for the model, as per the original kanhaoning implementation.

    Args:
        signal_batch (np.ndarray): A batch of EEG signals, shape (P_clips, ss_per_video, F, C, T),
                                   where P_clips is the number of video clips in the batch,
                                   and ss_per_video is the total subjects/trials available per video.
                                   F is num_freq_bands.
        Q (int): The Q parameter for Meiosis (group size for selecting 2nd subject in pair).
                 This also defines the number of samples taken for each view (A or B) within a group.

    Returns:
        np.ndarray: The concatenated group views for the batch,
                    shape (2 * P_clips * Q, F, C, T). This is the 'groups' tensor
                    that directly enters the model in pretrain.py.
    """
    groups_for_model_input = [] # This will collect (Q, F, C, T) for View A and View B
    P_clips, ss_per_video, F_bands, C_channels, T_timesteps = signal_batch.shape

    # loop over P_clips (video clips in the batch)
    for i in range(P_clips):
        current_video_signal = signal_batch[i] # (ss_per_video, F, C, T)
        
        # Step 1: Sample and adjust subject indices for Meiosis (rand_subs_stre)
        # `bs` here in group_sample_subjects is the P_clips from outer loop context (not strictly used by sampling)
        processed_rand_subs_stre = group_sample_subjects(P_clips, ss_per_video, Q) 

        # Step 2: Determine split point for Meiosis
        split = random.randint(1, T_timesteps - 2) # time_points dimension (last dim)

        # Step 3: Apply Meiosis augmentation
        # The result `augmented_video_signals_raw` is (2*num_recomb_pairs + 1, F, C, T)
        # e.g., (45, F, C, T) for SEED-like, or (33, F, C, T) for DEAP-like.
        augmented_video_signals_raw = Meiosis(current_video_signal, Q, processed_rand_subs_stre, split)
        
        # Step 4: Select specific Q samples for view_A and Q samples for view_B
        # This is CRITICAL. The original code (SSL_training.py) does this selection *after* Meiosis,
        # and forms the `groups` tensor that enters the model.
        num_augmented_samples_per_video = augmented_video_signals_raw.shape[0]
        
        # Original code: `rand_subs = random.sample(range(ss - 1), 2 * Q)`
        # `ss - 1` refers to (total_subjects - 1) in the *original* raw data before Meiosis.
        # But here, `augmented_video_signals_raw` already contains the results of Meiosis.
        # The `range(ss - 1)` part from original SSL code seems to implicitly refer to the number of
        # augmented samples available to pick from. It's usually `(2*num_recomb_pairs+1) - 1`.
        
        # Let's align with the range. It picks 2*Q indices from the *Meiosis output*.
        # Ensure we have enough samples to pick 2*Q
        if num_augmented_samples_per_video < 2 * Q: 
            raise ValueError(f"Not enough augmented samples ({num_augmented_samples_per_video}) "
                             f"to pick 2*Q ({2*Q}). Adjust mock data or Q.")

        # `random.sample(range(num_augmented_samples_per_video - 1), 2 * Q)` from original
        # Let's adjust range to match actual available indices `num_augmented_samples_per_video`
        rand_subs_selected = random.sample(range(num_augmented_samples_per_video), 2 * Q)
        rand_subs_selected.sort() # Original code does this `rand_subs.sort()`
        
        # The original code's splitting: `rand_subs1 = rand_subs[Q:]` and `rand_subs2 = rand_subs[:Q]`
        # This is a specific way of splitting the 2*Q selected indices into two views.
        # This means `rand_subs1` contains the second half of selected indices, and `rand_subs2` the first half.
        rand_subs1_view_A = rand_subs_selected[Q:] # Q samples
        rand_subs2_view_B = rand_subs_selected[:Q] # Q samples

        # Get the actual group samples for View A and View B
        group_1_view_data = augmented_video_signals_raw[rand_subs1_view_A] # (Q, F, C, T)
        group_2_view_data = augmented_video_signals_raw[rand_subs2_view_B] # (Q, F, C, T)
        
        # Append to the list that will be concatenated for the model input
        groups_for_model_input.append(group_1_view_data)
        groups_for_model_input.append(group_2_view_data)

    # Concatenate all (P_clips * 2) views into one tensor
    # Resulting shape: (2 * P_clips * Q, F, C, T)
    return np.concatenate(groups_for_model_input, axis=0)