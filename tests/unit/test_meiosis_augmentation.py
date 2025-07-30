import numpy as np
import pytest
import torch

from ss_emerge.augmentations.meiosis import Meiosis 
from ss_emerge.augmentations.meiosis import group_samples_for_contrastive_learning
from ss_emerge.utils.data_helpers import get_group_representations 

NUM_SUBJECTS_PER_GROUP = 40
NUM_CHANNELS = 62
TIME_POINTS = 200
Q_SAMPLES_PER_AUGMENTED_VIEW = 2
NUM_FREQ_BANDS = 5

def test_meiosis_output_shape():
    """Test Meiosis output shape."""
    signal = np.random.rand(NUM_SUBJECTS_PER_GROUP, 1, NUM_CHANNELS, TIME_POINTS)
    rand_subs_stre = list(range(NUM_SUBJECTS_PER_GROUP))
    np.random.shuffle(rand_subs_stre) 
    split = TIME_POINTS // 2
    augmented_signal = Meiosis(signal, Q_SAMPLES_PER_AUGMENTED_VIEW, rand_subs_stre, split)
    expected_num_samples = 2 * 16 + 1
    assert augmented_signal.shape[0] == expected_num_samples
    assert augmented_signal.shape[1:] == signal.shape[1:]

def test_meiosis_recombination_logic():
    """Test Meiosis recombination logic."""
    signal_len = TIME_POINTS
    split = signal_len // 2
    signal = np.array([np.full((1, NUM_CHANNELS, signal_len), i) for i in range(NUM_SUBJECTS_PER_GROUP)])
    
    si_idx_first_pair = 0 
    sj_idx_first_pair = 1 
    
    mock_rand_subs_stre = list(range(NUM_SUBJECTS_PER_GROUP))
    mock_rand_subs_stre[0], mock_rand_subs_stre[Q_SAMPLES_PER_AUGMENTED_VIEW] = si_idx_first_pair, sj_idx_first_pair
    
    augmented_signal = Meiosis(signal, Q_SAMPLES_PER_AUGMENTED_VIEW, mock_rand_subs_stre, split)

    index_of_xj = 16 
    assert np.all(augmented_signal[0, :, :, :split] == signal[si_idx_first_pair, :, :, :split])
    assert np.all(augmented_signal[0, :, :, split:] == signal[sj_idx_first_pair, :, :, split:])
    assert np.all(augmented_signal[index_of_xj, :, :, :split] == signal[sj_idx_first_pair, :, :, :split])
    assert np.all(augmented_signal[index_of_xj, :, :, split:] == signal[si_idx_first_pair, :, :, split:])

def test_meiosis_edge_split_point():
    """Test Meiosis with split points near edges."""
    signal = np.random.rand(NUM_SUBJECTS_PER_GROUP, 1, NUM_CHANNELS, TIME_POINTS)
    rand_subs_stre = list(range(NUM_SUBJECTS_PER_GROUP))
    np.random.shuffle(rand_subs_stre)

    split_early = 1
    augmented_signal_early = Meiosis(signal, Q_SAMPLES_PER_AUGMENTED_VIEW, rand_subs_stre, split_early)
    assert augmented_signal_early.shape[0] == (2 * 16 + 1)

    split_late = TIME_POINTS - 1
    augmented_signal_late = Meiosis(signal, Q_SAMPLES_PER_AUGMENTED_VIEW, rand_subs_stre, split_late)
    assert augmented_signal_late.shape[0] == (2 * 16 + 1)

def test_meiosis_input_type():
    """Ensure Meiosis handles numpy array inputs."""
    signal = np.random.rand(NUM_SUBJECTS_PER_GROUP, 1, NUM_CHANNELS, TIME_POINTS)
    rand_subs_stre = list(range(NUM_SUBJECTS_PER_GROUP))
    np.random.shuffle(rand_subs_stre)
    split = TIME_POINTS // 2
    
    try:
        Meiosis(signal, Q_SAMPLES_PER_AUGMENTED_VIEW, rand_subs_stre, split)
    except Exception as e:
        pytest.fail(f"Meiosis raised an unexpected exception for valid numpy input: {e}")

NUM_VIDEO_CLIPS_IN_BATCH = 4
NUM_SUBJECTS_PER_VIDEO = 45

def test_group_samples_for_contrastive_learning_output_shape():
    """Test group_samples_for_contrastive_learning output shape."""
    mock_signal_batch = np.random.rand(
        NUM_VIDEO_CLIPS_IN_BATCH, NUM_SUBJECTS_PER_VIDEO, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS
    )

    augmented_groups = group_samples_for_contrastive_learning(
        mock_signal_batch, Q_SAMPLES_PER_AUGMENTED_VIEW
    )

    expected_total_samples = NUM_VIDEO_CLIPS_IN_BATCH * 2 * Q_SAMPLES_PER_AUGMENTED_VIEW

    assert augmented_groups.shape[0] == expected_total_samples
    assert augmented_groups.shape[1:] == (NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS)


def test_group_samples_for_contrastive_learning_subject_uniqueness_seed():
    """Test subject uniqueness logic within group sampling."""
    from ss_emerge.utils.data_helpers import group_sample_subjects

    processed_rand_subs_stre = group_sample_subjects(
        NUM_VIDEO_CLIPS_IN_BATCH, NUM_SUBJECTS_PER_VIDEO, Q_SAMPLES_PER_AUGMENTED_VIEW
    )
    
    assert isinstance(processed_rand_subs_stre, list)
    assert len(processed_rand_subs_stre) == NUM_SUBJECTS_PER_VIDEO
    assert sorted(processed_rand_subs_stre) == sorted(list(range(NUM_SUBJECTS_PER_VIDEO)))

    num_recomb_pairs_seed = 22
    
    max_idx_to_check = num_recomb_pairs_seed + Q_SAMPLES_PER_AUGMENTED_VIEW
    if max_idx_to_check > len(processed_rand_subs_stre):
        max_idx_to_check = len(processed_rand_subs_stre) - Q_SAMPLES_PER_AUGMENTED_VIEW
        num_recomb_pairs_seed = max_idx_to_check

    for i in range(num_recomb_pairs_seed):
        si = processed_rand_subs_stre[i]
        sj = processed_rand_subs_stre[i + Q_SAMPLES_PER_AUGMENTED_VIEW] 
        
        assert (si // 3) != (sj // 3), \
            f"Subject uniqueness violation found: {si} (Subject {si//3}) and {sj} (Subject {sj//3}) at indices {i}, {i+Q_SAMPLES_PER_AUGMENTED_VIEW}"

def test_get_group_representations_max_pool_logic():
    """Test get_group_representations MaxPool1D logic."""
    B_clips = 2
    Q_samples = 2
    D_embedding = 10

    embeddings_np = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0], 
        [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
        [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0], 
        [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],
        [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0], 
        [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0],
    ])
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)

    group_reps = get_group_representations(embeddings_tensor, Q_samples)

    expected_group_reps_np = np.array([
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], 
        [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0], 
        [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0], 
        [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0], 
    ])
    assert torch.allclose(group_reps, torch.tensor(expected_group_reps_np, dtype=torch.float32))