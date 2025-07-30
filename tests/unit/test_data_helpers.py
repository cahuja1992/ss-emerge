import numpy as np
import pytest
from ss_emerge.utils.data_helpers import calculate_de
from ss_emerge.utils.data_helpers import group_sample_subjects

def test_group_sample_subjects_output_properties():
    """Test group_sample_subjects output properties and subject uniqueness."""
    ss_total_samples_per_video = 45
    Q = 2

    processed_rand_subs_stre = group_sample_subjects(
        4, ss_total_samples_per_video, Q
    )
    
    assert isinstance(processed_rand_subs_stre, list)
    assert len(processed_rand_subs_stre) == ss_total_samples_per_video
    assert sorted(processed_rand_subs_stre) == sorted(list(range(ss_total_samples_per_video)))

    num_recomb_pairs_seed = 22 
    
    for i in range(num_recomb_pairs_seed):
        si = processed_rand_subs_stre[i]
        sj = processed_rand_subs_stre[i + Q]
        
        assert (si // 3) != (sj // 3), \
            f"Subject uniqueness violation found: {si} (Subject {si//3}) and {sj} (Subject {sj//3}) at indices {i}, {i+Q}"