import pytest
import numpy as np
import torch
import os
import tempfile
import shutil

from ss_emerge.datasets.seed_iv_dataset import SEED_IVDataset

NUM_SAMPLES_IV = 20
NUM_CHANNELS_IV = 62
TIME_POINTS_IV = 200
SFREQ_IV = 200
NUM_CLASSES_IV = 4
NUM_FREQ_BANDS_IV = 5

BANDS_IV = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}

@pytest.fixture(scope="module")
def mock_seed_iv_data_files():
    """Fixture to create temporary .npy files for SEED-IV dataset."""
    temp_dir = tempfile.mkdtemp()
    data_path = os.path.join(temp_dir, "x_train_SEED_IV.npy")
    labels_path = os.path.join(temp_dir, "y_train_SEED_IV.npy")

    mock_eeg_data = np.random.randn(NUM_SAMPLES_IV, 1, NUM_CHANNELS_IV, TIME_POINTS_IV).astype(np.float32)
    mock_labels = np.random.randint(0, NUM_CLASSES_IV, (NUM_SAMPLES_IV,)).astype(np.int64)

    np.save(data_path, mock_eeg_data)
    np.save(labels_path, mock_labels)

    yield data_path, labels_path

    shutil.rmtree(temp_dir)

def test_seed_iv_dataset_initialization(mock_seed_iv_data_files):
    """Test SEED_IVDataset initialization."""
    data_path, labels_path = mock_seed_iv_data_files
    dataset = SEED_IVDataset(data_path, labels_path, sfreq=SFREQ_IV, bands=BANDS_IV)

    assert len(dataset) == NUM_SAMPLES_IV
    assert hasattr(dataset, 'processed_features')
    assert hasattr(dataset, 'labels')
    assert dataset.processed_features.shape[0] == NUM_SAMPLES_IV
    assert dataset.labels.shape[0] == NUM_SAMPLES_IV
    assert dataset.processed_features.shape[1:] == (NUM_FREQ_BANDS_IV, NUM_CHANNELS_IV)

def test_seed_iv_dataset_getitem_output_shape(mock_seed_iv_data_files):
    """Test __getitem__ output shape."""
    data_path, labels_path = mock_seed_iv_data_files
    dataset = SEED_IVDataset(data_path, labels_path, sfreq=SFREQ_IV, bands=BANDS_IV)

    features, label = dataset[0]

    assert isinstance(features, torch.Tensor)
    assert features.shape == (NUM_FREQ_BANDS_IV, NUM_CHANNELS_IV, 1)
    assert features.dtype == torch.float32

    assert isinstance(label, torch.Tensor)
    assert label.ndim == 0
    assert label.dtype == torch.long
    assert 0 <= label.item() < NUM_CLASSES_IV

def test_seed_iv_dataset_de_calculation_integrity(mock_seed_iv_data_files):
    """Test DE feature calculation integrity."""
    data_path, labels_path = mock_seed_iv_data_files
    dataset = SEED_IVDataset(data_path, labels_path, sfreq=SFREQ_IV, bands=BANDS_IV)

    features, _ = dataset[0]

    assert not torch.all(features == 0)
    assert torch.isfinite(features).all()