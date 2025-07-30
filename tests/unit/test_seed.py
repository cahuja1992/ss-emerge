import pytest
import numpy as np
import torch
import os
import tempfile
import shutil

from ss_emerge.datasets.seed_dataset import SEEDDataset

NUM_SAMPLES = 20
NUM_CHANNELS = 62
TIME_POINTS = 200
SFREQ = 200
NUM_CLASSES = 3
NUM_FREQ_BANDS = 5

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}

@pytest.fixture(scope="module")
def mock_seed_data_files():
    """Fixture to create temporary .npy files for SEED dataset."""
    temp_dir = tempfile.mkdtemp()
    data_path = os.path.join(temp_dir, "x_train_SEED.npy")
    labels_path = os.path.join(temp_dir, "y_train_SEED.npy")

    mock_eeg_data = np.random.randn(NUM_SAMPLES, 1, NUM_CHANNELS, TIME_POINTS).astype(np.float32)
    mock_labels = np.random.randint(0, NUM_CLASSES, (NUM_SAMPLES,)).astype(np.int64)

    np.save(data_path, mock_eeg_data)
    np.save(labels_path, mock_labels)

    yield data_path, labels_path

    shutil.rmtree(temp_dir)

def test_seed_dataset_initialization(mock_seed_data_files):
    """Test SEEDDataset initialization."""
    data_path, labels_path = mock_seed_data_files
    dataset = SEEDDataset(data_path, labels_path, sfreq=SFREQ, bands=BANDS)

    assert len(dataset) == NUM_SAMPLES
    assert hasattr(dataset, 'processed_features')
    assert hasattr(dataset, 'labels')
    assert dataset.processed_features.shape[0] == NUM_SAMPLES
    assert dataset.labels.shape[0] == NUM_SAMPLES
    assert dataset.processed_features.shape[1:] == (NUM_FREQ_BANDS, NUM_CHANNELS)

def test_seed_dataset_getitem_output_shape(mock_seed_data_files):
    """Test __getitem__ output shape."""
    data_path, labels_path = mock_seed_data_files
    dataset = SEEDDataset(data_path, labels_path, sfreq=SFREQ, bands=BANDS)

    features, label = dataset[0]

    assert isinstance(features, torch.Tensor)
    assert features.shape == (NUM_FREQ_BANDS, NUM_CHANNELS, 1)
    assert features.dtype == torch.float32

    assert isinstance(label, torch.Tensor)
    assert label.ndim == 0
    assert label.dtype == torch.long
    assert 0 <= label.item() < NUM_CLASSES

def test_seed_dataset_de_calculation_integrity(mock_seed_data_files):
    """Test DE feature calculation integrity."""
    data_path, labels_path = mock_seed_data_files
    dataset = SEEDDataset(data_path, labels_path, bands=BANDS, sfreq=SFREQ)

    features, _ = dataset[0]

    assert not torch.all(features == 0)
    assert torch.isfinite(features).all()