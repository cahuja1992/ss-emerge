import numpy as np
import torch
from torch.utils.data import Dataset
import os

# Import the DE calculation utility
from ss_emerge.utils.data_helpers import calculate_de

class SEED_IVDataset(Dataset):
    """
    Dataset class for the SEED-IV EEG emotion dataset.
    Loads preprocessed numpy arrays (raw EEG segments) and extracts Differential Entropy (DE) features.
    """
    def __init__(self, data_path, labels_path, sfreq=200, is_train=True, 
                 bands={'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)}):
        """
        Args:
            data_path (str): Path to the .npy file containing EEG data (e.g., 'x_train_SEED_IV.npy').
                             Expected shape after loading: (num_samples, 1, num_channels, time_points)
                             or (num_samples, num_channels, time_points).
            labels_path (str): Path to the .npy file containing labels (e.g., 'y_train_SEED_IV.npy').
            sfreq (int): Sampling frequency of the EEG data.
            is_train (bool): Whether this is the training set (for potential future augmentation differences).
            bands (dict): Dictionary defining frequency bands for DE calculation.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at: {labels_path}")

        self.eeg_data = np.load(data_path)
        self.labels = np.load(labels_path)

        # Ensure labels are 1D (num_samples,)
        if self.labels.ndim > 1:
            self.labels = self.labels.reshape(-1) # Flatten if (num_samples, 1) or similar

        # Adjust data shape if it's (num_samples, 1, num_channels, time_points) -> (num_samples, num_channels, time_points)
        # as calculate_de expects (channels, time_points)
        if self.eeg_data.ndim == 4 and self.eeg_data.shape[1] == 1:
            self.eeg_data = self.eeg_data.squeeze(1) # Remove the dummy frequency/feature dimension

        # Verify consistency
        if len(self.eeg_data) != len(self.labels):
            raise ValueError(f"Mismatched number of samples: data ({len(self.eeg_data)}) vs labels ({len(self.labels)})")

        self.sfreq = sfreq
        self.bands = bands
        self.num_bands = len(bands)
        self.is_train = is_train

        # Pre-calculate DE features for all samples
        print(f"Loading and processing {len(self.eeg_data)} samples for DE features for SEED-IV. This may take a moment...")
        processed_features = []
        for i, sample_data in enumerate(self.eeg_data):
            # sample_data is (num_channels, time_points)
            de_features_sample = calculate_de(sample_data, self.sfreq, self.bands)
            processed_features.append(de_features_sample)
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(self.eeg_data)} samples for SEED-IV.")
        self.processed_features = np.array(processed_features, dtype=np.float32)
        print("DE feature processing complete for SEED-IV.")
        
    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # Return DE features for the segment and its label
        # self.processed_features is (num_samples, num_bands, num_channels)
        # We need to return (num_bands, num_channels, 1) for the model's T dimension.
        features = self.processed_features[idx] # (num_bands, num_channels)
        # Add a dummy time dimension of size 1, so it becomes (num_bands, num_channels, 1)
        features = np.expand_dims(features, axis=-1) 
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)