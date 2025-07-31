import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random 
from sklearn.model_selection import train_test_split 

class SEEDDataset(Dataset):
    """
    Dataset class for the SEED EEG emotion dataset.
    Loads pre-processed numpy arrays containing Differential Entropy (DE) features.
    Supports loading data at video-clip level (for pretraining) or segment level (for finetuning).
    """
    def __init__(self, data_root, is_train=True, mode='finetune', # 'pretrain' or 'finetune'
                 sfreq=200, bands=None, return_de_features=True, 
                 label_proportion=1.0, random_seed=42):
        """
        Args:
            data_root (str): Root directory where processed data is saved (e.g., './data/SEED/').
            is_train (bool): Whether this is the training set.
            mode (str): 'pretrain' to load video-level data, 'finetune' to load segment-level data.
            sfreq (int): Original sampling frequency of the EEG data (for context).
            bands (dict): Dictionary defining frequency bands (not directly used for DE calc here).
            return_de_features (bool): If True, assumes data_path contains DE features.
            label_proportion (float): Proportion of labels to use (for few-shot learning).
            random_seed (int): Seed for label subsetting.
        """
        self.mode = mode
        
        if self.mode == 'pretrain':
            data_file = 'x_train_SEED_videos.npy' if is_train else 'x_test_SEED_videos.npy'
            labels_file = 'y_train_SEED_videos.npy' if is_train else 'y_test_SEED_videos.npy'
        elif self.mode == 'finetune':
            data_file = 'x_train_SEED_segments.npy' if is_train else 'x_test_SEED_segments.npy'
            labels_file = 'y_train_SEED_segments.npy' if is_train else 'y_test_SEED_segments.npy'
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'pretrain' or 'finetune'.")

        data_path = os.path.join(data_root, data_file)
        labels_path = os.path.join(data_root, labels_file)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at: {labels_path}")

        self.eeg_data = np.load(data_path, allow_pickle=True) # allow_pickle=True for video-level (ragged)
        self.labels = np.load(labels_path)

        # Ensure labels are 1D
        if self.labels.ndim > 1:
            self.labels = self.labels.reshape(-1)

        if len(self.eeg_data) != len(self.labels):
            raise ValueError(f"Mismatched number of samples/video_clips: data ({len(self.eeg_data)}) vs labels ({len(self.labels)})")

        self.sfreq = sfreq 
        self.bands = bands
        self.return_de_features = return_de_features # Should be True for SS-EMERGE model
        
        # --- Label Subsetting for Few-Shot Learning (only applies to training data in 'finetune' mode) ---
        if is_train and self.mode == 'finetune' and label_proportion < 1.0:
            print(f"Applying label proportion {label_proportion*100}% for training segments.")
            
            temp_labels_for_stratify = self.labels
            if self.labels.ndim > 1 and self.labels.shape[1] > 1: 
                temp_labels_for_stratify = np.argmax(self.labels, axis=1)

            original_indices = np.arange(len(self.eeg_data))
            _, subset_indices, _, _ = train_test_split(
                original_indices, temp_labels_for_stratify, 
                test_size=(1.0 - label_proportion), 
                random_state=random_seed, 
                stratify=temp_labels_for_stratify
            )
            
            self.eeg_data = self.eeg_data[subset_indices]
            self.labels = self.labels[subset_indices]
            print(f"  Reduced training segments to {len(self.eeg_data)}.")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # If mode is 'pretrain': returns (num_subjects_per_video_group, num_bands, num_channels, num_time_windows_de)
        # If mode is 'finetune': returns (num_bands, num_channels, num_time_windows_de)
        features = self.eeg_data[idx] 
        label = self.labels[idx]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)