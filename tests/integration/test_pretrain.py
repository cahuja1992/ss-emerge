import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import all necessary components from our library
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.projection_head import ProjectionHead
from ss_emerge.models.contrastive_loss import NTXentLoss
from ss_emerge.augmentations.meiosis import group_samples_for_contrastive_learning
from ss_emerge.utils.data_helpers import get_group_representations # For pooling after encoder

# Mock dataset parameters for integration test
BATCH_SIZE_VIDEOS = 2 # P in thesis
NUM_SUBJECTS_PER_VIDEO_GROUP = 45 # ss in original SEED data context for Meiosis (15 subjects * 3 trials)
NUM_CHANNELS = 62
TIME_POINTS = 200 # For MEIOSIS split point
NUM_FREQ_BANDS = 5 # F in SS-EMERGE input (DE features)
Q_SAMPLES_PER_AUGMENTED_VIEW = 2 # Q in thesis

# Encoder parameters (matching those in SS_EMERGE_Encoder tests)
D_SPECTRAL = 128
GAT_OUT_CHANNELS = 256
TCN_CHANNELS = [512, 512]
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2]
FINAL_ENCODER_DIM = 512 # Output of encoder after global pooling

# Projection Head parameters
PROJECTION_HEAD_HIDDEN_FEATURES = [1024, 2048, 4096]
PROJECTION_HEAD_FINAL_DIM = 4096 # Final output of projection head

# Contrastive Loss parameters
TEMPERATURE = 0.1

class MockDataset(TensorDataset):
    """
    A simple mock dataset for integration testing.
    Generates dummy data of shape (P, ss, F, C, T) to simulate the output of dataset processing.
    The labels are also dummy as they're not used in pretraining directly.
    """
    def __init__(self, num_samples, batch_size_videos, num_subjects_per_video_group, 
                 num_freq_bands, num_channels, time_points):
        self.data = torch.randn(
            num_samples * batch_size_videos, # Total 'video clips' to ensure enough data
            num_subjects_per_video_group, 
            num_freq_bands, 
            num_channels, 
            time_points
        )
        self.labels = torch.randint(0, 3, (num_samples * batch_size_videos,)) # Dummy labels
        super().__init__(self.data, self.labels)

def setup_pretraining_components():
    """Helper to instantiate model components for pretraining tests."""
    encoder = SS_EMERGE_Encoder(
        F_bands=NUM_FREQ_BANDS,
        D_spectral=D_SPECTRAL,
        C_channels=NUM_CHANNELS,
        T_timesteps=TIME_POINTS,
        gat_out_channels=GAT_OUT_CHANNELS,
        tcn_channels=TCN_CHANNELS,
        tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dilations=TCN_DILATIONS,
        dropout_prob=0.0 # Disable dropout for deterministic testing
    )
    projection_head = ProjectionHead(
        in_features=FINAL_ENCODER_DIM,
        hidden_features=PROJECTION_HEAD_HIDDEN_FEATURES,
        dropout_prob=0.0 # Disable dropout for deterministic testing
    )
    contrastive_loss_fn = NTXentLoss(temperature=TEMPERATURE)

    return encoder, projection_head, contrastive_loss_fn

# Mock edge indices for GATs (simplified for integration test)
# In a real scenario, these would come from the dataset/graph_helpers.
MOCK_SPATIAL_EDGE_INDEX = torch.randint(0, NUM_CHANNELS, (2, 100), dtype=torch.long)
MOCK_TEMPORAL_EDGE_INDEX = torch.stack([torch.arange(TIME_POINTS - 1), torch.arange(1, TIME_POINTS)], dim=0)


# --- Integration Tests for Pretraining Loop ---

def test_pretraining_single_step():
    """
    Test a single forward-backward pass in the pretraining loop.
    Ensure loss is calculated and model weights change.
    """
    encoder, projection_head, contrastive_loss_fn = setup_pretraining_components()
    
    # Combined model for pretraining
    class PretrainModel(nn.Module):
        def __init__(self, encoder, projection_head):
            super().__init__()
            self.encoder = encoder
            self.projection_head = projection_head
        
        def forward(self, x, spatial_edge_index, temporal_edge_index):
            embeddings = self.encoder(x, spatial_edge_index, temporal_edge_index)
            projected_embeddings = self.projection_head(embeddings)
            return projected_embeddings

    model = PretrainModel(encoder, projection_head)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Mock a batch from the DataLoader: (P, ss, F, C, T), (P) labels
    mock_batch_data = torch.randn(
        BATCH_SIZE_VIDEOS, NUM_SUBJECTS_PER_VIDEO_GROUP, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS
    )
    
    # Get initial model weights for comparison
    initial_weights = [p.clone() for p in model.parameters()]

    # Simulate one training step
    model.train() # Important for BatchNorm/Dropout (though disabled here for determinism)
    optimizer.zero_grad()

    # Step 1: Augment the batch (using the augmentations module)
    # Ensure data is on CPU before converting to numpy
    augmented_eeg_signals = group_samples_for_contrastive_learning(
        mock_batch_data.cpu().numpy(), # Explicitly move to CPU before .numpy()
        Q_SAMPLES_PER_AUGMENTED_VIEW
    )
    augmented_eeg_signals = torch.tensor(augmented_eeg_signals, dtype=torch.float32).to(mock_batch_data.device) # Move back to original device


    # Step 2: Pass through Encoder and Projection Head
    projected_embeddings = model(
        augmented_eeg_signals, 
        MOCK_SPATIAL_EDGE_INDEX, # These are global, no .to(device) needed here in test
        MOCK_TEMPORAL_EDGE_INDEX # (assuming they are already on the correct device if needed)
    ) 

    # Step 3: Get group-level representations and calculate loss
    group_representations = get_group_representations(
        projected_embeddings, Q_SAMPLES_PER_AUGMENTED_VIEW
    ) 

    # Split group_representations into two views (A and B) for NTXentLoss
    num_views_per_type = BATCH_SIZE_VIDEOS 
    z_i_views = group_representations[:num_views_per_type]
    z_j_views = group_representations[num_views_per_type:]

    # Calculate loss
    loss = contrastive_loss_fn(z_i_views, z_j_views)

    # Ensure loss is a scalar and not NaN/Inf
    assert loss.ndim == 0
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()

    # Perform backward pass and optimizer step
    loss.backward()
    optimizer.step()

    # Check if weights have changed after one step (basic sanity check)
    weights_changed = False
    for i, p in enumerate(model.parameters()):
        # Check if gradient exists and if the parameter value has changed
        if p.grad is not None and not torch.equal(p.data, initial_weights[i].data):
            weights_changed = True
            break
    assert weights_changed, "Model weights did not change after one training step."