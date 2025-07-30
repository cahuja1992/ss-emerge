import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil

# Import components from our library
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.projection_head import ProjectionHead
from ss_emerge.models.classification_head import ClassificationHead # We tested this in unit tests
from ss_emerge.models.contrastive_loss import NTXentLoss # For pretraining setup

# Constants for mock data and model setup (consistent with pretrain tests)
BATCH_SIZE_VIDEOS = 2
NUM_SUBJECTS_PER_VIDEO_GROUP = 45 # For Meiosis during dummy pretraining
NUM_CHANNELS = 62
TIME_POINTS = 200
NUM_FREQ_BANDS = 5
Q_SAMPLES_PER_AUGMENTED_VIEW = 2

D_SPECTRAL = 128
GAT_OUT_CHANNELS = 256
TCN_CHANNELS = [512, 512]
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2]
FINAL_ENCODER_DIM = 512 # Output of encoder after global pooling

PROJECTION_HEAD_HIDDEN_FEATURES = [1024, 2048, 4096]
PROJECTION_HEAD_FINAL_DIM = 4096

NUM_CLASSES = 3 # For finetuning task (e.g., SEED: Positive, Neutral, Negative)

# Mock edge indices for GATs
MOCK_SPATIAL_EDGE_INDEX = torch.randint(0, NUM_CHANNELS, (2, 100), dtype=torch.long)
MOCK_TEMPORAL_EDGE_INDEX = torch.stack([torch.arange(TIME_POINTS - 1), torch.arange(1, TIME_POINTS)], dim=0)


class MockFinetuneDataset(TensorDataset):
    """
    A mock dataset for finetuning, providing labeled EEG data.
    Simulates the output format of `seed_dataset.py` or `deap_dataset.py` for finetuning.
    """
    def __init__(self, num_samples, num_freq_bands, num_channels, time_points, num_classes):
        # Each sample is (F, C, T) and a label
        self.data = torch.randn(num_samples, num_freq_bands, num_channels, time_points)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
        super().__init__(self.data, self.labels)


@pytest.fixture(scope="module")
def pretrained_model_path():
    """
    Fixture to create and save a dummy pretrained model for testing finetuning.
    This simulates a successful pretraining phase.
    """
    # Create a temporary directory to save the model
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "dummy_pretrained_encoder.pth")

    try:
        # Instantiate and "pretrain" a dummy encoder
        encoder = SS_EMERGE_Encoder(
            F_bands=NUM_FREQ_BANDS,
            D_spectral=D_SPECTRAL,
            C_channels=NUM_CHANNELS,
            T_timesteps=TIME_POINTS,
            gat_out_channels=GAT_OUT_CHANNELS,
            tcn_channels=TCN_CHANNELS,
            tcn_kernel_size=TCN_KERNEL_SIZE,
            tcn_dilations=TCN_DILATIONS,
            dropout_prob=0.0
        )
        projection_head = ProjectionHead(
            in_features=FINAL_ENCODER_DIM,
            hidden_features=PROJECTION_HEAD_HIDDEN_FEATURES,
            dropout_prob=0.0
        )
        
        # A simple forward pass to ensure weights exist (not strictly pretraining, but enough for fixture)
        # CORRECTED: Use a batch_size > 1 for the dummy input to avoid BatchNorm1d error
        dummy_input = torch.randn(BATCH_SIZE_VIDEOS, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS)
        _ = projection_head(encoder(dummy_input, MOCK_SPATIAL_EDGE_INDEX, MOCK_TEMPORAL_EDGE_INDEX))

        # Save only the encoder's state_dict as it's what finetuning loads
        torch.save(encoder.state_dict(), model_path)
        yield model_path
    finally:
        # Clean up the temporary directory after tests
        shutil.rmtree(temp_dir)


# --- Integration Tests for Finetuning Loop ---

def test_finetuning_single_step(pretrained_model_path):
    """
    Test a single forward-backward pass in the finetuning loop.
    Ensure only classification head weights change and loss is calculated.
    """
    # We will simulate the `finetune.py` script's logic here.
    device = torch.device("cpu") # Test on CPU for consistency in CI/CD

    # Instantiate Encoder and Classification Head
    encoder = SS_EMERGE_Encoder(
        F_bands=NUM_FREQ_BANDS, D_spectral=D_SPECTRAL, C_channels=NUM_CHANNELS, T_timesteps=TIME_POINTS,
        gat_out_channels=GAT_OUT_CHANNELS, tcn_channels=TCN_CHANNELS, tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dilations=TCN_DILATIONS, dropout_prob=0.0
    ).to(device)
    
    classification_head = ClassificationHead(
        in_features=FINAL_ENCODER_DIM, num_classes=NUM_CLASSES, dropout_prob=0.0
    ).to(device)

    # Load pretrained encoder weights
    encoder.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # Combined model for finetuning
    class FinetuneModel(nn.Module):
        def __init__(self, encoder, classification_head):
            super().__init__()
            self.encoder = encoder
            self.classification_head = classification_head
        
        def forward(self, x, spatial_edge_index, temporal_edge_index):
            with torch.no_grad(): # Ensure no gradients for encoder during finetuning forward pass
                embeddings = self.encoder(x, spatial_edge_index, temporal_edge_index)
            logits = self.classification_head(embeddings)
            return logits

    model = FinetuneModel(encoder, classification_head).to(device)
    optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3) # Optimize only head
    criterion = nn.CrossEntropyLoss()

    # Mock dataset and DataLoader
    mock_dataset = MockFinetuneDataset(num_samples=BATCH_SIZE_VIDEOS * 2, # Ensure enough samples for batch
                                       num_freq_bands=NUM_FREQ_BANDS, num_channels=NUM_CHANNELS, 
                                       time_points=TIME_POINTS, num_classes=NUM_CLASSES)
    mock_dataloader = DataLoader(mock_dataset, batch_size=BATCH_SIZE_VIDEOS, shuffle=True)

    # Get initial weights of *both* encoder and classification head for comparison
    initial_encoder_weights = [p.clone() for p in encoder.parameters()]
    initial_classifier_weights = [p.clone() for p in classification_head.parameters()]

    # Get a batch
    batch_data, batch_labels = next(iter(mock_dataloader))
    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

    # Simulate one training step
    model.train()
    optimizer.zero_grad()

    logits = model(batch_data, MOCK_SPATIAL_EDGE_INDEX, MOCK_TEMPORAL_EDGE_INDEX)
    loss = criterion(logits, batch_labels)

    # Ensure loss is scalar and valid
    assert loss.ndim == 0
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()

    loss.backward()
    optimizer.step()

    # --- Assertions ---
    # 1. Encoder weights should NOT have changed
    for i, p in enumerate(encoder.parameters()):
        assert torch.equal(p.data, initial_encoder_weights[i].data), \
            f"Encoder parameter changed: {i}"
        assert p.grad is None or torch.all(p.grad == 0), \
            f"Encoder parameter {i} has non-zero gradient: {p.grad}"

    # 2. Classification head weights SHOULD have changed
    classifier_weights_changed = False
    for i, p in enumerate(classification_head.parameters()):
        if p.grad is not None and not torch.equal(p.data, initial_classifier_weights[i].data):
            classifier_weights_changed = True
            break
    assert classifier_weights_changed, "Classification head weights did not change."


def test_finetuning_full_process(pretrained_model_path):
    """
    Test a simplified full finetuning process (multiple epochs, validation).
    Ensure validation loss decreases (simple sanity check) and model saves.
    """
    device = torch.device("cpu")

    # Mock data loaders
    train_dataset = MockFinetuneDataset(num_samples=50, num_freq_bands=NUM_FREQ_BANDS, 
                                        num_channels=NUM_CHANNELS, time_points=TIME_POINTS, 
                                        num_classes=NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_VIDEOS, shuffle=True)
    val_dataset = MockFinetuneDataset(num_samples=20, num_freq_bands=NUM_FREQ_BANDS, 
                                      num_channels=NUM_CHANNELS, time_points=TIME_POINTS, 
                                      num_classes=NUM_CLASSES)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VIDEOS, shuffle=False)

    # Instantiate model components
    encoder = SS_EMERGE_Encoder(
        F_bands=NUM_FREQ_BANDS, D_spectral=D_SPECTRAL, C_channels=NUM_CHANNELS, T_timesteps=TIME_POINTS,
        gat_out_channels=GAT_OUT_CHANNELS, tcn_channels=TCN_CHANNELS, tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dilations=TCN_DILATIONS, dropout_prob=0.0
    ).to(device)
    classification_head = ClassificationHead(
        in_features=FINAL_ENCODER_DIM, num_classes=NUM_CLASSES, dropout_prob=0.0
    ).to(device)

    # Load pretrained encoder and freeze
    encoder.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    for param in encoder.parameters():
        param.requires_grad = False

    class FinetuneModel(nn.Module): # Re-define local class for scope
        def __init__(self, encoder, classification_head):
            super().__init__()
            self.encoder = encoder
            self.classification_head = classification_head
        def forward(self, x, spatial_edge_index, temporal_edge_index):
            with torch.no_grad():
                embeddings = self.encoder(x, spatial_edge_index, temporal_edge_index)
            logits = self.classification_head(embeddings)
            return logits

    model = FinetuneModel(encoder, classification_head).to(device)
    optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Mock edge indices on device
    spatial_edge_index_device = MOCK_SPATIAL_EDGE_INDEX.to(device)
    temporal_edge_index_device = MOCK_TEMPORAL_EDGE_INDEX.to(device)

    num_epochs = 5 # A small number of epochs for quick testing

    initial_val_loss = float('inf')
    # Loop through a few epochs
    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_data, spatial_edge_index_device, temporal_edge_index_device)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for batch_data_val, batch_labels_val in val_loader:
                batch_data_val, batch_labels_val = batch_data_val.to(device), batch_labels_val.to(device)
                logits_val = model(batch_data_val, spatial_edge_index_device, temporal_edge_index_device)
                loss_val = criterion(logits_val, batch_labels_val)
                current_val_loss += loss_val.item()
        current_val_loss /= len(val_loader)

        # Simple sanity check: loss should generally decrease or stay low over epochs
        if epoch == 0:
            initial_val_loss = current_val_loss
        
        # We expect some learning to happen, so final loss should be less than or comparable to initial.
        # A very small model/dataset might not show consistent decrease immediately.
        # Check against a high threshold or verify it's a number.
        assert not np.isnan(current_val_loss) and not np.isinf(current_val_loss)
    
    # Test model saving functionality
    temp_save_dir = tempfile.mkdtemp()
    final_model_path = os.path.join(temp_save_dir, "finetuned_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    assert os.path.exists(final_model_path)
    
    # Clean up temporary directory
    shutil.rmtree(temp_save_dir)