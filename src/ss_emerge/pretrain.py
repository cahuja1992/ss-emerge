import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import argparse
import yaml # Import PyYAML

# Import all necessary components from our library
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.projection_head import ProjectionHead
from ss_emerge.models.contrastive_loss import NTXentLoss
from ss_emerge.models.resnet import ResNetEEG
from ss_emerge.augmentations.meiosis import group_samples_for_contrastive_learning
from ss_emerge.utils.data_helpers import get_group_representations
from ss_emerge.utils.graph_helpers import get_spatial_edge_index, get_temporal_edge_index # To generate dynamic edge indices

# Import actual dataset classes
from ss_emerge.datasets.seed_dataset import SEEDDataset
from ss_emerge.datasets.seed_iv_dataset import SEED_IVDataset

# Define a placeholder for the dataset (will be replaced by actual SEED/DEAP datasets)
# Removed DummyDataset as we'll use actual datasets or a simplified mock for testing where needed.


def train_one_epoch(model, dataloader, optimizer, criterion, 
                    spatial_edge_index, temporal_edge_index, Q_samples):
    model.train()
    total_loss = 0
    for batch_idx, (batch_data, _) in enumerate(dataloader):
        # batch_data: (P_clips, ss_per_video, F, C, T)
        
        # Ensure batch_data is on the correct device for Meiosis conversion
        batch_data = batch_data.cpu() # Meiosis expects numpy

        optimizer.zero_grad()

        # Augment the batch using Meiosis
        augmented_eeg_signals_np = group_samples_for_contrastive_learning(
            batch_data.numpy(), Q_samples
        )
        # Move augmented data back to device after numpy conversion
        augmented_eeg_signals = torch.tensor(augmented_eeg_signals_np, dtype=torch.float32).to(model.encoder.parameters().__next__().device)

        # Pass through Encoder and Projection Head
        projected_embeddings = model(
            augmented_eeg_signals, 
            spatial_edge_index, 
            temporal_edge_index
        )
        
        # Get group-level representations for contrastive loss
        group_representations = get_group_representations(
            projected_embeddings, Q_samples
        )
        
        # Split into two views for NTXentLoss
        # This split assumes `group_samples_for_contrastive_learning` produces `(2 * P * Q, ...)`
        # and `get_group_representations` then produces `(2 * P, D)`.
        # So `num_video_clips_in_batch` is `P`.
        num_video_clips_in_batch = batch_data.shape[0] 
        z_i_views = group_representations[:num_video_clips_in_batch]
        z_j_views = group_representations[num_video_clips_in_batch:]

        # Calculate loss
        loss = criterion(z_i_views, z_j_views)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, 
                       spatial_edge_index, temporal_edge_index, Q_samples):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(dataloader):
            batch_data = batch_data.cpu() # Meiosis expects numpy

            augmented_eeg_signals_np = group_samples_for_contrastive_learning(
                batch_data.numpy(), Q_samples
            )
            augmented_eeg_signals = torch.tensor(augmented_eeg_signals_np, dtype=torch.float32).to(model.encoder.parameters().__next__().device)

            projected_embeddings = model(
                augmented_eeg_signals, 
                spatial_edge_index, 
                temporal_edge_index
            )
            
            group_representations = get_group_representations(
                projected_embeddings, Q_samples
            )
            
            num_video_clips_in_batch = batch_data.shape[0]
            z_i_views = group_representations[:num_video_clips_in_batch]
            z_j_views = group_representations[num_video_clips_in_batch:]

            loss = criterion(z_i_views, z_j_views)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract general settings
    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']
    
    # Extract pretraining settings
    pretrain_cfg = config['pretrain']
    epochs = pretrain_cfg['epochs']
    batch_size = pretrain_cfg['batch_size']
    Q_samples = pretrain_cfg['Q']
    temperature = pretrain_cfg['temperature']
    lr = pretrain_cfg['learning_rate']
    save_interval = pretrain_cfg['save_interval']
    pretrained_model_dir = pretrain_cfg['pretrained_model_dir']

    # Extract model architecture settings
    model_cfg = config['model']
    encoder_cfg = model_cfg['encoder']
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # --- Dataset Loading ---
    # Determine which dataset to load based on config
    DatasetClass = None
    if dataset_name == "SEED":
        DatasetClass = SEEDDataset
    elif dataset_name == "SEED_IV":
        DatasetClass = SEED_IVDataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Instantiate datasets using paths from config
    # Note: For actual data processing, you'd need the processed .npy files.
    # The `data_path` refers to the pre-saved numpy arrays from `Data_process` scripts.
    train_dataset = DatasetClass(
        data_path=os.path.join(data_root, config['train_data_path']),
        labels_path=os.path.join(data_root, config['train_labels_path']),
        sfreq=encoder_cfg['T_timesteps'], # Assuming T_timesteps is used as SFREQ for DE, adjust if actual SFREQ is different
        bands=config.get('bands', {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)})
    )
    val_dataset = DatasetClass( # Use same parameters for val set for simplicity in this general script
        data_path=os.path.join(data_root, config['test_data_path']), # Using test data as validation
        labels_path=os.path.join(data_root, config['test_labels_path']),
        sfreq=encoder_cfg['T_timesteps'], # Assuming T_timesteps is used as SFREQ for DE
        bands=config.get('bands', {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)})
    )

    # DataLoader expects a batch where each item is a video clip's full data (P, ss, F, C, T)
    # Our DatasetClass __getitem__ returns (F, C, 1) or (F, C, T) depending on the 'T' interpretation.
    # For Meiosis, it needs (P, ss, F, C, T). This means the Dataset needs to yield
    # a full video clip's data (ss subjects, F, C, T) per __getitem__ call, not just one segment.
    # This requires a significant change to how `SEEDDataset` works or how the DataLoader is set up.
    # The kanhaoning `DataLoader` batches `P` *video clips*, each containing `ss` samples.
    # So the dataset needs to be a `VideoClipDataset` not a `SegmentDataset`.

    # For now, let's adjust the DataLoader to expect `(P, ss, F, C, T)` from a `Dataset` that has `P` `__len__` items.
    # This implies the Dataset's `__getitem__` would return a batch of `ss` subjects' data for one video.
    # This is currently NOT how our `SEEDDataset` works. It returns 1 sample (F, C, 1).
    # Re-reading kanhaoning `SSL_training(SEED).py`: `train_loader = data.DataLoader(train_dataset, P, shuffle=True)`
    # where `train_dataset` is `data.TensorDataset(x_train, y_train)`.
    # `x_train` is `(num_video_clips, num_subjects_per_video_group, 1, channels, time_points)`.
    # So their `x_train` is already pre-batched by `video_clip`.

    # Let's adjust `src/pretrain.py`'s DataLoader to expect the `kanhaoning` format:
    # A Dataset that returns `(num_subjects_per_video_group, F, C, T)` for one `video_clip_idx`.
    # So `train_dataset` needs to be pre-processed data *per video clip*, not per segment.
    # This means `x_train_SEED.npy` should be `(num_video_clips, num_subjects_per_video_group, F, C, T)`.
    # Our `SEEDDataset` currently loads `(num_segments, F, C, 1)`

    # Temporary hack for DataLoader if Dataset is still `(segment_idx, F, C, 1)`:
    # We need to construct batches of (P, ss, F, C, T) from segment-level data.
    # This is a complex batching strategy, requiring a custom `collate_fn`.
    # For now, let's make `DummyDataset` return a mock `(ss, F, C, T)` to fit the `DataLoader`.
    
    # We will temporarily use a simplified `DummyPretrainDataset` to avoid overcomplicating until
    # we correctly adapt `SEEDDataset` for video-level batching.
    class DummyPretrainDataset(TensorDataset):
        def __init__(self, num_video_clips_total, num_subjects_per_video_group, 
                     num_freq_bands, num_channels, time_points):
            # Each item in the dataset is one video clip's data from all its subjects
            self.data = torch.randn(
                num_video_clips_total, 
                num_subjects_per_video_group, 
                num_freq_bands, 
                num_channels, 
                time_points
            )
            self.labels = torch.randint(0, 3, (num_video_clips_total,)) # Dummy labels for videos
            super().__init__(self.data, self.labels)

    # Adjust dataset instantiation for DummyPretrainDataset
    # The num_samples in config['num_samples'] means total video clips available
    num_video_clips_total = config.get('num_video_clips_total', 20) # Default for mock
    train_dataset = DummyPretrainDataset(
        num_video_clips_total, encoder_cfg['C_channels'], NUM_FREQ_BANDS, 
        encoder_cfg['C_channels'], encoder_cfg['T_timesteps']
    )
    val_dataset = DummyPretrainDataset(
        num_video_clips_total // 2, encoder_cfg['C_channels'], NUM_FREQ_BANDS, 
        encoder_cfg['C_channels'], encoder_cfg['T_timesteps']
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # --- Model Components ---
    encoder = SS_EMERGE_Encoder(
        F_bands=encoder_cfg['F_bands'], D_spectral=encoder_cfg['D_spectral'], 
        C_channels=encoder_cfg['C_channels'], T_timesteps=encoder_cfg['T_timesteps'],
        gat_out_channels=encoder_cfg['gat_out_channels'], 
        tcn_channels=encoder_cfg['tcn_channels'], 
        tcn_kernel_size=encoder_cfg['tcn_kernel_size'], 
        tcn_dilations=encoder_cfg['tcn_dilations'], 
        dropout_prob=encoder_cfg['dropout_prob']
    )
    projection_head = ProjectionHead(
        in_features=encoder.final_tcn_output_dim, # Get actual output dim from encoder
        hidden_features=model_cfg['projection_head']['hidden_features'],
        dropout_prob=model_cfg['projection_head']['dropout_prob']
    )
    contrastive_loss_fn = NTXentLoss(temperature=temperature)

    # Combined model for pretraining
    class PretrainModel(nn.Module):
        def __init__(self, encoder, projection_head):
            super().__init__()
            self.encoder = encoder
            self.projection_head = projection_head
        
        def forward(self, x_video_batch, spatial_edge_index, temporal_edge_index):
            # x_video_batch: (P_clips, ss_per_video, F, C, T)
            # The encoder expects (B_eff, F, C, T) where B_eff is total augmented samples.
            # This forward pass needs to handle the augmentation inside, or expect augmented.
            # Let's assume the `train_one_epoch` handles the augmentation and passes flat (B_eff, F, C, T)

            # This `forward` method should match the `model()` call in `train_one_epoch`.
            # `train_one_epoch` passes `augmented_eeg_signals` which is `(B_eff, F, C, T)`.
            # So, this `forward` just passes `x` (which is `augmented_eeg_signals`).
            embeddings = self.encoder(x_video_batch, spatial_edge_index, temporal_edge_index)
            projected_embeddings = self.projection_head(embeddings)
            return projected_embeddings

    model = PretrainModel(encoder, projection_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Edge indices (need to be tensors on device)
    # These should be generated dynamically based on actual dataset parameters if not fixed
    spatial_edge_index = get_spatial_edge_index(encoder_cfg['C_channels']).to(device)
    temporal_edge_index = get_temporal_edge_index(encoder_cfg['T_timesteps']).to(device)


    # Create save directory if it doesn't exist
    os.makedirs(pretrained_model_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting pretraining for {epochs} epochs (Experiment: {experiment_name})...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, contrastive_loss_fn, 
                                     spatial_edge_index, temporal_edge_index, Q_samples)
        val_loss = validate_one_epoch(model, val_loader, contrastive_loss_fn, 
                                      spatial_edge_index, temporal_edge_index, Q_samples)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model checkpoint (saving encoder.state_dict only)
        if (epoch + 1) % save_interval == 0:
            torch.save(model.encoder.state_dict(), os.path.join(pretrained_model_dir, f"{experiment_name}_encoder_epoch_{epoch+1}.pth"))
            print(f"Encoder model saved at epoch {epoch+1} to {pretrained_model_dir}")

    print("Pretraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS-EMERGE Self-Supervised Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use, -1 for CPU.")
    
    args = parser.parse_args()
    main(args)