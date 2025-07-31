import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
import argparse
import yaml

from ss_emerge.datasets.seed_dataset import SEEDDataset
from ss_emerge.datasets.seed_iv_dataset import SEED_IVDataset
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.projection_head import ProjectionHead
from ss_emerge.models.contrastive_loss import NTXentLoss
from ss_emerge.models.resnet import ResNetEEG
from ss_emerge.augmentations.meiosis import group_samples_for_contrastive_learning
from ss_emerge.utils.data_helpers import get_group_representations
from ss_emerge.utils.graph_helpers import get_spatial_edge_index, get_temporal_edge_index

NUM_FREQ_BANDS = 5 

ACTUAL_NUM_SUBJECTS_PER_VIDEO_GROUP_SEED = 45
ACTUAL_NUM_CHANNELS_SEED = 62 
ACTUAL_NUM_TIME_WINDOWS_DE_SEED = 265


def train_one_epoch(model, dataloader, optimizer, criterion, 
                    spatial_edge_index, temporal_edge_index, Q_samples):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    if num_batches == 0:
        print("Warning: Dataloader is empty. Skipping training epoch.")
        return 0.0

    for batch_idx, (batch_data, _) in enumerate(dataloader):
        print(f"  Batch {batch_idx+1}/{num_batches}: Processing data of shape {batch_data.shape}")
        
        batch_data = batch_data.cpu()
        optimizer.zero_grad()
        
        augmented_eeg_signals_np = group_samples_for_contrastive_learning(
            batch_data.numpy(), Q_samples
        )
        augmented_eeg_signals = torch.tensor(augmented_eeg_signals_np, dtype=torch.float32).to(model.encoder.parameters().__next__().device)
        print(f"  Batch {batch_idx+1}/{num_batches}: Augmented signal shape: {augmented_eeg_signals.shape}")

        projected_embeddings = model(
            augmented_eeg_signals, 
            spatial_edge_index, 
            temporal_edge_index
        )
        print(f"  Batch {batch_idx+1}/{num_batches}: Projected embeddings shape: {projected_embeddings.shape}")
        
        group_representations = get_group_representations(
            projected_embeddings, Q_samples
        )
        print(f"  Batch {batch_idx+1}/{num_batches}: Group representations shape: {group_representations.shape}")
        
        num_video_clips_in_batch = batch_data.shape[0] 
        z_i_views = group_representations[:num_video_clips_in_batch]
        z_j_views = group_representations[num_video_clips_in_batch:]

        loss = criterion(z_i_views, z_j_views)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  Batch {batch_idx+1}/{num_batches}: Loss = {loss.item():.4f}")

    return total_loss / num_batches


def validate_one_epoch(model, dataloader, criterion, 
                       spatial_edge_index, temporal_edge_index, Q_samples):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    if num_batches == 0:
        print("Warning: Validation dataloader is empty. Skipping validation epoch.")
        return 0.0

    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(dataloader):
            batch_data = batch_data.cpu()

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

    return total_loss / num_batches


def main(args):
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']
    
    pretrain_cfg = config['pretrain']
    epochs = pretrain_cfg['epochs']
    batch_size = pretrain_cfg['batch_size']
    Q_samples = pretrain_cfg['Q']
    temperature = pretrain_cfg['temperature']
    lr = pretrain_cfg['learning_rate']
    save_interval = pretrain_cfg['save_interval']
    pretrained_model_dir = pretrain_cfg['pretrained_model_dir']

    model_cfg = config['model']
    encoder_cfg = model_cfg['encoder']
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # --- Dataset Loading ---
    DatasetClass = None
    if dataset_name == "SEED":
        DatasetClass = SEEDDataset
    elif dataset_name == "SEED_IV":
        DatasetClass = SEED_IVDataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    class DummyPretrainDataset(TensorDataset):
        def __init__(self, num_video_clips_total, num_subjects_per_video_group, 
                     num_freq_bands, num_channels, time_points):
            self.data = torch.randn(
                num_video_clips_total, 
                num_subjects_per_video_group, 
                num_freq_bands, 
                num_channels, 
                time_points
            )
            self.labels = torch.randint(0, 3, (num_video_clips_total,))
            super().__init__(self.data, self.labels)

    num_video_clips_total = config.get('num_video_clips_total', 15) 
    train_dataset = DummyPretrainDataset(
        num_video_clips_total=num_video_clips_total,
        num_subjects_per_video_group=ACTUAL_NUM_SUBJECTS_PER_VIDEO_GROUP_SEED, # 45
        num_freq_bands=NUM_FREQ_BANDS, # 5
        num_channels=ACTUAL_NUM_CHANNELS_SEED, # 62
        time_points=ACTUAL_NUM_TIME_WINDOWS_DE_SEED # 265
    )
    val_dataset = DummyPretrainDataset(
        num_video_clips_total=num_video_clips_total // 2,
        num_subjects_per_video_group=ACTUAL_NUM_SUBJECTS_PER_VIDEO_GROUP_SEED,
        num_freq_bands=NUM_FREQ_BANDS,
        num_channels=ACTUAL_NUM_CHANNELS_SEED,
        time_points=ACTUAL_NUM_TIME_WINDOWS_DE_SEED
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
        in_features=encoder.final_tcn_output_dim, 
        hidden_features=model_cfg['projection_head']['hidden_features'],
        dropout_prob=model_cfg['projection_head']['dropout_prob']
    )
    contrastive_loss_fn = NTXentLoss(temperature=temperature)

    class PretrainModel(nn.Module):
        def __init__(self, encoder, projection_head):
            super().__init__()
            self.encoder = encoder
            self.projection_head = projection_head
        
        def forward(self, x_video_batch, spatial_edge_index, temporal_edge_index):
            embeddings = self.encoder(x_video_batch, spatial_edge_index, temporal_edge_index)
            projected_embeddings = self.projection_head(embeddings)
            return projected_embeddings

    model = PretrainModel(encoder, projection_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    spatial_edge_index = get_spatial_edge_index(encoder_cfg['C_channels']).to(device)
    temporal_edge_index = get_temporal_edge_index(encoder_cfg['T_timesteps']).to(device)

    os.makedirs(pretrained_model_dir, exist_ok=True)
    
    print(f"Starting pretraining for {epochs} epochs (Experiment: {experiment_name})...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, contrastive_loss_fn, 
                                     spatial_edge_index, temporal_edge_index, Q_samples)
        val_loss = validate_one_epoch(model, val_loader, contrastive_loss_fn, 
                                      spatial_edge_index, temporal_edge_index, Q_samples)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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