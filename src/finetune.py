import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
import yaml
import tempfile 
import shutil 

from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.classification_head import ClassificationHead
from ss_emerge.models.resnet import ResNetEEG
from ss_emerge.utils.graph_helpers import get_spatial_edge_index, get_temporal_edge_index

from ss_emerge.datasets.seed_dataset import SEEDDataset
from ss_emerge.datasets.seed_iv_dataset import SEED_IVDataset


class SS_EMERGE_Finetune_Model(nn.Module):
    def __init__(self, encoder, classification_head):
        super().__init__()
        self.encoder = encoder
        self.classification_head = classification_head
    
    def forward(self, x, spatial_edge_index, temporal_edge_index):
        if not self.encoder.training and not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                embeddings = self.encoder(x, spatial_edge_index, temporal_edge_index, mode='embedding')
        else:
            # Encoder is trainable (e.g., training from scratch)
            embeddings = self.encoder(x, spatial_edge_index, temporal_edge_index, mode='embedding')
            
        logits = self.classification_head(embeddings)
        return logits

def train_finetune_one_epoch(model, dataloader, optimizer, criterion, 
                             spatial_edge_index, temporal_edge_index):
    model.train()
    total_loss = 0.0
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        batch_data = batch_data.to(model.encoder.parameters().__next__().device) 
        batch_labels = batch_labels.to(model.encoder.parameters().__next__().device)

        optimizer.zero_grad()
        
        logits = model(batch_data, spatial_edge_index, temporal_edge_index) 
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_finetune_one_epoch(model, dataloader, criterion, 
                                 spatial_edge_index, temporal_edge_index):
    model.eval()
    total_loss = 0.0
    with torch.no_grad(): 
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
            batch_data = batch_data.to(model.encoder.parameters().__next__().device) 
            batch_labels = batch_labels.to(model.encoder.parameters().__next__().device)

            logits = model(batch_data, spatial_edge_index, temporal_edge_index)
            loss = criterion(logits, batch_labels)
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
    num_classes = config['num_classes']
    
    # Extract finetuning settings
    finetune_cfg = config['finetune']
    epochs = finetune_cfg['epochs']
    batch_size = finetune_cfg['batch_size']
    lr = finetune_cfg['learning_rate']
    save_interval = finetune_cfg['save_interval']
    finetuned_model_dir = finetune_cfg['finetuned_model_dir']
    label_proportion = finetune_cfg.get('label_proportion', 1.0) # Default to 1.0

    # Path to pretrained encoder (from args, or could be passed via config)
    pretrained_model_path_from_args = args.pretrained_model_path 
    train_from_scratch = (pretrained_model_path_from_args.upper() == "NONE") # Special string to denote no pretraining

    # Extract model architecture settings
    model_cfg = config['model']
    model_type = model_cfg['model_type'] # "SS_EMERGE_Encoder" or "ResNetEEG"

    # Determine which encoder's parameters to use for dataset initialization
    # and for generating edge indices.
    common_encoder_params = None
    if model_type == "SS_EMERGE_Encoder":
        common_encoder_params = model_cfg['encoder']
    elif model_type == "ResNetEEG":
        common_encoder_params = model_cfg['ResNetEEG']
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'SS_EMERGE_Encoder' or 'ResNetEEG'.")


    # Get device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # --- Dataset Loading ---
    DatasetClass = None
    if dataset_name == "SEED":
        DatasetClass = SEEDDataset
    elif dataset_name == "SEED_IV":
        DatasetClass = SEED_IVDataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Instantiate Dataset for FINETUNING
    train_dataset = DatasetClass(
        data_root=data_root, # Pass data_root
        is_train=True,
        mode='finetune', # <--- KEY CHANGE: Specify 'finetune' mode
        sfreq=common_encoder_params['T_timesteps'], # Original SFREQ for DE calculation context
        bands=config.get('bands', {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)}),
        return_de_features=(model_type == "SS_EMERGE_Encoder"), # True for SS-EMERGE, False for ResNetEEG
        label_proportion=label_proportion, # Pass label proportion for few-shot
        random_seed=42 # For reproducibility of subsetting
    )
    val_dataset = DatasetClass(
        data_root=data_root, # Pass data_root
        is_train=False,
        mode='finetune', # <--- KEY CHANGE: Specify 'finetune' mode
        sfreq=common_encoder_params['T_timesteps'],
        bands=config.get('bands', {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)}),
        return_de_features=(model_type == "SS_EMERGE_Encoder"),
        label_proportion=1.0 # Validation always uses 100% labels
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Components Instantiation ---
    encoder = None
    classifier_in_features = None 
    classifier_dropout_prob = model_cfg['classification_head']['dropout_prob']

    if model_type == "SS_EMERGE_Encoder":
        encoder_cfg = model_cfg['encoder']
        encoder = SS_EMERGE_Encoder(
            F_bands=encoder_cfg['F_bands'], D_spectral=encoder_cfg['D_spectral'], 
            C_channels=encoder_cfg['C_channels'], T_timesteps=encoder_cfg['T_timesteps'],
            gat_out_channels=encoder_cfg['gat_out_channels'], 
            tcn_channels=encoder_cfg['tcn_channels'], 
            tcn_kernel_size=encoder_cfg['tcn_kernel_size'], 
            tcn_dilations=encoder_cfg['tcn_dilations'], 
            dropout_prob=encoder_cfg['dropout_prob']
        )
        classifier_in_features = encoder.final_tcn_output_dim

        # Load pretrained weights IF not training from scratch
        if not train_from_scratch:
            if not os.path.exists(pretrained_model_path_from_args):
                raise FileNotFoundError(f"Pretrained model checkpoint not found at: {pretrained_model_path_from_args}")
            print(f"Loading pretrained encoder from: {pretrained_model_path_from_args}")
            encoder.load_state_dict(torch.load(pretrained_model_path_from_args, map_location=device))
            # Freeze encoder parameters if loaded from pretraining
            for param in encoder.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen.")
        else:
            print("Training SS_EMERGE_Encoder from scratch (parameters will be unfrozen).")
            # Encoder parameters are trainable by default if not frozen
        
    elif model_type == "ResNetEEG":
        resnet_cfg = model_cfg['ResNetEEG']
        encoder = ResNetEEG(
            in_channels=resnet_cfg['in_channels'], 
            num_classes=num_classes, # Passed to ResNetEEG's internal classifier
            output_embedding_dim=resnet_cfg['output_embedding_dim'], # Final dim of ResNetEEG before its internal heads
            dropout_prob=resnet_cfg['dropout_prob'] # Dropout for ResNetEEG's internal layers
        )
        classifier_in_features = encoder.final_embedding_dim # ResNetEEG's output_embedding_dim
        
        # For ResNetEEG baseline, we always train from scratch in this context, so encoder parameters are trainable.
        print("Training ResNetEEG encoder from scratch.")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'SS_EMERGE_Encoder' or 'ResNetEEG'.")

    # Instantiate Classification Head (always the same structure, taking features from encoder)
    classification_head = ClassificationHead(
        in_features=classifier_in_features,
        num_classes=num_classes, 
        dropout_prob=classifier_dropout_prob
    )

    # Initialize the combined FinetuneModel
    model = SS_EMERGE_Finetune_Model(encoder, classification_head).to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    spatial_edge_index = get_spatial_edge_index(common_encoder_params['C_channels']).to(device)
    temporal_edge_index = get_temporal_edge_index(common_encoder_params['T_timesteps']).to(device)


    # Create save directory if it doesn't exist
    os.makedirs(finetuned_model_dir, exist_ok=True)

    # Training loop
    print(f"Starting finetuning for {epochs} epochs (Experiment: {experiment_name})...")
    for epoch in range(epochs):
        train_loss = train_finetune_one_epoch(model, train_loader, optimizer, criterion, 
                                              spatial_edge_index, temporal_edge_index)
        val_loss = validate_finetune_one_epoch(model, val_loader, criterion, 
                                               spatial_edge_index, temporal_edge_index)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save finetuned model checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(finetuned_model_dir, f"{experiment_name}_finetuned_epoch_{epoch+1}.pth"))
            print(f"Finetuned model saved at epoch {epoch+1} to {finetuned_model_dir}")

    print("Finetuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS-EMERGE Finetuning")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--pretrained_model_path", type=str, default="NONE", 
                        help="Path to the pretrained encoder checkpoint. Set to 'NONE' to train from scratch.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use, -1 for CPU.")
    
    args = parser.parse_args()

    main(args)