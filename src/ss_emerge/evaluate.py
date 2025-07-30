import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
import yaml 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.classification_head import ClassificationHead
from ss_emerge.utils.graph_helpers import get_spatial_edge_index, get_temporal_edge_index 
from ss_emerge.finetune import SS_EMERGE_Finetune_Model 

from ss_emerge.datasets.seed_dataset import SEEDDataset
from ss_emerge.datasets.seed_iv_dataset import SEED_IVDataset


def evaluate_model(model, dataloader, criterion, spatial_edge_index, temporal_edge_index, num_classes):
    """
    Evaluates the model on the given dataloader and returns metrics.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        for batch_data, batch_labels in dataloader:
            # Ensure data and labels are on the correct device
            batch_data = batch_data.to(model.encoder.parameters().__next__().device) 
            batch_labels = batch_labels.to(model.encoder.parameters().__next__().device)

            logits = model(batch_data, spatial_edge_index, temporal_edge_index)
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,       
    }
    
    return metrics

def main(args):
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract general settings
    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    data_root = config['data_root']
    num_classes = config['num_classes']

    # Extract model architecture settings
    model_cfg = config['model']
    encoder_cfg = model_cfg['encoder']
    classifier_cfg = model_cfg['classification_head']

    # Path to finetuned model (from args)
    finetuned_model_path = args.model_path 
    
    # Get device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # --- Model Components ---
    encoder = SS_EMERGE_Encoder(
        F_bands=encoder_cfg['F_bands'], D_spectral=encoder_cfg['D_spectral'], 
        C_channels=encoder_cfg['C_channels'], T_timesteps=encoder_cfg['T_timesteps'],
        gat_out_channels=encoder_cfg['gat_out_channels'], 
        tcn_channels=encoder_cfg['tcn_channels'], 
        tcn_kernel_size=encoder_cfg['tcn_kernel_size'], 
        tcn_dilations=encoder_cfg['tcn_dilations'], 
        dropout_prob=0.0 # Dropout is off in eval mode anyway
    )
    classification_head = ClassificationHead(
        in_features=encoder.final_tcn_output_dim, 
        num_classes=num_classes, 
        dropout_prob=0.0 # Dropout is off in eval mode anyway
    )

    model = SS_EMERGE_Finetune_Model(encoder, classification_head)
    
    # Load state dict from the finetuned model path
    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(f"Finetuned model checkpoint not found at: {finetuned_model_path}")
    
    model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    model.to(device)
    model.eval() # Crucial: Set model to evaluation mode

    # Freeze encoder parameters (important for evaluation too, though no_grad context manager handles gradients)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # --- Dataset Loading ---
    DatasetClass = None
    if dataset_name == "SEED":
        DatasetClass = SEEDDataset
    elif dataset_name == "SEED_IV":
        DatasetClass = SEED_IVDataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load test dataset for evaluation
    eval_dataset = DatasetClass(
        data_path=os.path.join(data_root, config['test_data_path']),
        labels_path=os.path.join(data_root, config['test_labels_path']),
        sfreq=encoder_cfg['T_timesteps'], # Assuming T_timesteps is original SFREQ for DE
        bands=config.get('bands', {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)})
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Edge indices (need to be tensors on device)
    spatial_edge_index = get_spatial_edge_index(encoder_cfg['C_channels']).to(device)
    temporal_edge_index = get_temporal_edge_index(encoder_cfg['T_timesteps']).to(device)

    criterion = nn.CrossEntropyLoss()

    print(f"Evaluating model from: {finetuned_model_path} on {dataset_name} test set.")
    metrics = evaluate_model(model, eval_loader, criterion, spatial_edge_index, temporal_edge_index, num_classes)

    print("\n--- Evaluation Results ---")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name.replace('_', ' ').capitalize()}: {value:.4f}")
        else:
            print(f"{metric_name.replace('_', ' ').capitalize()}: {value}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS-EMERGE Model Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model checkpoint.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use, -1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    main(args)