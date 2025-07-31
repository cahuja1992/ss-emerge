import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse

from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.classification_head import ClassificationHead
from ss_emerge.finetune import SS_EMERGE_Finetune_Model 

NUM_FREQ_BANDS = 5
NUM_CHANNELS = 62
TIME_POINTS = 200
D_SPECTRAL = 128
GAT_OUT_CHANNELS = 256
TCN_CHANNELS = [512, 512]
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2]
FINAL_ENCODER_DIM = 512
NUM_CLASSES = 3 

MOCK_SPATIAL_EDGE_INDEX = torch.randint(0, NUM_CHANNELS, (2, 100), dtype=torch.long)
MOCK_TEMPORAL_EDGE_INDEX = torch.stack([torch.arange(TIME_POINTS - 1), torch.arange(1, TIME_POINTS)], dim=0)


# Prediction function
def predict_samples(model, dataloader, spatial_edge_index, temporal_edge_index):
    """
    Makes predictions on new data using a trained model.
    """
    model.eval() # Set model to evaluation mode
    all_preds = []

    with torch.no_grad(): # Disable gradient calculations
        for batch_data in dataloader:
            # Dataloader might yield (data,) or (data, labels) if labels exist.
            # For prediction, we expect only data.
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0] # Take the data tensor

            batch_data = batch_data.to(model.encoder.parameters().__next__().device) 

            logits = model(batch_data, spatial_edge_index, temporal_edge_index)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds)

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # Instantiate model components
    encoder = SS_EMERGE_Encoder(
        F_bands=NUM_FREQ_BANDS, D_spectral=D_SPECTRAL, C_channels=NUM_CHANNELS, T_timesteps=TIME_POINTS,
        gat_out_channels=GAT_OUT_CHANNELS, tcn_channels=TCN_CHANNELS, tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dilations=TCN_DILATIONS, dropout_prob=0.0
    )
    classification_head = ClassificationHead(
        in_features=FINAL_ENCODER_DIM, num_classes=NUM_CLASSES, dropout_prob=0.0
    )

    model = SS_EMERGE_Finetune_Model(encoder, classification_head)
    
    # Load state dict
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Finetuned model checkpoint not found at: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # Crucial: Set model to evaluation mode

    # Freeze encoder (already done during finetuning, good practice for prediction)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Load input data
    if not os.path.exists(args.input_data_path):
        raise FileNotFoundError(f"Input data not found at: {args.input_data_path}")
    
    input_data_np = np.load(args.input_data_path)
    # Assume input_data_np is (num_samples, F, C, T)
    input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32)

    # Create DataLoader for prediction
    # Mock dataset for prediction (replaces actual dataset loading)
    class TempPredictionDataset(TensorDataset):
        def __init__(self, data_tensor):
            super().__init__(data_tensor)

    predict_dataset = TempPredictionDataset(input_data_tensor)
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    # Edge indices on device
    spatial_edge_index_device = MOCK_SPATIAL_EDGE_INDEX.to(device)
    temporal_edge_index_device = MOCK_TEMPORAL_EDGE_INDEX.to(device)

    print(f"Making predictions using model from: {args.model_path}")
    predictions = predict_samples(model, predict_loader, spatial_edge_index_device, temporal_edge_index_device)

    # Save predictions
    np.save(args.output_path, predictions)
    print(f"Predictions saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS-EMERGE Model Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model checkpoint.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input data (Numpy array).")
    parser.add_argument("--output_path", type=str, default="./predictions.npy", help="Path to save the predictions.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use, -1 for CPU.")
    
    args = parser.parse_args()
    main(args)