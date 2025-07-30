import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
import argparse

# Import components from our library (model definitions)
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.classification_head import ClassificationHead
from ss_emerge.finetune import SS_EMERGE_Finetune_Model # The model structure to load

# Import the actual predict function/main from the script
from ss_emerge.predict import predict_samples, main as predict_main # These imports will initially fail

# Constants for mock data and model setup (consistent with previous tests)
NUM_FREQ_BANDS = 5
NUM_CHANNELS = 62
TIME_POINTS = 200
D_SPECTRAL = 128
GAT_OUT_CHANNELS = 256
TCN_CHANNELS = [512, 512]
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2]
FINAL_ENCODER_DIM = 512
NUM_CLASSES = 3 # Example: Positive, Neutral, Negative

# Mock edge indices (global for the test module)
MOCK_SPATIAL_EDGE_INDEX = torch.randint(0, NUM_CHANNELS, (2, 100), dtype=torch.long)
MOCK_TEMPORAL_EDGE_INDEX = torch.stack([torch.arange(TIME_POINTS - 1), torch.arange(1, TIME_POINTS)], dim=0)


@pytest.fixture(scope="module")
def finetuned_model_path():
    """
    Fixture to create and save a dummy finetuned model for testing prediction.
    """
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "dummy_finetuned_model_for_predict.pth")

    try:
        # Instantiate and save a dummy finetuned model
        encoder = SS_EMERGE_Encoder(
            F_bands=NUM_FREQ_BANDS, D_spectral=D_SPECTRAL, C_channels=NUM_CHANNELS, T_timesteps=TIME_POINTS,
            gat_out_channels=GAT_OUT_CHANNELS, tcn_channels=TCN_CHANNELS, tcn_kernel_size=TCN_KERNEL_SIZE,
            tcn_dilations=TCN_DILATIONS, dropout_prob=0.0
        )
        classification_head = ClassificationHead(
            in_features=FINAL_ENCODER_DIM, num_classes=NUM_CLASSES, dropout_prob=0.0
        )
        
        model = SS_EMERGE_Finetune_Model(encoder, classification_head)
        
        # Simulate some training to get non-zero weights in classifier
        # This is a minimal forward/backward pass to get trained weights
        optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        dummy_input = torch.randn(16, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS)
        dummy_labels = torch.randint(0, NUM_CLASSES, (16,), dtype=torch.long)
        
        model.train()
        optimizer.zero_grad()
        logits = model(dummy_input, MOCK_SPATIAL_EDGE_INDEX, MOCK_TEMPORAL_EDGE_INDEX)
        loss = criterion(logits, dummy_labels)
        loss.backward()
        optimizer.step()
        
        torch.save(model.state_dict(), model_path)
        yield model_path
    finally:
        shutil.rmtree(temp_dir)


class MockPredictionDataset(TensorDataset):
    """
    A mock dataset for prediction, providing unlabeled EEG data.
    """
    def __init__(self, num_samples, num_freq_bands, num_channels, time_points):
        self.data = torch.randn(num_samples, num_freq_bands, num_channels, time_points)
        # No labels for prediction dataset
        super().__init__(self.data)


# --- Integration Tests for Prediction Script ---

def test_predict_samples_function(finetuned_model_path):
    """
    Test the core predict_samples function's logic.
    Ensures it returns predictions in the correct format.
    """
    device = torch.device("cpu")

    # Instantiate and load the finetuned model
    encoder = SS_EMERGE_Encoder(
        F_bands=NUM_FREQ_BANDS, D_spectral=D_SPECTRAL, C_channels=NUM_CHANNELS, T_timesteps=TIME_POINTS,
        gat_out_channels=GAT_OUT_CHANNELS, tcn_channels=TCN_CHANNELS, tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dilations=TCN_DILATIONS, dropout_prob=0.0
    ).to(device)
    classification_head = ClassificationHead(
        in_features=FINAL_ENCODER_DIM, num_classes=NUM_CLASSES, dropout_prob=0.0
    ).to(device)
    model = SS_EMERGE_Finetune_Model(encoder, classification_head).to(device)
    model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    model.eval() # Set to eval mode

    # Mock data loader for prediction
    num_predict_samples = 10
    predict_dataset = MockPredictionDataset(num_predict_samples, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS)
    predict_loader = DataLoader(predict_dataset, batch_size=5, shuffle=False)

    # Edge indices on device
    spatial_edge_index_device = MOCK_SPATIAL_EDGE_INDEX.to(device)
    temporal_edge_index_device = MOCK_TEMPORAL_EDGE_INDEX.to(device)

    # Call the actual predict_samples function
    predictions = predict_samples(model, predict_loader, 
                                  spatial_edge_index_device, temporal_edge_index_device)

    # Assertions on returned predictions
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array."
    assert predictions.shape == (num_predict_samples,), "Predictions shape mismatch."
    assert np.all(predictions >= 0) and np.all(predictions < NUM_CLASSES), "Predictions should be valid class indices."


def test_predict_main_script(finetuned_model_path):
    """
    Test the main function of src/predict.py by calling it with mocked arguments.
    Ensures the script runs without errors and produces a result.
    """
    # Create a temporary directory for output (e.g., saved predictions)
    temp_output_dir = tempfile.mkdtemp()
    output_filepath = os.path.join(temp_output_dir, "predictions.npy")

    try:
        # Mock argparse arguments
        mock_args = argparse.Namespace(
            model_path=finetuned_model_path,
            input_data_path="dummy_input_data.npy", # Mock this file exists or is handled
            output_path=output_filepath,
            batch_size=16,
            num_predict_samples=20, # Number of samples the mock dataset will provide
            gpu=-1 # Force CPU
        )

        # Create a dummy input data file that main() would attempt to load
        dummy_input_data_np = np.random.rand(mock_args.num_predict_samples, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS).astype(np.float32)
        np.save(mock_args.input_data_path, dummy_input_data_np)

        # Call the actual main function
        predict_main(mock_args)

        # Assertions
        assert os.path.exists(output_filepath), "Output prediction file was not created."
        predictions_loaded = np.load(output_filepath)
        assert predictions_loaded.shape == (mock_args.num_predict_samples,), "Loaded predictions shape mismatch."
        
    finally:
        # Clean up dummy files and directories
        if os.path.exists(mock_args.input_data_path):
            os.remove(mock_args.input_data_path)
        shutil.rmtree(temp_output_dir)