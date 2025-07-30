import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil

# Import components from our library
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder
from ss_emerge.models.classification_head import ClassificationHead
from ss_emerge.finetune import SS_EMERGE_Finetune_Model # The model structure to load

# Import the actual evaluate function from the script
from ss_emerge.evaluate import evaluate_model, main as evaluate_main

# Constants for mock data and model setup (consistent with finetune/pretrain tests)
NUM_FREQ_BANDS = 5
NUM_CHANNELS = 62
TIME_POINTS = 200
D_SPECTRAL = 128
GAT_OUT_CHANNELS = 256
TCN_CHANNELS = [512, 512]
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2]
FINAL_ENCODER_DIM = 512
NUM_CLASSES = 3 # For finetuning task (e.g., SEED: Positive, Neutral, Negative)

# Mock edge indices for GATs
MOCK_SPATIAL_EDGE_INDEX = torch.randint(0, NUM_CHANNELS, (2, 100), dtype=torch.long)
MOCK_TEMPORAL_EDGE_INDEX = torch.stack([torch.arange(TIME_POINTS - 1), torch.arange(1, TIME_POINTS)], dim=0)


class MockEvalDataset(TensorDataset):
    """
    A mock dataset for evaluation, providing labeled EEG data.
    """
    def __init__(self, num_samples, num_freq_bands, num_channels, time_points, num_classes):
        self.data = torch.randn(num_samples, num_freq_bands, num_channels, time_points)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
        super().__init__(self.data, self.labels)


@pytest.fixture(scope="module")
def finetuned_model_path():
    """
    Fixture to create and save a dummy finetuned model for testing evaluation.
    This simulates a successful finetuning phase.
    """
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "dummy_finetuned_model.pth")

    try:
        # Instantiate and "finetune" a dummy model structure
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
        optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        dummy_input = torch.randn(16, NUM_FREQ_BANDS, NUM_CHANNELS, TIME_POINTS)
        dummy_labels = torch.randint(0, NUM_CLASSES, (16,), dtype=torch.long)
        
        # A simple forward/backward pass
        model.train()
        optimizer.zero_grad()
        logits = model(dummy_input, MOCK_SPATIAL_EDGE_INDEX, MOCK_TEMPORAL_EDGE_INDEX)
        loss = criterion(logits, dummy_labels)
        loss.backward()
        optimizer.step()
        
        # Save the full model state_dict (encoder and classification head)
        torch.save(model.state_dict(), model_path)
        yield model_path
    finally:
        shutil.rmtree(temp_dir)


# --- Integration Tests for Evaluation Script ---

def test_evaluate_model_function(finetuned_model_path):
    """
    Test the core evaluate_model function's logic.
    """
    device = torch.device("cpu")

    # Load the finetuned model (encoder and classifier)
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

    # Ensure encoder is frozen (although it should be from finetuning, good to assert)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Mock data loader for evaluation
    eval_dataset = MockEvalDataset(num_samples=30, num_freq_bands=NUM_FREQ_BANDS,
                                    num_channels=NUM_CHANNELS, time_points=TIME_POINTS, num_classes=NUM_CLASSES)
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(model, eval_loader, criterion, 
                             MOCK_SPATIAL_EDGE_INDEX.to(device), MOCK_TEMPORAL_EDGE_INDEX.to(device), NUM_CLASSES)

    # Assertions on returned metrics
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics

    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["f1_macro"], float)
    assert isinstance(metrics["precision_macro"], float)
    assert isinstance(metrics["recall_macro"], float)

    # Values should be reasonable (e.g., loss > 0, accuracy between 0 and 1)
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1
    assert 0 <= metrics["precision_macro"] <= 1
    assert 0 <= metrics["recall_macro"] <= 1


def test_evaluate_main_script(finetuned_model_path):
    """
    Test the main function of src/evaluate.py by calling it with mocked arguments.
    """
    # Create a temporary directory for output (if evaluate_main saves anything, though it typically just prints)
    temp_output_dir = tempfile.mkdtemp()

    try:
        # Mock argparse arguments
        mock_args = argparse.Namespace(
            model_path=finetuned_model_path,
            batch_size=16,
            num_eval_samples=50,
            gpu=-1, # Force CPU
            output_dir=temp_output_dir # If evaluate_main were to save results
        )

        # Call the actual main function
        evaluate_main(mock_args)

        # Assertions (primarily, no crash, and prints expected output)
        # We can't directly capture stdout for strict assertion here without mocking sys.stdout,
        # but the test passing implies the script ran without unhandled exceptions.
        # Check for non-NaN/Inf outputs by running a small evaluation here
        
        # Re-instantiate model and dataloader to check results if desired (beyond just "no crash")
        # This part of the test could be more robust if evaluate_main returned values,
        # but as a script, it usually prints.
        # For simplicity, we primarily ensure it runs without errors.
        
        device = torch.device("cpu")
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
        
        eval_dataset = MockEvalDataset(num_samples=mock_args.num_eval_samples, num_freq_bands=NUM_FREQ_BANDS,
                                       num_channels=NUM_CHANNELS, time_points=TIME_POINTS, num_classes=NUM_CLASSES)
        eval_loader = DataLoader(eval_dataset, batch_size=mock_args.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate_model(model, eval_loader, criterion, 
                                 MOCK_SPATIAL_EDGE_INDEX.to(device), MOCK_TEMPORAL_EDGE_INDEX.to(device), NUM_CLASSES)
        
        assert not np.isnan(metrics["loss"]) and not np.isinf(metrics["loss"])
        assert not np.isnan(metrics["accuracy"]) and not np.isinf(metrics["accuracy"])
        assert not np.isnan(metrics["f1_macro"]) and not np.isinf(metrics["f1_macro"])


    finally:
        shutil.rmtree(temp_output_dir)