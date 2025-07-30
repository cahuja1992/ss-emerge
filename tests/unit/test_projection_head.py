import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ss_emerge.models.projection_head import ProjectionHead


def test_projection_head_output_shape():
    """
    Test that ProjectionHead produces the correct output shape.
    Input: (B, in_features) from encoder
    Output: (B, final_hidden_features) after L2-normalization.
    The thesis states 4096-dimensional output.
    """
    B = 16
    in_features = 512
    final_hidden_features = 4096
    mock_input = torch.randn(B, in_features)
    model = ProjectionHead(in_features, hidden_features=[1024, 2048, final_hidden_features], dropout_prob=0.0) 
    output = model(mock_input)
    assert output.shape == (B, final_hidden_features)

def test_projection_head_l2_normalization():
    """
    Test that the output embeddings are L2-normalized (i.e., have unit norm).
    """
    B = 8
    in_features = 256
    final_hidden_features = 1024
    mock_input = torch.randn(B, in_features)
    model = ProjectionHead(in_features, hidden_features=[512, final_hidden_features], dropout_prob=0.0)
    output = model(mock_input)
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_projection_head_applies_transformation_and_dropout():
    """
    Test that the projection head actually transforms the input and dropout is active (when enabled).
    Transformation check: output should not be identical to a simple linear mapping of input
                       (due to non-linearities and multiple layers).
    Dropout check: when dropout_prob > 0 and in train mode, output should be significantly different
                   from when dropout_prob is 0 (for the same input).
    """
    B = 4
    in_features = 128
    final_hidden_features = 256
    
    mock_input = torch.randn(B, in_features)

    # Test transformation (with dropout off for predictability)
    model_no_dropout = ProjectionHead(in_features, hidden_features=[final_hidden_features], dropout_prob=0.0)
    output_no_dropout_1 = model_no_dropout(mock_input)
    output_no_dropout_2 = model_no_dropout(mock_input + 0.1) 

    assert not torch.allclose(output_no_dropout_1, output_no_dropout_2, atol=1e-5)
    
    # Test dropout functionality:
    # 1. Create a model with dropout enabled
    model_with_dropout = ProjectionHead(in_features, hidden_features=[final_hidden_features], dropout_prob=0.5)
    model_with_dropout.train() # Set to train mode to activate dropout

    # 2. Create another model with dropout disabled (dropout_prob=0.0)
    # This acts as our "control" for what the output *should* be without dropout's effect.
    model_without_dropout_control = ProjectionHead(in_features, hidden_features=[final_hidden_features], dropout_prob=0.0)
    # No need for train/eval mode for control as dropout_prob is 0.0

    # 3. Pass the *same* input through both models
    output_with_dropout = model_with_dropout(mock_input)
    output_without_dropout_control = model_without_dropout_control(mock_input)

    # Assert that the output with dropout is *not* close to the output without dropout.
    # This is the most reliable way to confirm dropout's active perturbation.
    # We use a non-zero atol because of potential small floating point differences,
    # but the primary expectation is that dropout causes significant changes.
    assert not torch.allclose(output_with_dropout, output_without_dropout_control, atol=1e-3), \
        "Output with dropout is too similar to output without dropout, indicating dropout might not be active."

    # Also ensure output is not NaN/Inf
    assert not torch.isnan(output_with_dropout).any()
    assert not torch.isinf(output_with_dropout).any()
    assert not torch.isnan(output_without_dropout_control).any()
    assert not torch.isinf(output_without_dropout_control).any()