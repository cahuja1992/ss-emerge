import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ss_emerge.models.classification_head import ClassificationHead

def test_classification_head_output_shape():
    """Test ClassificationHead output shape."""
    B = 16
    in_features = 512
    num_classes = 3

    mock_input = torch.randn(B, in_features)
    model = ClassificationHead(in_features, num_classes, dropout_prob=0.0)

    output = model(mock_input)

    assert output.shape == (B, num_classes)
    assert output.dtype == torch.float32

def test_classification_head_returns_logits():
    """Test ClassificationHead returns logits."""
    B = 4
    in_features = 128
    num_classes = 4

    mock_input = torch.randn(B, in_features)
    model = ClassificationHead(in_features, num_classes, dropout_prob=0.0)

    output = model(mock_input)

    assert output.max().item() > 1.0 or output.min().item() < 0.0, \
        "Output values appear to be probabilities, not logits."
    assert not torch.allclose(output.sum(dim=1), torch.ones(B), atol=1e-5), \
        "Output sums to 1 across class dimension, implying internal softmax."

def test_classification_head_applies_transformation_and_dropout():
    """Test ClassificationHead applies transformation and dropout."""
    B = 4
    in_features = 128
    num_classes = 3
    
    mock_input = torch.randn(B, in_features)

    model_no_dropout = ClassificationHead(in_features, num_classes, dropout_prob=0.0)
    output_no_dropout_1 = model_no_dropout(mock_input)
    output_no_dropout_2 = model_no_dropout(mock_input * 1.1)

    assert not torch.allclose(output_no_dropout_1, output_no_dropout_2, atol=1e-5), \
        "Output is too similar for scaled input, indicating no transformation."
    
    model_with_dropout = ClassificationHead(in_features, num_classes, dropout_prob=0.5)
    model_with_dropout.train()
    
    mock_input_for_dropout_test = torch.randn(B, in_features) 
    output_dropout_test = model_with_dropout(mock_input_for_dropout_test)
    
    model_without_dropout_control = ClassificationHead(in_features, num_classes, dropout_prob=0.0)
    output_without_dropout_control = model_without_dropout_control(mock_input_for_dropout_test)

    assert not torch.allclose(output_dropout_test, output_without_dropout_control, atol=1e-3), \
        "Output with dropout is too similar to output without dropout, indicating dropout might not be active."

    assert not torch.isnan(output_dropout_test).any()
    assert not torch.isinf(output_dropout_test).any()
    assert not torch.isnan(output_without_dropout_control).any()
    assert not torch.isinf(output_without_dropout_control).any()