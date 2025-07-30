import pytest
import torch
import torch.nn.functional as F
import numpy as np

from ss_emerge.models.contrastive_loss import NTXentLoss

def test_ntxent_loss_basic_calculation():
    """Test a basic calculation of NTXentLoss."""
    temperature = 1.0
    loss_fn = NTXentLoss(temperature=temperature)

    B = 2
    D = 2

    z_i_np = np.array([[1.0, 0.0], [0.0, 1.0]])
    z_j_np = np.array([[1.0, 0.0], [0.0, 1.0]])

    z_i = torch.tensor(z_i_np, dtype=torch.float32)
    z_j = torch.tensor(z_j_np, dtype=torch.float32)
    
    loss_val_perfect = loss_fn(z_i[0].unsqueeze(0), z_j[0].unsqueeze(0))
    assert torch.isclose(loss_val_perfect, torch.tensor(0.0, dtype=torch.float32), atol=1e-5)

    expected_loss_val_np = np.log(np.exp(1.0) + 2.0) - 1.0 
    expected_loss_val = torch.tensor(expected_loss_val_np, dtype=torch.float32)
    
    loss_val_multi_pair = loss_fn(z_i, z_j)
    assert torch.isclose(loss_val_multi_pair, expected_loss_val, atol=1e-5)


def test_ntxent_loss_temperature_effect():
    """Test that temperature affects the loss as expected."""
    temperature_low = 0.1
    temperature_high = 10.0
    loss_fn_low = NTXentLoss(temperature=temperature_low)
    loss_fn_high = NTXentLoss(temperature=temperature_high)

    z_i_np = np.array([[0.8, 0.6], [0.7, 0.7]])
    z_j_np = np.array([[0.7, 0.7], [0.8, 0.6]])

    z_i = torch.tensor(z_i_np, dtype=torch.float32)
    z_j = torch.tensor(z_j_np, dtype=torch.float32)

    loss_low_temp = loss_fn_low(z_i, z_j)
    loss_high_temp = loss_fn_high(z_i, z_j)

    assert loss_low_temp > loss_high_temp

def test_ntxent_loss_normalization_input():
    """Test that the loss function performs normalization if input is not already normalized."""
    temperature = 1.0
    loss_fn = NTXentLoss(temperature=temperature)

    z_i = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    z_j = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)

    z_i_norm = F.normalize(z_i, dim=-1)
    z_j_norm = F.normalize(z_j, dim=-1)

    loss_non_norm = loss_fn(z_i, z_j)
    loss_norm = loss_fn(z_i_norm, z_j_norm)

    assert torch.isclose(loss_non_norm, loss_norm, atol=1e-5)