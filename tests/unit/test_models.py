import pytest
import torch
import torch.nn as nn
import numpy as np

from ss_emerge.models.ss_emerge_encoder import SpectralEmbedding
from ss_emerge.models.ss_emerge_encoder import TCNBlock 
from ss_emerge.models.ss_emerge_encoder import GATLayer
from ss_emerge.models.ss_emerge_encoder import SS_EMERGE_Encoder

def test_spectral_embedding_output_shape():
    """Test SpectralEmbedding output shape."""
    B, F, C, T = 4, 5, 62, 128
    D_latent = 128
    mock_input = torch.randn(B, F, C, T)
    model = SpectralEmbedding(F, D_latent)
    output = model(mock_input)
    assert output.shape == (B, C, T, D_latent)

def test_spectral_embedding_linear_transformation():
    """Test SpectralEmbedding linear transformation."""
    F_bands = 5
    D_latent = 10
    B, C, T = 1, 1, 1 
    input1 = torch.randn(B, F_bands, C, T)
    scale_factor = 2.0
    input2 = input1 * scale_factor
    model = SpectralEmbedding(F_bands, D_latent)
    output1 = model(input1)
    output2 = model(input2)
    assert not torch.equal(output1, output2)

def test_tcn_block_output_shape():
    """Test TCNBlock output shape."""
    B = 8
    in_channels = 128
    out_channels = 256
    kernel_size = 3
    dilation = 1
    padding = (kernel_size - 1) * dilation
    T_in = 100

    mock_input = torch.randn(B, in_channels, T_in)
    model = TCNBlock(in_channels, out_channels, kernel_size, dilation, padding, dropout=0.2)
    output = model(mock_input)

    assert output.shape == (B, out_channels, T_in)

def test_tcn_block_residual_connection():
    """Test TCNBlock residual connection."""
    B = 2
    in_channels_diff = 64
    out_channels_diff = 128
    kernel_size = 3
    dilation = 1
    padding = (kernel_size - 1) * dilation
    T_in = 50

    mock_input_diff_channels = torch.randn(B, in_channels_diff, T_in)
    model_diff = TCNBlock(in_channels_diff, out_channels_diff, kernel_size, dilation, padding, dropout=0.0)

    output_diff = model_diff(mock_input_diff_channels)

    assert hasattr(model_diff, 'downsample') and model_diff.downsample is not None

    in_channels_same = 128
    out_channels_same = 128
    model_same = TCNBlock(in_channels_same, out_channels_same, kernel_size, dilation, padding, dropout=0.0)
    assert model_same.downsample is None

def test_tcn_block_causal_padding():
    """Test TCNBlock causal padding."""
    kernel_size = 5
    dilation = 2
    expected_padding_amount = (kernel_size - 1) * dilation
    
    model = TCNBlock(1, 1, kernel_size, dilation, expected_padding_amount, dropout=0.0)
    
    assert model.causal_padding_amount == expected_padding_amount
    assert model.conv1.padding == (0,) 
    assert model.conv2.padding == (0,)

def test_tcn_block_dilation_effect():
    """Test TCNBlock dilation effect."""
    in_channels = 10
    out_channels = 10
    kernel_size = 3
    dilation_val = 2
    padding = (kernel_size - 1) * dilation_val
    
    model = TCNBlock(in_channels, out_channels, kernel_size, dilation_val, padding, dropout=0.0)
    
    assert model.conv1.dilation == (dilation_val,)
    assert model.conv2.dilation == (dilation_val,)

def test_gat_layer_spatial_output_shape():
    """Test GATLayer (spatial mode) output shape."""
    B, C, T, D_features = 2, 62, 128, 128
    out_channels = 256
    heads = 2
    
    mock_x = torch.randn(B, C, T, D_features)
    num_edges = 100
    mock_spatial_edge_index = torch.randint(0, C, (2, num_edges), dtype=torch.long) 
    
    model = GATLayer(D_features, out_channels, heads=heads, graph_type='spatial')
    output = model(mock_x, mock_spatial_edge_index)
    
    assert output.shape == (B, C, T, out_channels)

def test_gat_layer_temporal_output_shape():
    """Test GATLayer (temporal mode) output shape."""
    B, C, T, D_features = 2, 62, 128, 128 
    out_channels = 256
    heads = 2
    
    mock_x = torch.randn(B, C, T, D_features)
    mock_temporal_edge_index = torch.stack([torch.arange(T-1), torch.arange(1, T)], dim=0)
    
    model = GATLayer(D_features, out_channels, heads=heads, graph_type='temporal')
    output = model(mock_x, mock_temporal_edge_index)
    
    assert output.shape == (B, C, T, out_channels)

def test_gat_layer_attention_mechanism_spatial():
    """Test GATLayer spatial attention mechanism."""
    B, C, T, D_features = 1, 3, 1, 10
    out_channels = 10
    heads = 1
    
    mock_x = torch.zeros(B, C, T, D_features)
    mock_x[0, 0, 0, :] = 1.0
    mock_x[0, 1, 0, :] = 0.5
    mock_x[0, 2, 0, :] = 0.2

    mock_spatial_edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long)
    
    model = GATLayer(D_features, out_channels, heads=heads, graph_type='spatial', dropout=0.0)
    
    output = model(mock_x, mock_spatial_edge_index)
    
    assert not torch.equal(output, mock_x)
    assert output.shape == (B, C, T, out_channels)

def test_gat_layer_concat_heads():
    """Test GATLayer head concatenation."""
    B, C, T, D_features = 1, 10, 10, 32
    out_channels_per_head = 16
    heads = 2

    mock_x = torch.randn(B, C, T, D_features)
    mock_edge_index = torch.randint(0, C, (2, 20), dtype=torch.long)

    model_concat = GATLayer(D_features, out_channels_per_head * heads, heads=heads, graph_type='spatial', concat=True)
    output_concat = model_concat(mock_x, mock_edge_index)
    assert output_concat.shape[-1] == out_channels_per_head * heads

    model_no_concat = GATLayer(D_features, out_channels_per_head, heads=heads, graph_type='spatial', concat=False)
    output_no_concat = model_no_concat(mock_x, mock_edge_index)
    assert output_no_concat.shape[-1] == out_channels_per_head

def test_ss_emerge_encoder_output_shape():
    """Test SS_EMERGE_Encoder output shape."""
    B, F_bands, C_channels, T_timesteps = 4, 5, 62, 128
    D_spectral = 128
    gat_out_channels = 256
    tcn_channels = [512, 512]
    tcn_kernel_size = 3
    tcn_dilations = [1, 2]
    final_embedding_dim = 512

    mock_input = torch.randn(B, F_bands, C_channels, T_timesteps)

    mock_spatial_edge_index = torch.randint(0, C_channels, (2, 100), dtype=torch.long)
    mock_temporal_edge_index = torch.stack([torch.arange(T_timesteps - 1), torch.arange(1, T_timesteps)], dim=0)

    model = SS_EMERGE_Encoder(
        F_bands=F_bands,
        D_spectral=D_spectral,
        C_channels=C_channels,
        T_timesteps=T_timesteps,
        gat_out_channels=gat_out_channels,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dilations=tcn_dilations,
        dropout_prob=0.0
    )
    output = model(mock_input, mock_spatial_edge_index, mock_temporal_edge_index)

    assert output.shape == (B, final_embedding_dim)

def test_ss_emerge_encoder_data_flow_integrity():
    """Test SS_EMERGE_Encoder data flow integrity."""
    B, F_bands, C_channels, T_timesteps = 1, 5, 1, 10
    D_spectral = 2
    gat_out_channels = 4
    tcn_channels = [8, 8]
    tcn_kernel_size = 2
    tcn_dilations = [1, 1]
    final_embedding_dim = 8

    mock_input = torch.ones(B, F_bands, C_channels, T_timesteps) * 0.1

    mock_spatial_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    mock_temporal_edge_index = torch.stack([torch.arange(T_timesteps - 1), torch.arange(1, T_timesteps)], dim=0)

    model = SS_EMERGE_Encoder(
        F_bands=F_bands,
        D_spectral=D_spectral,
        C_channels=C_channels,
        T_timesteps=T_timesteps,
        gat_out_channels=gat_out_channels,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dilations=tcn_dilations,
        dropout_prob=0.0
    )

    output1 = model(mock_input, mock_spatial_edge_index, mock_temporal_edge_index)
    output2 = model(mock_input * 2.0, mock_spatial_edge_index, mock_temporal_edge_index)

    assert not torch.allclose(output1, torch.zeros_like(output1), atol=1e-5)
    assert not torch.allclose(output1, output2, atol=1e-5)

def test_ss_emerge_encoder_tcn_pooling_correctness():
    """Test SS_EMERGE_Encoder TCN pooling correctness."""
    B, F_bands, C_channels, T_timesteps = 2, 5, 2, 50 
    D_spectral = 32 
    gat_out_channels = 64
    tcn_channels = [128, 256] 
    tcn_kernel_size = 3
    tcn_dilations = [1, 2]
    final_embedding_dim = tcn_channels[-1]

    mock_input = torch.randn(B, F_bands, C_channels, T_timesteps)
    mock_spatial_edge_index = torch.randint(0, C_channels, (2, 5), dtype=torch.long)
    mock_temporal_edge_index = torch.stack([torch.arange(T_timesteps - 1), torch.arange(1, T_timesteps)], dim=0)

    model = SS_EMERGE_Encoder(
        F_bands=F_bands,
        D_spectral=D_spectral,
        C_channels=C_channels,
        T_timesteps=T_timesteps,
        gat_out_channels=gat_out_channels,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dilations=tcn_dilations,
        dropout_prob=0.0
    )
    output = model(mock_input, mock_spatial_edge_index, mock_temporal_edge_index)
    
    assert output.shape[-1] == final_embedding_dim
    assert output.dim() == 2