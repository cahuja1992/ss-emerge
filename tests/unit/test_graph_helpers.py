import pytest
import torch
import numpy as np

# Import the functions to be tested
from ss_emerge.utils.graph_helpers import get_spatial_edge_index, get_temporal_edge_index

def test_get_spatial_edge_index_output_shape():
    """
    Test that get_spatial_edge_index returns a tensor with shape (2, num_edges).
    """
    num_channels = 10
    edge_index = get_spatial_edge_index(num_channels)
    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long

def test_get_spatial_edge_index_no_self_loops():
    """
    Test that the generated spatial edge_index does not contain self-loops (node connected to itself).
    """
    num_channels = 20
    edge_index = get_spatial_edge_index(num_channels)
    # Check if any source node is equal to its target node
    assert not torch.any(edge_index[0] == edge_index[1]), "Self-loops found in spatial edge_index."

def test_get_spatial_edge_index_valid_node_indices():
    """
    Test that all node indices in spatial_edge_index are within the valid range [0, num_channels-1].
    """
    num_channels = 15
    edge_index = get_spatial_edge_index(num_channels)
    assert torch.all(edge_index >= 0), "Node index less than 0 found in spatial edge_index."
    assert torch.all(edge_index < num_channels), "Node index greater than or equal to num_channels found in spatial edge_index."

def test_get_spatial_edge_index_single_channel():
    """
    Test behavior for a single channel (should result in no edges).
    """
    num_channels = 1
    edge_index = get_spatial_edge_index(num_channels)
    assert edge_index.shape == (2, 0), "Spatial edge_index should be empty for single channel."

def test_get_spatial_edge_index_zero_channels():
    """
    Test behavior for zero channels (should result in no edges).
    """
    num_channels = 0
    edge_index = get_spatial_edge_index(num_channels)
    assert edge_index.shape == (2, 0), "Spatial edge_index should be empty for zero channels."


# --- Tests for get_temporal_edge_index ---

def test_get_temporal_edge_index_output_shape():
    """
    Test that get_temporal_edge_index returns a tensor with shape (2, num_edges).
    """
    num_timesteps = 20
    edge_index = get_temporal_edge_index(num_timesteps)
    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long
    assert edge_index.shape[1] == num_timesteps - 1 # Expect N-1 edges for N timesteps (i -> i+1)

def test_get_temporal_edge_index_sequential_connections():
    """
    Test that temporal edge_index represents sequential connections (i -> i+1).
    """
    num_timesteps = 5
    edge_index = get_temporal_edge_index(num_timesteps)
    expected_source = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    expected_target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    
    assert torch.equal(edge_index[0], expected_source), "Temporal source nodes are incorrect."
    assert torch.equal(edge_index[1], expected_target), "Temporal target nodes are incorrect."

def test_get_temporal_edge_index_valid_node_indices():
    """
    Test that all node indices in temporal_edge_index are within the valid range [0, num_timesteps-1].
    """
    num_timesteps = 10
    edge_index = get_temporal_edge_index(num_timesteps)
    assert torch.all(edge_index >= 0), "Node index less than 0 found in temporal edge_index."
    assert torch.all(edge_index < num_timesteps), "Node index greater than or equal to num_timesteps found in temporal edge_index."

def test_get_temporal_edge_index_single_timestep():
    """
    Test behavior for a single time step (should result in no edges).
    """
    num_timesteps = 1
    edge_index = get_temporal_edge_index(num_timesteps)
    assert edge_index.shape == (2, 0), "Temporal edge_index should be empty for single time step."

def test_get_temporal_edge_index_zero_timesteps():
    """
    Test behavior for zero time steps (should result in no edges).
    """
    num_timesteps = 0
    edge_index = get_temporal_edge_index(num_timesteps)
    assert edge_index.shape == (2, 0), "Temporal edge_index should be empty for zero time steps."