import torch
import numpy as np

def get_spatial_edge_index(num_channels):
    """
    Generates a mock spatial edge_index for a graph of EEG channels.
    In a real scenario, this would be based on electrode topology or connectivity.
    For testing, we create a simple adjacency matrix (e.g., a random sparse graph
    or a fully connected graph for very small num_channels).

    Args:
        num_channels (int): The number of EEG channels.

    Returns:
        torch.Tensor: A tensor of shape (2, num_edges) representing the edge indices.
    """
    if num_channels <= 1:
        return torch.empty((2, 0), dtype=torch.long) # No edges for 0 or 1 channel

    # For testing, let's create a simple set of random connections
    # Or a fully connected graph if num_channels is very small, for robustness.
    
    # Option 1: Simple random edges (sparse graph)
    num_edges = min(num_channels * 5, num_channels * (num_channels - 1)) # Limit number of edges
    if num_edges == 0 and num_channels > 1: # Ensure at least some connections if possible
        num_edges = num_channels # Default to at least N edges

    source_nodes = torch.randint(0, num_channels, (num_edges,), dtype=torch.long)
    target_nodes = torch.randint(0, num_channels, (num_edges,), dtype=torch.long)
    
    # Remove self-loops
    valid_indices = source_nodes != target_nodes
    source_nodes = source_nodes[valid_indices]
    target_nodes = target_nodes[valid_indices]

    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    # Ensure it's bidirectional if graph is undirected
    # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # edge_index = torch.unique(edge_index, dim=1) # Remove duplicates after making bidirectional

    return edge_index

def get_temporal_edge_index(num_timesteps):
    """
    Generates a temporal edge_index for a graph of time steps (e.g., adjacent connections).
    Assumes a simple sequential graph where each time step connects to its immediate successor.

    Args:
        num_timesteps (int): The number of time points in a segment.

    Returns:
        torch.Tensor: A tensor of shape (2, num_edges) representing the edge indices.
    """
    if num_timesteps <= 1:
        return torch.empty((2, 0), dtype=torch.long) # No edges for 0 or 1 timestep

    # Simple causal/sequential connections: i -> i+1
    source_nodes = torch.arange(num_timesteps - 1, dtype=torch.long)
    target_nodes = torch.arange(1, num_timesteps, dtype=torch.long)
    
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    # For temporal GATs, sometimes bidirectional connections (i <-> i+1) are used for context,
    # or just causal (i -> i+k) for prediction. The problem description suggests causal.
    # The GATLayer in SS-EMERGE uses this.
    # We can also add self-loops (i -> i) if needed for attention to own node.
    
    # TODO: SOGNN style a more complex temporal graph (e.g., k-nearest neighbors in time)

    return edge_index
