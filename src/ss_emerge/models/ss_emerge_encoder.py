import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpectralEmbedding(nn.Module):
    def __init__(self, F_bands, D_latent):
        super(SpectralEmbedding, self).__init__()
        self.linear = nn.Linear(F_bands, D_latent)
        self.D_latent = D_latent 

    def forward(self, x):
        B, F, C, T = x.shape
        x_reshaped_for_linear = x.permute(0, 2, 3, 1).reshape(B * C * T, F) 
        z_ct = self.linear(x_reshaped_for_linear) 
        z = z_ct.reshape(B, C, T, self.D_latent) 
        return z

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        self.causal_padding_amount = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=0, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=0, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.norm = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = F.pad(x, (self.causal_padding_amount, 0))
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = F.pad(out, (self.causal_padding_amount, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.norm(out + residual) 
        return out


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.5, concat=True, graph_type='spatial'):
        super(GATLayer, self).__init__()
        self.graph_type = graph_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        if concat:
            gat_out_dim_per_head = out_channels // heads
            assert out_channels % heads == 0, "out_channels must be divisible by heads if concat=True"
        else:
            gat_out_dim_per_head = out_channels 

        self.gat_conv = GATConv(in_channels, gat_out_dim_per_head, heads=heads, dropout=dropout, concat=concat)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        B, C, T, D_features = x.shape
        output_slices = []

        if self.graph_type == 'spatial':
            for b_idx in range(B):
                for t_idx in range(T):
                    node_features = x[b_idx, :, t_idx, :] 
                    gat_output_slice = self.leaky_relu(self.gat_conv(node_features, edge_index)) 
                    output_slices.append(gat_output_slice)
            
            stacked_output = torch.stack(output_slices, dim=0)
            output = stacked_output.reshape(B, T, C, -1).permute(0, 2, 1, 3) 

        elif self.graph_type == 'temporal':
            for b_idx in range(B):
                for c_idx in range(C):
                    node_features = x[b_idx, c_idx, :, :] 
                    gat_output_slice = self.leaky_relu(self.gat_conv(node_features, edge_index)) 
                    output_slices.append(gat_output_slice)
            
            stacked_output = torch.stack(output_slices, dim=0)
            output = stacked_output.reshape(B, C, T, -1)

        else:
            raise ValueError("Invalid graph_type. Must be 'spatial' or 'temporal'.")
            
        return output


class SS_EMERGE_Encoder(nn.Module):
    def __init__(self, F_bands, D_spectral, C_channels, T_timesteps,
                 gat_out_channels, tcn_channels, tcn_kernel_size, tcn_dilations,
                 dropout_prob=0.5):
        super(SS_EMERGE_Encoder, self).__init__()

        self.F_bands = F_bands
        self.D_spectral = D_spectral
        self.C_channels = C_channels
        # T_timesteps here refers to the final time dimension of the DE features, e.g., 265
        self.T_timesteps = T_timesteps 
        self.gat_out_channels = gat_out_channels
        self.tcn_channels = tcn_channels
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dilations = tcn_dilations
        self.dropout_prob = dropout_prob

        # 1. Spectral Modelling
        self.spectral_embedding = SpectralEmbedding(F_bands, D_spectral)

        # 2. Spatio-Temporal Modelling (GATs)
        self.spatial_gat = GATLayer(D_spectral, gat_out_channels, heads=4, dropout=dropout_prob, graph_type='spatial')
        self.temporal_gat = GATLayer(gat_out_channels, gat_out_channels, heads=4, dropout=dropout_prob, graph_type='temporal')
        
        # 3. Temporal Modelling (TCN)
        # TCN input channels will be C * gat_out_channels
        tcn_in_channels = C_channels * gat_out_channels
        tcn_layers = []
        current_tcn_in_channels = tcn_in_channels
        
        for i, dilation in enumerate(tcn_dilations):
            out_tcn_c = tcn_channels[i]
            padding_amount = (tcn_kernel_size - 1) * dilation
            tcn_layers.append(
                TCNBlock(current_tcn_in_channels, out_tcn_c, tcn_kernel_size,
                         dilation, padding_amount, dropout_prob)
            )
            current_tcn_in_channels = out_tcn_c
            
        self.tcn_sequence = nn.Sequential(*tcn_layers)
        self.final_tcn_output_dim = current_tcn_in_channels
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, spatial_edge_index=None, temporal_edge_index=None, mode='embedding'):
        B_eff, F, C, T = x.shape

        z_spectral = self.spectral_embedding(x) # (B_eff, C, T, D_spectral)
        z_spatial_gat = self.spatial_gat(z_spectral, spatial_edge_index) # (B_eff, C, T, gat_out_channels)
        z_temporal_gat = self.temporal_gat(z_spatial_gat, temporal_edge_index) # (B_eff, C, T, gat_out_channels)

        z_merged = z_temporal_gat

        # CORRECTED RESHAPE LOGIC:
        # From (B_eff, C, T, D_out) to (B_eff, C * D_out, T)
        # Permute to (B_eff, T, C, D_out)
        tcn_in = z_merged.permute(0, 2, 1, 3)
        # Reshape to (B_eff, T, C*D_out)
        tcn_in = tcn_in.reshape(B_eff, T, self.C_channels * self.gat_out_channels)
        # Transpose to (B_eff, C*D_out, T)
        tcn_in = tcn_in.transpose(1, 2)
        
        # A simpler, single-line version of the above:
        # tcn_in = z_merged.permute(0, 2, 1, 3).reshape(B_eff, self.T_timesteps, -1).transpose(1,2)

        z_tcn_out = self.tcn_sequence(tcn_in)
        final_embedding = self.global_pool(z_tcn_out).squeeze(-1)

        return final_embedding