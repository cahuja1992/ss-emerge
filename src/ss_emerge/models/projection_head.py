import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features=[1024, 2048, 4096], dropout_prob=0.5):
        super(ProjectionHead, self).__init__()
        layers = []
        current_features = in_features
        for i, h_dim in enumerate(hidden_features):
            layers.append(nn.Linear(current_features, h_dim))
            if i < len(hidden_features) - 1:
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_prob)) # Ensure dropout_prob is correctly used here
            current_features = h_dim

        self.mlp = nn.Sequential(*layers)
        self.final_features = hidden_features[-1]

    def forward(self, x):
        z = self.mlp(x)
        z = F.normalize(z, p=2, dim=1)
        return z