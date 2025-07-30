import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean") # For the infoNCE formulation

    def forward(self, z_i, z_j):
        """
        Calculates the NT-Xent loss (normalized temperature-scaled cross-entropy loss).
        Adapted from kanhaoning/self-supervised-group-meiosis-contrastive-learning-for-eeg-based-emotion-recognition/SSL_training(SEED).py.

        Args:
            z_i (torch.Tensor): Embeddings from the first augmented view (B, D).
            z_j (torch.Tensor): Embeddings from the second augmented view (B, D).
                                 B is the number of positive pairs in the batch.

        Returns:
            torch.Tensor: The scalar NT-Xent loss.
        """
        # Ensure embeddings are L2-normalized
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        B_size = z_i.shape[0]
        
        # Calculate specific similarity matrices as per original code
        # logits_ab: similarities between z_i and z_j (positive pairs for anchors in z_i)
        # logits_aa: similarities within z_i (negatives for anchors in z_i, excluding self)
        logits_ab = torch.matmul(z_i, z_j.T) / self.temperature  # (B, B)
        logits_ba = torch.matmul(z_j, z_i.T) / self.temperature  # (B, B)
        logits_aa = torch.matmul(z_i, z_i.T) / self.temperature  # (B, B)
        logits_bb = torch.matmul(z_j, z_j.T) / self.temperature  # (B, B)
        
        # Mask self-similarity (diagonal) for `aa` and `bb` logits to a very small negative number
        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(B_size, device=z_i.device), B_size) # (B, B) Identity matrix
        
        logits_aa = logits_aa - masks * LARGE_NUM # Set diagonal to -inf
        logits_bb = logits_bb - masks * LARGE_NUM # Set diagonal to -inf

        # Labels for CrossEntropyLoss: target is the index of the positive sample.
        # For torch.cat([logits_ab, logits_aa], dim=1), the positive sample is at index `i` (diagonal)
        # in the `logits_ab` part. So labels are simply `[0, 1, ..., B-1]`.
        labels = torch.arange(B_size, device=z_i.device, dtype=torch.long)
        
        # Loss for anchors from z_i (first view)
        loss_a = self.criterion(torch.cat([logits_ab, logits_aa], dim=1), labels) # Input to criterion is (B, 2B)

        # Loss for anchors from z_j (second view)
        loss_b = self.criterion(torch.cat([logits_ba, logits_bb], dim=1), labels) # Input to criterion is (B, 2B)
        
        # Total loss is the average of loss_a and loss_b as per SimCLR paper (L = (L_a + L_b) / 2P)
        # Since criterion already takes mean, we just average the two means.
        loss = (loss_a + loss_b) / 2 # Corrected averaging

        return loss
    