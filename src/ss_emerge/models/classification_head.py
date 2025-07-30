import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout_prob=0.5):
        super(ClassificationHead, self).__init__()

        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_prob) # Dropout applied after 1st layer

        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob) # Dropout applied after 2nd layer

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_prob) # Dropout applied after 3rd layer

        self.class_logits = nn.Linear(128, num_classes) # Final layer outputs logits

    def forward(self, x):
        # x: (B, in_features)
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.drop3(self.relu3(self.bn3(self.fc3(x))))
        
        logits = self.class_logits(x)
        return logits