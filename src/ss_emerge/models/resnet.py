import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper functions for specific convolutional kernels
def conv1x15(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x15 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 15), stride=stride,
                     padding=(0, 7), groups=groups, bias=False, dilation=dilation)

def conv1x13(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x13 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 13), stride=stride,
                     padding=(0, 6), groups=groups, bias=False, dilation=dilation)

def conv1x11(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x11 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 11), stride=stride,
                     padding=(0, 5), groups=groups, bias=False, dilation=dilation)

def conv1x9(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=stride,
                     padding=(0, 4), groups=groups, bias=False, dilation=dilation)

def conv1x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 7), stride=stride,
                     padding=(0, 3), groups=groups, bias=False, dilation=dilation)

def conv1x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 5), stride=stride,
                     padding=(0, 2), groups=groups, bias=False, dilation=dilation)

def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)

def conv15x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """15x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(15, 1), stride=stride,
                     padding=(7, 0), groups=groups, bias=False, dilation=dilation)

def conv13x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """13x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(13, 1), stride=stride,
                     padding=(6, 0), groups=groups, bias=False, dilation=dilation)

def conv11x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """11x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(11, 1), stride=stride,
                     padding=(5, 0), groups=groups, bias=False, dilation=dilation)

def conv9x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """9x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(9, 1), stride=stride,
                     padding=(4, 0), groups=groups, bias=False, dilation=dilation)

def conv7x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(7, 1), stride=stride,
                     padding=(3, 0), groups=groups, bias=False, dilation=dilation)

def conv5x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(5, 1), stride=stride,
                     padding=(2, 0), groups=groups, bias=False, dilation=dilation)

def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride,
                     padding=(1, 0), groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, i, Layer, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Determine convolutional layers based on Layer parameter (from original code)
        if Layer == "1":
            if i == 1:
                self.conv1 = conv1x15(inplanes, planes, stride=(1, 2))
                self.conv2 = conv15x1(planes, planes, stride=(2, 1))
            else:
                self.conv1 = conv1x15(inplanes, planes, stride=(1, 1))
                self.conv2 = conv15x1(planes, planes)
        elif Layer == "2":
            if i == 1:
                self.conv1 = conv1x11(inplanes, planes, stride=(1, 2))
                self.conv2 = conv11x1(planes, planes, stride=(2, 1))
            else:
                self.conv1 = conv1x11(inplanes, planes, stride=(1, 1))
                self.conv2 = conv11x1(planes, planes)
        elif Layer == "3":
            if i == 1:
                self.conv1 = conv1x7(inplanes, planes, stride=(1, 2))
            else: # This block had a logic error in original code, self.conv1 was used for 7x1
                self.conv1 = conv7x1(inplanes, planes, stride=(1, 1)) # This was conv1x7 previously
            self.conv2 = conv3x1(planes, planes)
        elif Layer == "4":
            if i == 1:
                self.conv1 = conv1x3(inplanes, planes, stride=(1, 2))
            else:
                self.conv1 = conv1x3(inplanes, planes, stride=(1, 1))
            self.conv2 = conv5x5(planes, planes) # This was a 5x5 square conv, not 1D

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEEG(nn.Module):
    def __init__(self, in_channels=1, num_classes=4,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout_prob=0.2, output_embedding_dim=512): # Added dropout_prob and output_embedding_dim
        super(ResNetEEG, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._arch = 'resnet18' # Retaining this from original code for reference
        self.layers_config = [2, 2, 2, 2] # Number of blocks in each layer from original ResNet18 config
        self.block = BasicBlock
        
        self.inplanes = 64 # Initial channels after first conv
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.dropout_prob = dropout_prob # Storing dropout probability
        self.final_embedding_dim = output_embedding_dim # Storing final embedding dimension for external use

        # Initial convolution layer
        # Input to ResNetEEG is (B, in_channels, C, T)
        # Original code used (1, 9) kernel, (1, 2) stride, (0, 4) padding
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4),
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling layer (original code used (1,3) kernel, (1,2) stride)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        
        # ResNet Layers
        self.layer1 = self._make_layer("1", self.block, 64, self.layers_config[0])
        self.layer2 = self._make_layer("2", self.block, 128, self.layers_config[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer("3", self.block, 256, self.layers_config[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer("4", self.block, 512, self.layers_config[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # Adaptive average pooling to get fixed size output regardless of input time dimension
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces spatial and temporal dimensions to 1x1

        # --- Projection Head (for contrastive learning) ---
        # This is built into the original ResNet_model.py, used when mode='contrast'
        self.fc_projector1 = nn.Linear(512 * self.block.expansion, 1024 * self.block.expansion)
        self.bn_fc_p1 = norm_layer(1024 * self.block.expansion) # Original code had BatchNorm1d(1024) here, adjusting to 2D
        
        self.fc_projector2 = nn.Linear(1024 * self.block.expansion, 2048 * self.block.expansion)
        self.bn_fc_p2 = norm_layer(2048 * self.block.expansion) # Original code had BatchNorm1d(2048) here
        
        self.fc_projector3 = nn.Linear(2048 * self.block.expansion, 4096 * self.block.expansion)
        
        # --- Classification Head (for supervised learning) ---
        # Also built into the original ResNet_model.py, used when mode='classifier'
        self.fc_classifier1 = nn.Linear(512 * self.block.expansion, 256 * self.block.expansion)
        self.bn_fc1 = norm_layer(256 * self.block.expansion) # Original code had BatchNorm1d(256) here
        
        self.fc_classifier2 = nn.Linear(256 * self.block.expansion, 128 * self.block.expansion)
        self.bn_fc2 = norm_layer(128 * self.block.expansion) # Original code had BatchNorm1d(128) here
        
        self.fc_classifier3 = nn.Linear(128 * self.block.expansion, num_classes) # Final classifier output

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, Layer, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # Determine stride for downsample based on Layer (from original code)
        if Layer == "1":
            stride = (2, 2)
        elif Layer == "2":
            stride = (2, 2)
        elif Layer == "3":
            stride = (1, 2)
        elif Layer == "4":
            stride = (1, 2)
        
        # Create downsample layer if needed
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Add first block (with downsample if needed)
        layers.append(block(1, Layer, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        # Add remaining blocks
        for i in range(2, blocks + 1):
            layers.append(block(i, Layer, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, spatial_edge_index=None, temporal_edge_index=None, mode='embedding'):
        """
        Forward pass for ResNetEEG.
        Args:
            x (torch.Tensor): Input tensor. Expected shape (B, in_channels, C, T).
            spatial_edge_index (torch.Tensor, optional): Ignored by ResNetEEG.
            temporal_edge_index (torch.Tensor, optional): Ignored by ResNetEEG.
            mode (str): 'embedding' to get the feature vector before heads,
                        'classifier' to get classification logits,
                        'contrast' to get contrastive projection.
        Returns:
            torch.Tensor: Output based on the specified mode.
        """
        # Feature extraction backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # Output shape (B, 512, 1, 1)
        x = torch.flatten(x, 1) # Flatten to (B, 512*block.expansion) = (B, 512)

        if mode == 'classifier':
            x = self.fc_classifier1(x)
            x = self.bn_fc1(x)
            x = self.relu(x)
            x = self.fc_classifier2(x)
            x = self.bn_fc2(x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training) # Apply dropout
            x = self.fc_classifier3(x)
            return x
        elif mode == 'contrast':
            x = self.fc_projector1(x)
            x = self.bn_fc_p1(x)
            x = self.relu(x) # Added ReLU as per standard projection head patterns
            x = F.dropout(x, p=self.dropout_prob, training=self.training) # Apply dropout
            x = self.fc_projector2(x)
            x = self.bn_fc_p2(x)
            x = self.relu(x) # Added ReLU
            x = F.dropout(x, p=self.dropout_prob, training=self.training) # Apply dropout
            x = self.fc_projector3(x)
            return x
        elif mode == 'embedding':
            return x
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'classifier', 'contrast', or 'embedding'.")
        