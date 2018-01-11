'''
[1] Long, J., Shelhamer, E., & Darrell, T. (2015). 
    Fully convolutional networks for semantic segmentation. 
    In Proceedings of the IEEE CVPR (pp. 3431-3440).
'''


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):

    def __init__(self):
        super(FCNN, self).__init__()
        # Learnable layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        nn.init.kaiming_normal(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        nn.init.kaiming_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        nn.init.kaiming_normal(self.conv3.weight)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, padding=2)
        nn.init.kaiming_normal(self.conv4.weight)
        self.upsample_mode = 'nearest'

    # to experiment with other upsampling techniques
    def set_upsample_mode(upsample_mode='nearest'):
        if (upsample_mode in ['nearest', 'linear', 'bilinear', 'trilinear']):
            self.upsample_mode = upsample_mode


    def forward(self, x):
        # x.size() = (N, 3, W, W) 
        x = F.relu(self.conv1(x)) 
        # x.size() = (N, 16, W, W) 
        x = F.relu(self.conv2(x))
        # x.size() = (N, 32, W, W) 
        x = F.max_pool2d(x, (2,2))
        # x.size() = (N, 32, W/2, W/2)
        x = F.relu(self.conv3(x))
        # x.size() = (N, 16, W/2, W/2)
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode) 
        # x.size() = (N, 16, W, W)
        x = self.conv4(x)
        # x.size() = (N, 2, W, W)
        return x

