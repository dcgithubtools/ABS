import torch.nn as nn
import torch.nn.functional as F
import archs.arch_util as arch_util
import torch
from utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class FSRCNN(torch.nn.Module):
    def __init__(self, input_channels, scale, d, s, m):
        super(FSRCNN, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []

        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
 

        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                                            stride=scale, padding=3, output_padding=1)


        # arch_util.default_init_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
        return out