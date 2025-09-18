
import torch.nn as nn

from Trainer import *


class VGG11WithAPI(nn.Module):
    def __init__(self, arch, num_classes=46):
        super().__init__()
        conv_blks=[]
        for (num_convs, out_channels) in arch:
            conv_blks.append(self.vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(*conv_blks, nn.Flatten(),
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                                 nn.LazyLinear(num_classes))
        self.net.apply(self.init_cnn)

    def vgg_block(self, num_convs, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def init_cnn(self, module):
        """Initialize weights for CNNs.

        Defined in :numref:`sec_lenet`"""
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
    def forward(self, image, npy1):
        """Defined in :numref:`sec_linear_concise`"""
        concatenated = torch.cat((image, npy1), dim=1)

        return self.net(concatenated)

