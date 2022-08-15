"""
Creates a MobileNet Model as defined in:
Andrew G. H., Menglong Z., Bo C., Dmitry K., Weijun W., Tobias W., Marco A., Hartwig A. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
arXiv preprint arXiv:1704.04861.
import from https://github.com/marvis/pytorch-mobilenet
"""

import torch.nn as nn

__all__ = ['mobilenetv1']


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.Softplus(beta=10)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Softplus(beta=10),
    
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.Softplus(beta=10),
    )


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, label = None):
        super(MobileNet, self).__init__()
        
        self.label = label

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = (x -0.5)*2
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        # x = nn.Softmax(dim=1)(x)
        return x

    def hidden_act(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        # x = self.fc(x)
        return x

    @property
    def name(self):
        return (
            'Mobilenet'
            '-{label}'
        ).format(
            label=self.label
        )

def mobilenetv1(**kwargs):
    """
    Constructs a MobileNet V1 model
    """
    return MobileNet(**kwargs)