from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch


class mnist(Module):
    def __init__(self, num_classes, label = None):
        super(mnist, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

        
        self.label = label

    def forward(self, x):
        x = (x -0.5)*2
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def hidden_act(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        aa = torch.nn.Sequential(*list(self.classifier.children())[:-2])
        x = aa(x)

        return x


    @property
    def name(self):
        return (
            'LeNet'
            '-{label}'
        ).format(
            label=self.label
        )

    
