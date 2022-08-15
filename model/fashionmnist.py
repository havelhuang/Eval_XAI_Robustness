from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch


class FashionCNN(nn.Module):
    
    def __init__(self,num_classes, label = None):
        super(FashionCNN, self).__init__()

        self.label = label
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=3136, out_features=800)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=800, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes)
        
    def forward(self, x):
        x = (x -0.5)*2
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def hidden_act(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

    @property
    def name(self):
        return (
            'FashionCNN'
            '-{label}'
        ).format(
            label=self.label
        )