import torch
import logging

logger = logging.Logger(__name__)

class TwoNN(torch.nn.Module): # McMahan et al., 2016; 199,210 parameters
    def __init__(self, resize, hidden_size, num_classes):
        super(TwoNN, self).__init__()
        self.in_features = resize**2
        self.num_hiddens = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=self.in_features, out_features=self.num_hiddens, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=self.num_hiddens, out_features=self.num_hiddens, bias=True),
            torch.nn.ReLU(True)
        )
        self.classifier = torch.nn.Linear(in_features=self.num_hiddens, out_features=self.num_classes, bias=True)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TwoCNN(torch.nn.Module): # McMahan et al., 2016; 1,663,370 parameters
    def __init__(self, in_channels, hidden_size, num_classes):
        super(TwoCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=(self.hidden_channels * 2) * (7 * 7), out_features=512, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
