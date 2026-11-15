import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # input is 4 stacked frames of 84x84, output after the convs is 64x7x7
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # one Q-value for each action (NOOP, UP, DOWN)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # frames come in as 0-255, scale them to 0-1
        x = x.float() / 255.0
        return self.fc_layers(self.conv_layers(x))
