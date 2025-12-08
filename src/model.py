import torch
import torch.nn as nn

def get_models():
    gen = Generator()
    dis = Discriminator()
    return gen, dis

class Generator(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(64 * 128 * 128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc(x))
        return x

