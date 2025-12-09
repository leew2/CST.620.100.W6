import torch
import torch.nn as nn

def get_models():
    gen = Generator()
    dis = Discriminator()
    return gen, dis

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        # use ConvTranspose2d to upsample back to the input spatial size
        self.conv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        # Use adaptive pooling so the classifier doesn't depend on input image size
        self.pool = nn.AdaptiveAvgPool2d(1)
        # after pooling the feature map will be (batch, 64, 1, 1) -> flattened to (batch, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # pool to (batch, 64, 1, 1) then flatten to (batch, 64)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        # sanity check: ensure features match linear layer in_features
        if x.dim() != 2:
            raise RuntimeError(f"Discriminator expected 2D input before fc, got {x.dim()}D tensor with shape {tuple(x.shape)}")
        if x.shape[1] != self.fc.in_features:
            raise RuntimeError(f"Discriminator fc expects in_features={self.fc.in_features}, but got input features={x.shape[1]} (input shape={tuple(x.shape)})")
        x = torch.sigmoid(self.fc(x))
        return x

