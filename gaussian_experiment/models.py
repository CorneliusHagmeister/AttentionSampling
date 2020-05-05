import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x =  self.net(x)
        return x

class Generator(nn.Module):
    def __init__(self,z_dim):
        super(Generator, self).__init__()
        
        self.z_dim=z_dim

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 2),
            nn.Tanh()
        )
    def forward(self, x):
        x =  self.net(x)
        return x