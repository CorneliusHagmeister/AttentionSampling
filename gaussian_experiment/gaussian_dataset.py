import numpy as np
from torch.utils.data import Dataset
import torch

class GaussianDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
        self.dataset=np.load(root_dir)
        self.dataset = torch.from_numpy(self.dataset)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample= self.dataset[idx]

        return sample