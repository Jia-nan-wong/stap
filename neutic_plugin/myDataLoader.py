import random, sys, os, copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDatasetPlugin(Dataset):
    def __init__(self, length, labels):
        self.data = torch.tensor(length, dtype=int)
        self.labels = torch.tensor(labels, dtype=int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]