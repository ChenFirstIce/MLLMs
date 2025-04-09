import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, adni, brast, ppmi, biomarkers):
        self.adni = adni
        self.brast = brast
        self.ppmi = ppmi
        self.biomarkers = biomarkers

    def __len__(self):
        return len(self.adni)

    def __getitem__(self, idx):
        return self.adni[idx], self.brast[idx], self.ppmi[idx], self.biomarkers[idx]