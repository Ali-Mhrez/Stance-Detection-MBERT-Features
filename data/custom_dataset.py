import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """A custom dataset class to wrap encodings and labels for training and evaluation.

    Args:
        encodings (list): A list of tokenized inputs.
        labels (list): A list of labels corresponding to the tokenized inputs.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the tokenized inputs and the corresponding label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item