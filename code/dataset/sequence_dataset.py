"""Protein Sequences Datasets"""
from torch.utils.data import Dataset


class ProteinSequenceDataset(Dataset):
    """
    PyTorch Dataset for handling protein sequences.

    This dataset class is designed to work with protein sequences. It currently provides a minimal implementation
    with a single placeholder sample.

    Example:
    ```python
    dataset = ProteinSequenceDataset()
    sample = dataset[0]
    ```

    Note:
    For a functional dataset, you should override the methods `__init__`, `__len__`, and `__getitem__` to suit
    your specific data loading needs.

    Methods:
        __init__(): Initializes an instance of the ProteinSequenceDataset class.
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(index): Retrieves a sample from the dataset based on the given index.

    Attributes:
        No specific attributes are defined in this base implementation.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the ProteinSequenceDataset class.
        """
        super().__init__()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return 1

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset based on the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Any: The retrieved sample.
        """
        return super().__getitem__(index)
