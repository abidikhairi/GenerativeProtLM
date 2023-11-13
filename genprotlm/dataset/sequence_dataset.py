"""Protein Sequences Datasets"""
import pandas as pd
from torch.utils.data import Dataset


class ProteinSequenceDataset(Dataset):
    """
    PyTorch Dataset for handling protein sequences.

    This dataset class is designed to work with protein sequences. It currently provides a minimal implementation
    with a single placeholder sample.

    Example:
    ```python
    dataset = ProteinSequenceDataset()
    s1, s2 = dataset[0]
    print(s1)
    # prints: ACBQVTAAAV
    print(s1)
    # prints: CBQVTAAAVB
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

    def __init__(self, seq_file: str, tokenizer) -> None:
        """
        Initializes a new instance of the ProteinSequenceDataset class.
        """
        super().__init__()
        self.frame = pd.read_csv(seq_file, sep=";", header=None)
        self.tokenizer = tokenizer

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
        sequence = list(self.frame.iloc[index, 0])
        import pdb; pdb.set_trace()
        sequence = [self.tokenizer.sos_token] + sequence + [self.tokenizer.eos_token]

        return sequence


if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/flursky/Work/rnd/phd/GenerativeProtLM/models/tokenizer')
    seq_file = "/home/flursky/Work/rnd/phd/GenerativeProtLM/data/refseq_human_proteome_23_11_04_23_00_51.csv"
    dataset = ProteinSequenceDataset(seq_file, tokenizer)

    print(dataset[0])