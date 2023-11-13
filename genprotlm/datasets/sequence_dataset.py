"""Protein Sequences Datasets"""
import pandas as pd
import torch as th
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer


class ProteinSequenceDataset(Dataset):
    """
    PyTorch Dataset for handling protein sequences.

    This dataset class is designed to work with protein sequences loaded from a CSV file.

    Example:
    ```python
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained('path/to/tokenizer')
    dataset = ProteinSequenceDataset(seq_file='path/to/sequences.csv', tokenizer=tokenizer)
    sample = dataset[0]
    ```

    Args:
        seq_file (str): The path to the CSV file containing protein sequences.
        tokenizer: The tokenizer to be used for tokenizing protein sequences.

    Note:
    Ensure that your tokenizer has a method to tokenize protein sequences appropriately.

    Methods:
        __init__(seq_file, tokenizer): Initializes an instance of the ProteinSequenceDataset class.
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(index): Retrieves a tokenized protein sequence from the dataset based on the given index.

    Attributes:
        frame (pd.DataFrame): The DataFrame containing protein sequences loaded from the CSV file.
        tokenizer: The tokenizer used for tokenizing protein sequences.
    """

    def __init__(self, seq_file: str, tokenizer) -> None:
        """
        Initializes a new instance of the ProteinSequenceDataset class.

        Args:
            seq_file (str): The path to the CSV file containing protein sequences.
            tokenizer: The tokenizer to be used for tokenizing protein sequences.
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
        return len(self.frame)

    def __getitem__(self, index):
        """
        Retrieves a tokenized protein sequence from the dataset based on the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Dict[str, th.Tensor]: The tokenized protein sequence.
        """
        if th.is_tensor(index):
            index = index.tolist()

        sequence = self.frame.iloc[index, 0]

        sequence = self.tokenizer(sequence, truncation=True, max_length=500)

        return sequence


def get_dataloader(seq_file: str, tokenizer: AutoTokenizer, batch_size: int = 16, num_workers: int = 2):
    """
    Creates a DataLoader for protein sequence data.

    Args:
        seq_file (str): The path to the CSV file containing protein sequences.
        tokenizer (AutoTokenizer): The tokenizer to be used for tokenizing protein sequences.
        batch_size (int, optional): The batch size for the DataLoader (default is 16).
        num_workers (int, optional): The number of worker processes for data loading (default is 2).

    Returns:
        DataLoader: A PyTorch DataLoader configured for protein sequence data.

    Example:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_model")
    dataloader = get_dataloader(seq_file='path/to/sequences.csv', tokenizer=tokenizer, batch_size=32, num_workers=4)
    for batch in dataloader:
        # Your training loop logic here
    ```
    """
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ProteinSequenceDataset(seq_file=seq_file, tokenizer=tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors='pt')

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=num_workers)

    return dataloader
