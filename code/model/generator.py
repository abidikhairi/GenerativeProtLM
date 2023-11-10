"""Provides classes for language generation"""
from torch import nn

from .config import LMHeadConfig


class LinearLMHead(nn.Module):
    """
    Linear Language Model Head.

    Args:
        config (LMHeadConfig): Configuration for the Language Model Head.

    This class represents a linear head for a language model in a neural network. It consists of a linear layer
    mapping the hidden states to logits for each token in the vocabulary.

    Attributes:
        linear (nn.Linear): Linear layer for language modeling.
    """

    def __init__(self, config: LMHeadConfig) -> None:
        """
        Initializes a new LinearLMHead instance.

        Args:
            config (LMHeadConfig): Configuration for the Language Model Head.
        """
        super().__init__()

        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        """
        Applies the forward pass of the Linear Language Model Head.

        Args:
            x (th.Tensor): The input tensor representing hidden states.

        Returns:
            th.Tensor: The output logits for each token in the vocabulary.
        """
        return self.linear(x)
