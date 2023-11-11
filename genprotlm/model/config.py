"""Model configuration"""
from typing import Tuple
from pydantic import BaseModel


class DecoderLayerConfig(BaseModel):
    """
    Configuration for a decoder layer.

    This class defines the configuration parameters for a decoder layer in a neural network model.

    Attributes:
        d_model (int): The dimension of the model's hidden states (default is 1024).
        nhead (int): The number of attention heads in the layer (default is 8).
        num_layers (int): The number of layers in the decoder (default is 4).
        dim_ff (int): The dimension of the feed-forward network (default is 2048).
        dropout (float): The dropout probability (default is 0.6).
    """
    d_model: int = 1024
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 2048
    dropout: float = 0.6


class DecoderConfig(BaseModel):
    """
    Configuration for a Transformer Decoder.

    This class defines the configuration parameters for a Transformer decoder in a neural network model.

    Attributes:
        layer_conf (DecoderLayerConfig): The configuration for a single decoder layer.
        num_layers (int): The number of decoder layers in the model (default is 4).
        padding_idx (int): The index for padding tokens (default is -1).
        vocab_size (int): The size of the vocabulary (default is 20).
    """
    layer_conf: DecoderLayerConfig = DecoderLayerConfig()
    num_layers: int = 4
    padding_idx: int = -1
    vocab_size: int = 20


class LMHeadConfig(BaseModel):
    """
    Configuration for the Language Model Head.

    This class defines the configuration parameters for the language model head in a neural network model.

    Attributes:
        vocab_size (int): The size of the vocabulary (default is 20).
        d_model (int): The dimension of the model's hidden states (default is 20).
    """
    vocab_size: int = 20
    d_model: int = 1024


class OptimizerConfig(BaseModel):
    """
    Configuration for the Transformer Optimizer.

    This class defines the configuration parameters for the optimizer in a neural network model using the
    Transformer architecture.

    Attributes:
        d_model (int): The dimension of the model's hidden states (default is 1024).
        warmup (int): The number of warmup steps for learning rate scheduling (default is 2000).
    """
    d_model: int = 1024
    warmup: int = 2000
    learning_rate: float = 0.0003
    betas: Tuple[float, float] = (0.9, 0.998)
