"""Transformer decoder modules classes/function"""
from typing import Tuple, Union

import torch as th
from torch import nn
from xformers.components.positional_embedding import RotaryEmbedding

from utils.cloning import clones

from .config import DecoderConfig, DecoderLayerConfig


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Args:
        config (DecoderLayerConfig): Configuration for the decoder layer.

    This class represents a single layer of a Transformer decoder in a neural network model.

    Attributes:
        config (DecoderLayerConfig): The configuration for this decoder layer.
    """

    def __init__(self, config: DecoderLayerConfig):
        """
        Initializes a new instance of the DecoderLayer.

        Args:
            config (DecoderLayerConfig): The configuration for the decoder layer.
        """
        super().__init__()

        self.config = config
        self.self_attn = th.nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )

        self.pe = PositionalEncoding(d_model=config.d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_ff),
            nn.ReLU(),
            nn.Linear(config.dim_ff, config.d_model)
        )

        self.attn_norm = nn.LayerNorm(config.d_model)
        self.feedforward_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: th.Tensor, mask: Union[th.BoolTensor, th.Tensor] = None):
        """
        Applies the forward pass of the decoder layer.

        Args:
            x (th.Tensor): The input tensor.
            mask (th.Tensor): The input mask.
        """
        query_pe, key_pe = self.pe(x, x)
        q = x + query_pe
        k = x + key_pe

        v = self.attn_norm(x)  # pre-normalization
        q = self.attn_norm(q)
        k = self.attn_norm(k)

        z, _ = self.self_attn(q, k, v, attn_mask=mask,
                              need_weights=False, is_causal=True)
        x = x + z

        x = self.feedforward_norm(x)  # pre-normalization
        x = x + self.feedforward(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer Models.

    Args:
        d_model (int): The dimension of the model's hidden states.

    This class implements positional encoding for Transformer models. It applies positional information
    to the input tensors `q` and `k`.

    Attributes:
        positional_encoding (RotaryEmbedding): The positional encoding module used for adding positional information.
    """

    def __init__(self, d_model: int) -> None:
        """
        Initializes a new PositionalEncoding instance.

        Args:
            d_model (int): The dimension of the model's hidden states.
        """
        super().__init__()
        self.positional_encoding = RotaryEmbedding(dim_model=d_model)

    def forward(self, q: th.Tensor, k: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Applies positional encoding to the input tensors.

        Args:
            q (th.Tensor): The query tensor.
            k (th.Tensor): The key tensor.

        Returns:
            Tuple[th.Tensor, th.Tensor]: The query and key tensors with positional encoding applied.
        """
        q, k = self.positional_encoding(q, k)

        q = q.squeeze(0)
        k = k.squeeze(0)

        return q, k


class Embedding(nn.Module):
    """
    Token Embedding Layer for Transformer Models.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model's hidden states.
        padding_idx (int): Index for padding tokens.

    This class represents the word embedding layer used in Transformer models, which maps input tokens to
    continuous embeddings and scales them by a factor.

    Attributes:
        embeddings (nn.Embedding): The embedding layer mapping tokens to continuous vectors.
        scale_factor (th.Tensor): The scaling factor for the embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int) -> None:
        """
        Initializes a new EmbeddingLayer instance.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model's hidden states.
        """
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx)
        self.scale_factor = th.tensor(1 / (d_model ** 0.5))

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        """
        Applies word embedding and scaling to the input tensor.

        Args:
            inputs (th.Tensor): The input tensor containing token indices.

        Returns:
            th.Tensor: The embedded and scaled tensor.
        """
        return self.embeddings(inputs) * self.scale_factor


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder Module.

    Args:
        config (DecoderConfig): Configuration for the Transformer Decoder.

    This class represents a Transformer decoder in a neural network model. It consists of multiple decoder layers
    and an embedding layer for processing input sequences.

    Attributes:
        layers (nn.ModuleList): List of Transformer decoder layers.
        embedding (Embedding): Word embedding layer for the input sequences.
    """

    def __init__(self, config: DecoderConfig) -> None:
        """
        Initializes a new TransformerDecoder instance.

        Args:
            config (DecoderConfig): Configuration for the Transformer Decoder.
        """
        super().__init__()

        d_model = config.layer_conf.d_model
        layer = DecoderLayer(config.layer_conf)
        self.layers = clones(layer, config.num_layers)

        self.embedding = Embedding(
            vocab_size=config.vocab_size, d_model=d_model, padding_idx=config.padding_idx)

    def forward(self, inputs: th.Tensor, mask: th.Tensor):
        """
        Applies the forward pass of the Transformer Decoder.

        Args:
            inputs (th.Tensor): The input tensor representing token indices.
            mask (th.Tensor): The input mask.

        Returns:
            th.Tensor: The output tensor after passing through the Transformer Decoder.
        """
        x = self.embedding(inputs)

        for layer in self.layers:
            x = layer(x, mask)

        return x
