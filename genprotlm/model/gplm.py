"""Module providing Generative Protein Language Model functions"""
from typing import Optional, Union, List, Tuple

import pytorch_lightning as pl
import torch as th
from torch import optim
from torch.optim import lr_scheduler

from .config import DecoderConfig, LMHeadConfig, OptimizerConfig
from .decoder import TransformerDecoder
from .generator import LinearLMHead
from .utils import scheduler_fn


class GProtLM(pl.LightningModule):
    """
    GProtLM: A PyTorch Lightning Module for a Transformer Language Model.

    Args:
        model_config (DecoderConfig): Configuration for the Transformer Decoder.
        head_config (LMHeadConfig): Configuration for the Language Model Head.
        optim_config (OptimizerConfig): Configuration for the Optimizer.

    This class represents a PyTorch Lightning module for training a Transformer language model. It consists of a
    Transformer decoder, a linear language model head, and configurations for the model, head, and optimizer.
    """

    def __init__(self, model_config: DecoderConfig, head_config: LMHeadConfig, optim_config: OptimizerConfig) -> None:
        """
        Initializes a new GProtLM instance.

        Args:
            model_config (DecoderConfig): Configuration for the Transformer Decoder.
            head_config (LMHeadConfig): Configuration for the Language Model Head.
            optim_config (OptimizerConfig): Configuration for the Optimizer.
        """
        super().__init__()

        self.save_hyperparameters()

        self.model = TransformerDecoder(model_config)
        self.head = LinearLMHead(head_config)

        self.optim_conf = optim_config

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[lr_scheduler._LRScheduler]]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[optim.Optimizer], List[lr_scheduler._LRScheduler]]: Optimizer and LR scheduler.
        """
        optimizer = th.optim.AdamW(
            self.parameters(), lr=self.optim_conf.learning_rate, betas=self.optim_conf.betas)
        scheduler = th.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: scheduler_fn(step, self.optim_conf.d_model, self.optim_conf.warmup))
        return [optimizer], [scheduler]

    def forward(self, x: th.Tensor, attn_mask: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the GProtLM model.

        Args:
            x (th.Tensor): Input tensor representing token indices. Shape: (batch_size, sequence_length).
            attn_mask (th.Tensor): Attention mask tensor to mask padded tokens. Shape: (batch_size, sequence_length).

        Returns:
            Tuple[th.Tensor, th.Tensor]: Output logits and hidden states.
            - hidden_states (th.Tensor): Hidden states from the transformer model. Shape: (batch_size, sequence_length, d_model).
        """
        h = self.model(x, attn_mask)

        return h

    def training_step(self, batch, batch_idx) -> Optional[Union[th.Tensor, Tuple[th.Tensor, ...]]]:
        """
        Training step for GProtLM.

        Args:
            batch: The batch of data.
            batch_idx: Index of the batch.

        Returns:
            Optional[Union[th.Tensor, Tuple[th.Tensor, ...]]]: Training step output.
        """
        vocab_size = self.hparams.model_config.vocab_size

        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask'].bool()
        labels = batch['labels']

        h = self(input_ids, attn_mask)
        z = self.head(h)

        loss = th.nn.functional.cross_entropy(z.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[Union[th.Tensor, Tuple[th.Tensor, ...]]]:
        """
        Validation step for GProtLM.

        Args:
            batch: The batch of data.
            batch_idx: Index of the batch.

        Returns:
            Optional[Union[th.Tensor, Tuple[th.Tensor, ...]]]: Validation step output.
        """
        vocab_size = self.hparams.model_config.vocab_size

        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask'].bool()
        labels = batch['labels']

        z, _ = self(input_ids, attn_mask)

        loss = th.nn.functional.cross_entropy(z.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

        return loss
