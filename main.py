""""Module provides functions to train GPLM"""
from typing import Any
from pytorch_lightning import Trainer
from transformers import PreTrainedTokenizerFast

from genprotlm.utils import get_default_parser
from genprotlm.model.config import DecoderConfig, DecoderLayerConfig, LMHeadConfig, OptimizerConfig
from genprotlm.model import GProtLM
from genprotlm.datasets import get_dataloader


def main(args: Any):
    """
    Main function for training the protein language model (GenProtLM)

    Args:
        args (Any): Command line parameters passed to train the script
    """

    vocab_size = 23 + 1
    d_model = 256
    nhead = 4
    dropout = 0.6
    dim_ff = 128
    num_layers = 2
    pading_idx = 3
    learning_rate = 0.0003
    betas = (0.9, 0.998)
    warmp_steps = 500
    batch_size = 16
    num_workers = 2


    layer_conf = DecoderLayerConfig(d_model=d_model, nhead=nhead, dropout=dropout, dim_ff=dim_ff)
    model_conf = DecoderConfig(num_layers=num_layers, padding_idx=pading_idx, vocab_size=vocab_size, layer_conf=layer_conf)
    head_conf = LMHeadConfig(vocab_size=vocab_size, d_model=d_model)
    optim_conf = OptimizerConfig(d_model=d_model, warmup=warmp_steps, learning_rate=learning_rate, betas=betas)

    model = GProtLM(model_conf, head_conf, optim_conf)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    dataloader = get_dataloader(args.training_file, tokenizer, batch_size=batch_size, num_workers=num_workers)

    trainer = Trainer(accelerator='cpu')
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = get_default_parser()

    parser.add_argument('--training-file', type=str, required=True, help="The protein sequence file")
    parser.add_argument('--tokenizer-path', type=str, required=True, help="The path to the pretrained tokenizer")
    parser.add_argument('--model-checkpoint-path', type=str, required=True, help="Model checkpoints path")

    cli_args = parser.parse_args()

    main(cli_args)
