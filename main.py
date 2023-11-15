""""Module provides functions to train GPLM"""
from typing import Any
from pytorch_lightning import Trainer, loggers, callbacks
from transformers import PreTrainedTokenizerFast
from clearml import Task, TaskTypes

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

    Task.init(project_name="protein-language-modeling",
              task_name="GenProtLM Training V0", task_type=TaskTypes.training)
    Task.current_task().add_tags(
        ['gpt', 'pre-normalization', 'human-proteome', 'refseq'])

    vocab_size = 23 + 1 # <UNK> token for compatibility reasons
    d_model = args.d_model
    nhead = args.nhead
    dropout = args.dropout
    dim_ff = args.dim_ff
    num_layers = args.num_layers
    padding_idx = 3 # fixed from the tokenizer
    learning_rate = args.learning_rate
    betas = tuple(args.betas)
    warmup_steps = args.warmup_steps
    batch_size = args.batch_size
    num_workers = args.num_workers

    layer_conf = DecoderLayerConfig(
        d_model=d_model, nhead=nhead, dropout=dropout, dim_ff=dim_ff)
    model_conf = DecoderConfig(
        num_layers=num_layers, padding_idx=padding_idx, vocab_size=vocab_size, layer_conf=layer_conf)
    head_conf = LMHeadConfig(vocab_size=vocab_size, d_model=d_model)
    optim_conf = OptimizerConfig(
        d_model=d_model, warmup=warmup_steps, learning_rate=learning_rate, betas=betas)

    model = GProtLM(model_conf, head_conf, optim_conf)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    train_dataloader = get_dataloader(
        args.training_file, tokenizer, batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = get_dataloader(
        args.validation_file, tokenizer, batch_size=batch_size, num_workers=num_workers)

    csv_logger = loggers.CSVLogger(save_dir="logs", name="genprotlm_v0")

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_path, filename="gplm-v0-{epoch}-{step}", save_top_k=1)

    trainer = Trainer(accelerator='gpu', logger=csv_logger,
                      callbacks=[checkpoint_callback], max_epochs=5,
                      log_every_n_steps=250)

    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    parser = get_default_parser()

    parser.add_argument('--d-model', type=int, default=256,
                        help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float,
                        default=0.6, help='Dropout rate')
    parser.add_argument('--dim-ff', type=int, default=256,
                        help='Dimension of the feedforward layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--learning-rate', type=float,
                        default=0.0003, help='Learning rate')
    parser.add_argument('--betas', nargs=2, type=float,
                        default=(0.9, 0.998), help='Betas for Adam optimizer')
    parser.add_argument('--warmup-steps', type=int,
                        default=500, help='Warm-up steps')
    parser.add_argument('--batch-size', type=int,
                        default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--training-file', type=str,
                        required=True, help="The protein sequence file")
    parser.add_argument('--validation-file', type=str,
                        required=True, help="The protein sequence validation file")
    parser.add_argument('--tokenizer-path', type=str, required=True,
                        help="The path to the pretrained tokenizer")
    parser.add_argument('--checkpoint-path', type=str,
                        required=True, help="Model checkpoints path")

    cli_args = parser.parse_args()

    main(cli_args)
