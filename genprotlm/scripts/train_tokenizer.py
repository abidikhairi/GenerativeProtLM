"""Train a tokenizer"""
import argparse
from tokenizers import Tokenizer, processors, models
from transformers import PreTrainedTokenizerFast

def main(save_path: str):
    """
    Creates a Protein FastTokenizer compatible with Hugging Face's transformers library and saves it.

    Args:
        save_path (str): The path where the trained tokenizer will be saved.

    This script sets up a Protein FastTokenizer using the `tokenizers` library, adds special tokens,
    and saves the trained tokenizer using the `PreTrainedTokenizerFast` from Hugging Face's transformers library.
    The resulting tokenizer is compatible with the Hugging Face's transformers library.

    Usage:
        python script_name.py --save-path /path/to/save/tokenizer

    Args:
        --save-path (str): The path where the trained tokenizer will be saved.
    """
    tokenizer = Tokenizer(models.Unigram())
    special_tokens = ['<sos>', '<eos>', '<pad>']

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(list("ACDEFGHIKLMNPQRSTVWY"))

    sos_token_id = tokenizer.token_to_id('<sos>')
    eos_token_id = tokenizer.token_to_id('<eos>')

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<sos>:0 $A:0 <eos>:0",
        special_tokens=[('<sos>', sos_token_id), ('<eos>', eos_token_id)]
    )

    trained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, sos_token="<sos>", eos_token="<eos>", pad_token="<pad>")

    trained_tokenizer.save_pretrained(save_path)

    loaded_tknzr = PreTrainedTokenizerFast.from_pretrained(save_path)
    print(loaded_tknzr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Creates a Protein FastTokenizer compatible with Huggingfaces")

    parser.add_argument('--save-path', type=str,
                        required=True, help="tokenizer save path")

    args = parser.parse_args()

    main(args.save_path)
