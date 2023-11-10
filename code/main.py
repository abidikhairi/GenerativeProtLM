""""Module provides functions to train GPLM"""
from utils import get_default_parser
from utils import get_default_logger
from model.config import DecoderConfig, DecoderLayerConfig
from model.decoder import TransformerDecoder
import torch as th


def main(args):
    logger = get_default_logger(__name__, 'logs/main.log')

    logger.info("hello world !")
    logger.warning("hello world !")
    logger.error("hello world !")
    
    layer_conf = DecoderLayerConfig(d_model=256, nhead=4, num_layers=4, dropout=0.6, dim_ff=128)
    model_conf = DecoderConfig(num_layers=2, padding_idx=0, vocab_size=20, layer_conf=layer_conf)
    
    batch_size = 16
    seq_len = 50

    input_seq = th.randint(low=0, high=20, size=(batch_size, seq_len))
    mask = th.triu(th.ones(seq_len, seq_len), diagonal=1)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))

    model = TransformerDecoder(model_conf)
    print(model)

    out = model(input_seq, mask)
    print(out.shape)


if __name__ == '__main__':
    parser = get_default_parser()

    args = parser.parse_args()

    main(args)
