import torch
from pretraining.byol import BYOL
import torchsummary
from src.loss import l2_distance
from preprocessing.processor import ConformerProcessor
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--data_path", type=str)
parser.add_argument("--n_mel_channels", type=int, default=80)
parser.add_argument("--n", type=int, default=17)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--kernel_size", type=int, default=31)
parser.add_argument("--eps", type=float, default=1e-5)
parser.add_argument("--alpha", type=float, default=0.99)

args = parser.parse_args()

df = pd.read_csv(args.data_path, sep="\t")

processor = ConformerProcessor('./vocabulary/grapheme.json')

model = BYOL(
    n_mel_channels=args.n_mel_channels,
    n=args.n,
    d_model=args.d_model,
    heads=args.heads,
    kernel_size=args.kernel_size,
    eps=args.eps,
    alpha=args.alpha
)