import os

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--num_grams", type=int, default=4)
parser.add_argument("--text", type=str)
parser.add_argument("--arpa", type=str, default='lm')

args = parser.parse_args()

output = f"{args.arpa}_{args.num_grams}gram.arpa"

os.system(f"lmplz -o {args.num_grams} --text {args.text} --arpa {output}")