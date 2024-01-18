import os
import torch
from preprocessing.processor import ConformerProcessor
from src.conformer import Conformer
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from src.metric import WER_score
parser = ArgumentParser()

parser.add_argument("--test_path", type=str)
parser.add_argument("--num_examples", type=int, default=None)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--arpa_path", type=str)

# Audio Processor Config
parser.add_argument("--num_mels", type=int, default=80)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--fft_size", type=int, default=400)
parser.add_argument("--hop_length", type=int, default=160)
parser.add_argument("--win_length", type=int, default=400)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)

# Text Processor Config
parser.add_argument("--vocab_path", type=str, default="./vocabulary/dictionary.json")
parser.add_argument("--pad_token", type=str, default="<pad>")
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--word_delim_token", type=str, default="|")

# Model Config
parser.add_argument("--encoder_n_layers", type=int, default=17)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--kernel_size", type=int, default=31)
parser.add_argument("--decoder_n_layers", type=int, default=1)
parser.add_argument("--decoder_dim", type=int, default=640)
parser.add_argument("--dropout_rate", type=float, default=0.1)

# Path
parser.add_argument("--result_folder", type=str, default='./results')
parser.add_argument("--saved_name", type=str, default=None)

args = parser.parse_args()

if os.path.exists(args.result_folder) == False:
    os.mkdir(args.result_folder)

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Processor Setup
processor = ConformerProcessor(
    vocab_path=args.vocab_path,
    num_mels=args.num_mels,
    sampling_rate=args.sampling_rate,
    n_fft=args.fft_size,
    hop_length=args.hop_length,
    win_length=args.win_length,
    fmin=args.fmin,
    fmax=args.fmax,
    pad_token=args.pad_token,
    unk_token=args.unk_token,
    word_delim_token=args.word_delim_token,
    lm_path=args.arpa_path
)

# Model Setup
model = Conformer(
    vocab_size=len(processor.dictionary),
    n_mel_channels=processor.num_mels,
    encoder_n_layers=args.encoder_n_layers,
    encoder_dim=args.encoder_dim,
    heads=args.heads,
    kernel_size=args.kernel_size,
    decoder_n_layers=args.decoder_n_layers,
    decoder_dim=args.decoder_dim,
    dropout_rate=args.dropout_rate
).to(device)

model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
model.to(device)
model.eval()

df = pd.read_csv(args.test_path, sep="\t")
if args.num_examples is not None:
    df = df[:args.num_examples]
df['text'] = df['text'].fillna('')

time_segment = True
if "start" not in df.columns or "end" not in df.columns:
    time_segment = False
    df['start'] = None
    df['end'] = None

use_type = True
if "type" not in df.columns:
    use_type = False
    df['type'] = None

labels = df['text'].to_list()
preds = []

print('=============== Start Testing ====================')
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    path = row['path']
    start, end, role = row['start'], row['end'], row['type']
    mel = processor.mel_spectrogram(processor.load_audio(path, start, end, role)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(mel)
    
    preds.append(processor.decode_beam_search(logits[0].cpu().numpy()))
print(f"=============== Finish Testing ====================\n")

print(f"WER Score: {WER_score(preds, labels)}")

if args.saved_name is not None:
    saved_filename = args.saved_name
else:
    test_name = os.path.basename(args.test_path)
    saved_filename = f"result_{test_name}"

result = {
    'path': df['path'].to_list(),
}
if time_segment:
    result['start'] = df['start'].to_list()
    result['end'] = df['end'].to_list()
if use_type:
    result['type'] = df['type'].to_list()
result['text'] = labels
result['predict'] = preds

pd.DataFrame(result).to_csv(f"{args.result_folder}/{saved_filename}", sep="\t", index=False)