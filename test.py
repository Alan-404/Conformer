import os
import torch
import torcheval.metrics.functional as F_metric
from preprocessing.processor import ConformerProcessor
from conformer import Conformer
from dataset import ConformerDataset
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--test_path", type=str)
parser.add_argument("--num_examples", type=int, default=None)
parser.add_argument("--checkpoint", type=str)

# Audio Processor Config
parser.add_argument("--num_mels", type=int, default=80)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--fft_size", type=int, default=400)
parser.add_argument("--hop_length", type=int, default=160)
parser.add_argument("--win_length", type=int, default=400)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)

# Text Processor Config
parser.add_argument("--vocab_path", type=str, default="./vocab.json")
parser.add_argument("--pad_token", type=str, default="<pad>")
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--word_delim_token", type=str, default="|")
parser.add_argument("--arpa_path", type=str)
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
    word_delim_token=args.word_delim_token
)

# Model Setup
model = Conformer(
    vocab_size=len(processor.dictionary.get_itos()),
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

testset = ConformerDataset(manifest_path=args.test_path, processor=processor, num_examples=args.num_examples)
preds = []

print('=============== Start Testing ====================')
for idx in tqdm(range(testset.__len__())):
    signal, _ = testset.__getitem__(idx)
    mel_spec = processor.mel_spectrogram(signal).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mel_spec)[0]

    preds.append(processor.decode_beam_search(output.cpu().numpy()))
print(f"=============== Finish Testing ====================\n")

testset.prompts['pred'] = preds

labels = testset.prompts['text'].to_list()
for idx in range(len(preds)):
    labels[idx] = str(labels[idx]).lower()
    preds[idx] = str(preds[idx])

print(f"WER Score: {F_metric.word_error_rate(preds, labels).item()}")

if args.saved_name is not None:
    saved_filename = args.saved_name
else:
    test_name = os.path.basename(args.test_path)
    saved_filename = f"result_{test_name}.csv"

testset.prompts.to_csv(f"./results/{saved_filename}", sep="\t", index=None)