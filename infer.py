import os
import torch
from model.conformer import Conformer
from preprocessing.processor import ConformerProcessor

########################################
from argparse import ArgumentParser

parser = ArgumentParser()

# Audio Processor Config
parser.add_argument("--num_mels", type=int, default=80)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--fft_size", type=int, default=480)
parser.add_argument("--hop_length", type=int, default=160)
parser.add_argument("--win_length", type=int, default=480)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)

# Text Processor Config
parser.add_argument("--vocab_path", type=str, default="./vocab.json")
parser.add_argument("--pad_token", type=str, default="<pad>")
parser.add_argument("--bos_token", type=str, default="<s>")
parser.add_argument("--eos_token", type=str, default="</s>")
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--word_delim_token", type=str, default="|")

parser.add_argument("--lm_path", type=str, default=None)

# Model Config
parser.add_argument("--n", type=int, default=6)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--kernel_size", type=int, default=31)
parser.add_argument("--eps", type=float, default=0.2)
parser.add_argument("--dropout_rate", type=float, default=0.1)

# Device Config
parser.add_argument("--device", type=str, default='cpu')

# Checkpoint Config
parser.add_argument("--checkpoint", type=str, default="./checkpoints")
parser.add_argument("--checkpoint_step", type=str, default=None)

# Example
parser.add_argument("--audio_path", type=str)

########################################

args = parser.parse_args()

# Setup Checkpoint
checkpoint = None
def find_latest_checkpoint(checkpoints: list):
    latest = 0
    for checkpoint in checkpoints:
        index = int(checkpoint.replace("checkpoint_", "").replace(".pt", ""))
        if latest < index:
            latest = index
    return latest
if os.path.exists(args.checkpoint) and os.listdir(args.checkpoint) != 0:
    if args.checkpoint_step is None:
        args.checkpoint_step = find_latest_checkpoint(os.listdir(args.checkpoint))
    checkpoint = f"{args.checkpoint}/checkpoint_{args.checkpoint_step}.pt"

# Device Setup
device = 'cpu'
if args.device == 'cuda' or args.device == 'gpu':
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
    bos_token=args.bos_token,
    eos_token=args.eos_token,
    word_delim_token=args.word_delim_token,
    lm_path=args.lm_path
)

# Model Setup
model = Conformer(
    vocab_size=len(processor.dictionary.get_itos()),
    n_mel_channels=processor.num_mels,
    n=args.n,
    d_model=args.d_model,
    heads=args.heads,
    kernel_size=args.kernel_size,
    eps=args.eps,
    dropout_rate=args.dropout_rate
).to(device)

model.load_state_dict(torch.load(checkpoint)['model'])
model.eval()

mel_input = processor.mel_spectrogram(processor.load_audio(args.audio_path)).unsqueeze(0)

with torch.no_grad():
    output = model(mel_input)

transcript = processor.decode_beam_search(output[0].cpu().numpy())

print(f"Transcription: {transcript}")