import os
from src.conformer import Conformer
from preprocessing.processor import ConformerProcessor
import torch

#######################################
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--checkpoint", type=str)
parser.add_argument("--saved_path", type=str, default="./conformer.onnx")

# Text Processor Config
parser.add_argument("--vocab_path", type=str, default="./vocab.json")
parser.add_argument("--pad_token", type=str, default="<pad>")
parser.add_argument("--bos_token", type=str, default="<s>")
parser.add_argument("--eos_token", type=str, default="</s>")
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--word_delim_token", type=str, default="|")

# Audio Processor Config
parser.add_argument("--num_mels", type=int, default=80)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--fft_size", type=int, default=480)
parser.add_argument("--hop_length", type=int, default=160)
parser.add_argument("--win_length", type=int, default=480)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)

# Model Config
parser.add_argument("--encoder_n_layers", type=int, default=17)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--kernel_size", type=int, default=31)
parser.add_argument("--decoder_n_layers", type=int, default=1)
parser.add_argument("--decoder_dim", type=int, default=640)
parser.add_argument("--dropout_rate", type=float, default=0.1)

# Dummy Data Example
parser.add_argument("--audio_path", type=str)

# Parse Config
args = parser.parse_args()
########################################

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

assert os.path.exists(args.audio_path)

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
    word_delim_token=args.word_delim_token
)

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

model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
model.eval()

dummy_input = processor.mel_spectrogram(processor.load_audio(args.audio_path)).unsqueeze(0).to(device)

torch.onnx.export(model, 
                  dummy_input, 
                  f=args.saved_path, 
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {
                          2: 'n_ctx'
                      },
                      'output': {
                          2: 'n_ctx'
                      }
                  }
                )
