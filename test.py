import os
import torch
import fire
from preprocessing.processor import ConformerProcessor
from src.conformer import Conformer
from tqdm import tqdm
import pandas as pd
from src.metric import WER_score

def test(result_folder: str,
         test_path: str,
         vocab_path: str,
         arpa_path: str,
         checkpoint: str,
         num_mels: int = 80,
         sampling_rate: int = 16000,
         fft_size: int = 400,
         hop_length: int = 160,
         win_length: int = 400,
         fmin: float = 0.0,
         fmax: float = 8000.0,
         pad_token: str = "<pad>",
         unk_token: str = "<unk>",
         word_delim_token: str = "|",
         encoder_n_layers: int = 17,
         encoder_dim: int = 512,
         heads: int = 8,
         kernel_size: int = 31,
         decoder_n_layers: int = 1,
         decoder_dim: int = 640,
         dropout_rate: float = 0.0,
         num_examples: int = None,
         saved_name: str = None):
    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Processor Setup
    processor = ConformerProcessor(
        vocab_path=vocab_path,
        num_mels=num_mels,
        sampling_rate=sampling_rate,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        pad_token=pad_token,
        unk_token=unk_token,
        word_delim_token=word_delim_token,
        lm_path=arpa_path
    )

    # Model Setup
    model = Conformer(
        vocab_size=len(processor.dictionary),
        n_mel_channels=processor.num_mels,
        encoder_n_layers=encoder_n_layers,
        encoder_dim=encoder_dim,
        heads=heads,
        kernel_size=kernel_size,
        decoder_n_layers=decoder_n_layers,
        decoder_dim=decoder_dim,
        dropout_rate=dropout_rate
    ).to(device)

    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    df = pd.read_csv(test_path, sep="\t")
    if num_examples is not None:
        df = df[:num_examples]
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
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = row['path']
        start, end, role = row['start'], row['end'], row['type']
        mel = processor.mel_spectrogram(processor.load_audio(path, start, end, role)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(mel)
        
        preds.append(processor.decode_beam_search(logits[0].cpu().numpy()))
    print(f"=============== Finish Testing ====================\n")

    print(f"WER Score: {WER_score(preds, labels)}")

    if saved_name is not None:
        saved_filename = saved_name
    else:
        test_name = os.path.basename(test_path)
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

    pd.DataFrame(result).to_csv(f"{result_folder}/{saved_filename}", sep="\t", index=False)

if __name__ == '__main__':
    fire.Fire(test)