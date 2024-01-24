import torch
from torch.utils.data import DataLoader
from module import ConformerModule

import lightning as L

from preprocessing.processor import ConformerProcessor
from dataset import ConformerDataset


processor = ConformerProcessor(
    vocab_path='./vocabulary/dictionary.json'
)

model = ConformerModule(
    processor=processor,
    encoder_n_layers=17,
    encoder_dim=512,
    heads=8,
    kernel_size=31,
    decoder_n_layers=1,
    decoder_dim=640,
    dropout_rate=0.1
)

trainer = L.Trainer(num_nodes=torch.cuda.device_count(), max_epochs=5, default_root_dir='./checkpoints/root', precision='bf16', logger=[])

def get_batch(batch) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, transcripts = zip(*batch)
        mels, mel_lengths = processor(signals, return_length=True)

        tokens, token_lengths = processor.tokenize(transcripts)

        return mels, tokens, mel_lengths, token_lengths

dataset = ConformerDataset('./datasets/vivos-train.csv', processor=processor, num_examples=150)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=get_batch)

trainer.fit(model, dataloader)