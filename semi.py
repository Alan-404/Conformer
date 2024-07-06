import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model.conformer import Conformer
from processing.processor import ConformerProcessor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from evaluation import ConformerCriterion
from model.transformation.mel import MelTransform

from pyctcdecode import build_ctcdecoder

import lightning as L

from typing import Optional, Tuple

class TrainingModule(L.LightningModule):
    def __init__(self, n_blocks: int, d_model: int, heads: int, kernel_size: int, n_layers: int, hidden_dim: int, dropout_rate: float, processor: ConformerProcessor, mel_transform: MelTransform, teacher_checkpoint: str, teacher_lm: Optional[str] = None, project_name: Optional[str] = None, run_id: Optional[str] = None, run_name: Optional[str] = None) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['processor', 'teacher_checkpoint', 'mel_transform', "project_name", "run_name", "run_id", 'teacher_lm'])
        self.automatic_optimization = False

        self.model = Conformer(
            vocab_size=len(processor.dictionary),
            n_mel_channels=processor.num_mels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        self.processor = processor
        self.mel_transform = mel_transform

        self.teacher = Wav2Vec2ForCTC.from_pretrained(teacher_checkpoint)
        teacher_processor = Wav2Vec2Processor.from_pretrained(teacher_checkpoint)
        
        self.teacher_processor = teacher_processor
        if teacher_lm is not None:
            self.teacher_processor = Wav2Vec2ProcessorWithLM(
                feature_extractor=teacher_processor.feature_extractor,
                tokenizer=teacher_processor.tokenizer,
                decoder=build_ctcdecoder(
                    labels=list(teacher_processor.tokenizer.get_vocab().keys()),
                    kenlm_model_path=teacher_lm
                )
            )

        self.criterion = ConformerCriterion(blank_id=processor.pad_idx)

        self.prev_loss = None
    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        # inputs is audio signals
        l_data = batch[0]

        l_tokens, l_token_lengths = batch[1], batch[2]

        u_data = batch[4]

        student_optim, teacher_optim = self.optimizers()

        # Unsupervised Data
        teacher_inputs = self.teacher_processor(l_data, return_tensors='pt', padding='longest', return_attention_mask=True)
        teacher_outputs = self.teacher(teacher_inputs.input_values, teacher_inputs.attention_mask).logits

        transcriptions = self.teacher_processor.batch_decode(torch.argmax(teacher_outputs))

        graphemes = []
        for item in transcriptions:
            graphemes.append(self.processor.sentence2graphemes(str(item).lower()))
        
        hard_pseudo_tokens, hard_pseudo_lengths = self.processor.tokenize(graphemes)

        with self.teacher_processor.as_target_processor():
            teacher_input_items = self.teacher_processor(transcriptions, padding=True, return_attention_mask=True)

        s_u_data, s_u_data_lengths = self.processor(u_data)
        unsupervised_outputs, unsupervised_lengths = self.model(s_u_data, s_u_data_lengths)
        unsupervised_loss = self.criterion.ctc_loss(unsupervised_outputs, hard_pseudo_tokens, input_lengths=unsupervised_lengths, target_lengths=hard_pseudo_lengths)

        self.manual_backward(unsupervised_loss)
        student_optim.step()

        # Supervised Data
        s_s_data, s_s_data_lengths = self.processor(l_data)
        labeled_outputs, labeled_lengths = self.model(s_s_data, s_s_data_lengths)
        supervised_loss = self.criterion.ctc_loss(labeled_outputs, l_tokens, labeled_lengths, l_token_lengths)

        self.manual_backward(supervised_loss)
        student_optim.step()

        # Teacher
        dot_product = supervised_loss - unsupervised_loss

        teacher_unsupervised_lengths = self.teacher._get_feat_extract_output_lengths(torch.count_nonzero(teacher_inputs.attention_mask, dim=-1))

        teacher_targets = []
        teacher_target_lengths = []

        for index, item in enumerate(teacher_input_items.attention_mask):
            teacher_targets.append(teacher_input_items.input_ids[index])
            teacher_target_lengths.append(torch.count_nonzero(item))

        teacher_unsupervised_loss = dot_product * self.criterion.ctc_loss(teacher_outputs.logits, torch.stack(teacher_targets), teacher_unsupervised_lengths, torch.tensor(teacher_target_lengths))
        
        teacher_supervised_inputs = self.teacher_processor(l_data, return_tensors='pt', padding='longest', return_attention_mask=True)
        teacher_supervised_outptus = self.teacher(
            teacher_supervised_inputs.input_values,
            teacher_supervised_inputs.attention_mask
        )

        teacher_supervised_lengths = self.teacher._get_feat_extract_output_lengths(torch.count_nonzero(teacher_supervised_inputs.attention_mask, dim=-1))
        supervised_loss = self.criterion.ctc_loss(teacher_supervised_outptus, l_tokens, teacher_supervised_lengths, l_token_lengths)

        self.manual_backward(teacher_unsupervised_loss + supervised_loss)
        teacher_optim.step()

        
    def configure_optimizers(self):
        student_optimizer = optim.AdamW(params=self.model.parameters())
        teacher_optimizer = optim.SGD(params=self.teacher.parameters())

        student_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=student_optimizer, T_max=1000)
        teacher_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=teacher_optimizer, T_max=1000)

        return [student_optimizer], [{'scheduler': student_scheduler, 'interval': "epoch"}, {'scheduler': teacher_scheduler, 'interval': "epoch"}]
        

        
