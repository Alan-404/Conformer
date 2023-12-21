# Conformer Model - Speech to Text
## Model Architecture
<img src="./assets/model.png"/>
<a href="https://arxiv.org/abs/2005.08100">Link Paper</a>

## Folder Structure
```
assets
.gitignore
conformer.py        # Model Source
dataset.py          # Dataset Loader
hotwords.json       # Hotwords of BEAM Search - Used in Post Processing
processor.py        # Data Processing Handler
README.md
requirements.txt
train.py            # Train Model
vocab.json          # Characters Dictionary
```

## Setup Environment
```
git clone https://git.cads.live/trind18/conformer_model.git
cd conformer_model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train
```
CUDA_VISIBLE_DEVICES={index} python3 train.py --device cuda --batch_size {train_batch_size} --val_batch_size {val_batch_size} --num_epochs {number_of_epochs}
```