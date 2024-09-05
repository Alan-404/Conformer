import os
from processing.processor import ConformerProcessor
import io
import subprocess
from tqdm import tqdm
import re

import fire

def main(
        saved_folder: str,
        text_path: str,
        tokenizer_path: str = "./tokenizer/vi.json",
        n_grams: int = 5
    ):
    assert os.path.exists(text_path)

    if os.path.exists(saved_folder) == False:
        os.makedirs(saved_folder)
    
    processor = ConformerProcessor(tokenizer_path=tokenizer_path)

    data = io.open(text_path).read().strip().split("\n")

    texts = []
    print("Formatting Text")
    for item in tqdm(data):
        text = str(item).upper()
        text = processor.clean_text(text)
        texts.append(text)
    
    last_idx = len(texts) - 1
    with open(f"{saved_folder}/lm_text.txt", 'w', encoding='utf8') as file:
        for index, line in enumerate(texts):
            file.write(line)
            if index != last_idx:
                file.write("\n")

    unique_words = []
    lexicon = []
    print("Getting Unique Words and Creating Lexicon")
    for text in tqdm(texts):
        words = text.split(" ")
        for word in words:
            if word not in unique_words:
                graphemes = " ".join(processor.word2graphemes(word))
                if processor.unk_token in graphemes:
                    continue
                unique_words.append(word)
                lexicon.append(f"{word} {graphemes} {processor.delim_token}")

    last_idx = len(unique_words) - 1
    with open(f"{saved_folder}/lexicon.txt", 'w', encoding='utf8') as file:
        for index, line in enumerate(lexicon):
            file.write(line)
            if index != last_idx:
                file.write("\n")
    
    print("Creating KenLM")
    subprocess.run(["kenlm/build/bin/lmplz", "-o", f"{n_grams}", "--text", f"{saved_folder}/lm_text.txt", "--arpa", f"{saved_folder}/lm.arpa", "-S", "2G"], capture_output=True, text=True)
    print("Finish Creating KenLM")

if __name__ == '__main__':
    fire.Fire(main)