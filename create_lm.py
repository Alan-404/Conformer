import os
from processing.processor import ConformerProcessor
import io
import subprocess
from tqdm import tqdm

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
        texts.append(str(item).upper().strip())

    unique_words = []
    lexicon = []
    print("Getting Unique Words")
    for text in tqdm(texts):
        words = text.split(" ")
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
                graphemes = " ".join(processor.word2graphemes(word))
                lexicon.append(f"{word} {graphemes} {processor.delim_token}")

    num_words = len(unique_words) - 1
    with open(f"{saved_folder}/lexicon.txt", 'w', encoding='utf8') as file:
        for index, line in enumerate(lexicon):
            file.write(line)
            if index != num_words:
                file.write("\n")
    
    
