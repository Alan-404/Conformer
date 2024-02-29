import os
import glob
import json
from tqdm import tqdm

import fire

def convert(path: str, source_dir: str = "."):
    folders = os.listdir(path)

    paths = dict()
    for folder in tqdm(folders):
        items = os.listdir(f'{path}/{folder}')
        for item in items:
            item = item[0]

            files = glob.glob(f"{path}/{folder}/{item}/*.json")

            for json_item in files:
                data = json.load(open(json_item, 'r'))
                filename = os.path.basename(json_item)
                paths[filename.replace(".json", "")] = {
                    "audio": json_item.replace(".json", ".flac"),
                    "segments": data['voice_activity']
                }

    root_folder = path.split("/")[-1]

    with open(f"{source_dir}/{root_folder}.json", 'w') as file:
        json.dump(file)

if __name__ == '__main__':
    fire.Fire(convert)
    
