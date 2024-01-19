from glob import glob
from pydub import AudioSegment
import os
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--folder_path", type=str)
parser.add_argument("--remove", type=bool, default=False)

args = parser.parse_args()

def convert(path: str):
    assert ".mp3" in path
    try:
        dst = path.replace(".mp3", ".wav")
        sound = AudioSegment.from_mp3(path)
        sound.export(dst, format="wav")
        return True
    except Exception as e:
        print(str(e))
        return False

files = glob(f"{args.folder_path}/*.mp3")

print("Converting All MP3 Files to Wave Format")
for item in tqdm(files, total=len(files)):
    convert(item)

if args.remove:
    print("Deleting All MP3 Files")
    for item in tqdm(files, total=len(files)):
        os.remove(item)

print("Finish")