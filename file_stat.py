import wav_io
import numpy as np
import os
from collections import Counter

PATH = "F://SpeechDatasets//tensorflow_speech_dataset"
EXCLUDED_DIRS = ["_background_noise_"]


def stat_dirs(dir_path, excluded_dirs):
    dirs = [os.path.join(dir_path, p) for p in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, p))
            and p not in excluded_dirs]
    return {os.path.basename(p) : _stat(p) for p in dirs}

def _stat(dir):
    count = Counter()
    for p in os.listdir(dir):
        count[p.split("_")[0]] += 1
    return count

stat = stat_dirs(PATH, EXCLUDED_DIRS)
for key in stat:
    print("WORD", key)
    for person, value in stat[key].most_common():
        print("\tPERSON {}, VALUE {}".format(person, value))
