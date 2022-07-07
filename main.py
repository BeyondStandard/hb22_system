from pathlib import Path
from torchaudio import transforms
from IPython.display import Audio

import pandas as pd
import math, random
import torch
import torchaudio


df = pd.DataFrame()
df["filename"] = pd.Series([], dtype=object)
df["class"] = pd.Series([], dtype=int)

dataset_path = Path.cwd() / 'Datasets'
df = pd.concat([
    pd.DataFrame(((i, int(index)) for i in dataset_type_folder.iterdir()), columns=df.columns)
    for index, dataset_type_folder in enumerate(dataset_path.iterdir())
], ignore_index=True)



class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)