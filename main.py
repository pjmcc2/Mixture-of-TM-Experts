import torch
import torchaudio
import numpy as np
import soundfile as sf

path = "D:/pjmcc/Documents/OSU_datasets/VocalSet/Example/f1_arpeggios_belt_c_u.wav"
example = torchaudio.load(path)
print(example[0].shape, example[1])

print(torchaudio.info(path,backend="soundfile"))