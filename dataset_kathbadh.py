import os
import re
import sys
import time
import random
import pickle

import tqdm
import torch
import torchaudio
import numpy as np
from torch import nn
from pathlib import Path
from torchaudio import load
from librosa.util import find_files
from joblib.parallel import Parallel, delayed
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed import is_initialized
import os
if os.name == 'posix':
    from torchaudio.sox_effects import apply_effects_file
else:
    
    def apply_effects_file(*args):pass
# from pydub import AudioSegment

EFFECTS = [
    # ["channels", "1"],
    # ["rate", "16000"],
    ["gain", "-3.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

# def apply_effects_to_wav_file(wav_path, effects):
#  # Load audio file
#     audio = AudioSegment.from_file(wav_path)

#     # Apply effects
#     for effect in effects:
#         if effect[0] == "gain":
#             gain_db = float(effect[1])
#             audio = audio + gain_db
#         elif effect[0] == "silence":
#             threshold_start = float(effect[1]) * 1000  # convert to milliseconds
#             threshold_duration = float(effect[2]) * 1000  # convert to milliseconds
#             pad_start = float(effect[3]) * 1000  # convert to milliseconds
#             pad_duration = float(effect[4]) * 1000  # convert to milliseconds
#             audio = audio.strip_silence(
#                 threshold=threshold_start,
#                 silence_dur=threshold_duration,
#                 padding=(pad_start, pad_duration)  # padding to avoid clipping
#             )
#     # Export processed audio to a temporary file
#     output_path = wav_path.replace(".wav", "_processed.wav")
#     audio.export(output_path, format="wav")

#     # Load processed waveform
#     processed_waveform, _ = torchaudio.load(output_path)

#     # Remove the temporary file
#     os.remove(output_path)

#     return processed_waveform

# Voxceleb 2 Speaker verification

def get_train_dataloader(dataset,batch_size,num_workers):
    sampler = DistributedSampler(dataset) if is_initialized() else None
    return DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )
class SpeakerVerifi_train(Dataset):
    def __init__(
        self, vad_config, key_list, file_path, meta_data, max_timestep=None, n_jobs=12
    ):
        self.roots = file_path
        self.root_key = key_list
        self.max_timestep = max_timestep
        self.vad_c = vad_config
        self.dataset = []
        self.all_speakers = []

        for index in range(len(self.root_key)):
            cache_path = (
                Path(os.path.dirname(__file__))
                / ".wav_lengths"
                / f"{self.root_key[index]}_length.pt"
            )
            cache_path.parent.mkdir(exist_ok=True)
            root = Path(self.roots[index])

            if not cache_path.is_file():

                def trimmed_length(path):
                    wav_sample, _ = apply_effects_file(path, EFFECTS)
                    # wav_sample,_ = apply_effects_to_wav_file(path,EFFECTS)
                    wav_sample = wav_sample.squeeze(0)
                    length = wav_sample.shape[0]
                    return length

                wav_paths = find_files(root)
                wav_lengths = Parallel(n_jobs=n_jobs)(
                    delayed(trimmed_length)(path)
                    for path in tqdm.tqdm(wav_paths, desc="Preprocessing")
                )
                wav_tags = wav_paths  # [Path(path).parts[-3:] for path in wav_paths]
                torch.save([wav_tags, wav_lengths], str(cache_path))
            else:
                wav_tags, wav_lengths = torch.load(str(cache_path))
                wav_paths = wav_tags  # [root.joinpath(*tag) for tag in wav_tags]

            speaker_dirs = [f.stem for f in root.iterdir() if f.is_dir()]
            self.all_speakers.extend(speaker_dirs)
            for path, length in zip(wav_paths, wav_lengths):
                if length > self.vad_c["min_sec"]:
                    self.dataset.append(path)

        self.all_speakers.sort()
        self.speaker_num = len(self.all_speakers)
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset[idx]
        wav, _ = apply_effects_file(str(path), EFFECTS)
        # wav_sample,_ = apply_effects_to_wav_file(str(path),EFFECTS)
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep != None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start : start + self.max_timestep]

        tag = Path(path).parts[-2]
        utterance_id = path  # "-".join(tags).replace(".wav", "")
        # print(tags[0])
        label = self.all_speakers.index(tag)
        return wav.numpy(), utterance_id, label

    def collate_fn(self, samples):
        return zip(*samples)




class SpeakerVerifi_test(Dataset):
    def __init__(self, vad_config, file_path, meta_data):
        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.vad_c = vad_config
        self.dataset = self.necessary_dict["pair_table"]

    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1 = os.path.join(self.root, list_pair[1])
            pair_2 = os.path.join(self.root, list_pair[2])
            one_pair = [list_pair[0], pair_1, pair_2]
            pair_table.append(one_pair)
        return {"spk_paths": None, "total_spk_num": None, "pair_table": pair_table}

    def __len__(self):
        return len(self.necessary_dict["pair_table"])

    def __getitem__(self, idx):
        y_label, x1_path, x2_path = self.dataset[idx]

        def path2name(path):
            return path  # Path("-".join((Path(path).parts)[-3:])).stem

        x1_name = path2name(x1_path)
        x2_name = path2name(x2_path)

        wav1, _ = apply_effects_file(x1_path, EFFECTS)
        wav2, _ = apply_effects_file(x2_path, EFFECTS)
        # wav1, _ = apply_effects_to_wav_file(x1_path, EFFECTS)
        # wav2, _ = apply_effects_to_wav_file(x2_path, EFFECTS)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        return wav1.numpy(), wav2.numpy(), x1_name, x2_name, int(y_label[0])

    def collate_fn(self, data_sample):
        wavs1, wavs2, x1_names, x2_names, ylabels = zip(*data_sample)
        all_wavs = wavs1 + wavs2
        all_names = x1_names + x2_names
        return all_wavs, all_names, ylabels
    
    def _get_eval_dataloader(self, batch_size, num_workers):
        return DataLoader(
            self.dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            collate_fn=self.collate_fn
        )
    