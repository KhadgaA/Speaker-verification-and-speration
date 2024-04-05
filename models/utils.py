import torch
from torch.nn.utils.rnn import pad_sequence
from s3prl.upstream.interfaces import UpstreamBase
from omegaconf import OmegaConf

import torch.nn.functional as F

def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    cfg = state["cfg"]

    task = cfg.task
    model = cfg.model

    return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        model, cfg, task = load_model(ckpt)
        self.model = model
        self.task = task

    def forward(self, wavs):
        if self.task.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        return {
            "default": features,
        }
