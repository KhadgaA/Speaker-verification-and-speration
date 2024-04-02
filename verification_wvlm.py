import torch

# import fire
from torchaudio import load
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

import torch.nn.functional as F

MODEL_LIST = [
    "ecapa_tdnn",
    "hubert_large",
    "wav2vec2_xlsr",
    "unispeech_sat",
    "wavlm_base_plus",
    "wavlm_large",
]


def init_model(model_name, checkpoint=None):
    if model_name == "unispeech_sat":
        config_path = "config/unispeech_sat.th"
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="unispeech_sat", config_path=config_path
        )
    elif model_name == "wavlm_base_plus":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=768, feat_type="wavlm_base_plus", config_path=config_path
        )
    elif model_name == "wavlm_large":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=config_path
        )
    elif model_name == "hubert_large":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="hubert_large_ll60k", config_path=config_path
        )
    elif model_name == "wav2vec2_xlsr":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="xlsr_53", config_path=config_path
        )
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type="fbank")

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"], strict=False)
    return model


def verification(model_name, wav1, wav2, use_gpu=True, checkpoint=None):

    assert model_name in MODEL_LIST, "The model_name should be in {}".format(MODEL_LIST)
    model = init_model(model_name, checkpoint)
    wav1, sr1 = load(wav1)
    wav2, sr2 = load(wav2)

    resample1 = Resample(orig_freq=int(sr1), new_freq=16000)
    resample2 = Resample(orig_freq=int(sr2), new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        model = model.cuda()
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    print(model_name)
    print(
        "The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(
            sim[0].item()
        )
    )


if __name__ == "__main__":
    # fire.Fire(verification)
    models = {
        "wavlm_base_plus": "./model_checkpoints/wavlm_base_plus_nofinetune.pth",
        "wavlm_large": "./model_checkpoints/wavlm_large_nofinetune.pth",
        "hubert_large": "./model_checkpoints/HuBERT_large_SV_fixed.th",
        "wav2vec2_xlsr": "./model_checkpoints/wav2vec2_xlsr_SV_fixed.th",
    }
    for model_name, chkpt in models.items():
        verification(
            model_name=model_name,
            wav1="/mnt/c/Users/KHADGA JYOTH ALLI/Desktop/programming/Class Work/IITJ/Speech Understanding/Assignment 2/trial_wavs/hn8GyCJIfLM_0000012.wav",
            wav2="/mnt/c/Users/KHADGA JYOTH ALLI/Desktop/programming/Class Work/IITJ/Speech Understanding/Assignment 2/trial_wavs/xTOk1Jz-F_g_0000015.wav",
            checkpoint=chkpt,
        )
