import torch

# import fire
# from torchaudio import load
# import torchaudio
# from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

import torch.nn.functional as F

from tqdm import tqdm
from compute_eer import eer
from compute_eer_ecapa import eval_network

# from compute_ecapa_multiprocess import eval_network
import argparse
import os

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

if __name__ == "__main__":
    # fire.Fire(verification)
    models = {
        "wavlm_base_plus": "./model_checkpoints/wavlm_base_plus_nofinetune.pth",
        "wavlm_large": "./model_checkpoints/wavlm_large_nofinetune.pth",
        "hubert_large": "./model_checkpoints/HuBERT_large_SV_fixed.th",
        "wav2vec2_xlsr": "./model_checkpoints/wav2vec2_xlsr_SV_fixed.th",
    }
    # voxceleb_hard = torchaudio.datasets.VoxCeleb1Verification(
    #     r"/scratch/data/m23csa003/voxceleb",
    #     download=True,
    #     meta_url="/scratch/data/m23csa003/voxceleb/list_test_hard2.txt"

    # )
    print("data_correctly parsed")

    # test_loader = torch.utils.data.DataLoader(
    #     voxceleb_hard, batch_size=1, shuffle=True
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="wavlm_base_plus", help="Model name"
    )
    parser.add_argument(
        "--n_samples", type=int, default=-1, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()
    model_names = args.model.split(" ")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print(device)
    else:
        device = torch.device("cpu")
    # print(chkpts)
    # for model_name, chkpt in models.items():
    # evaluation(
    #     model_names=model_names,
    #     checkpoints=models,
    #     loader=test_loader,
    #     n_samples=args.n_samples,
    #     device = device,
    # )
    test_file_dir = (
        "/mnt/d/programming/datasets/VoxCeleb/list_test_hard2.txt"
        if os.name == "posix"
        else "D:/programming/datasets/VoxCeleb/list_test_hard2.txt"
    )

    test_wavs_dir = (
        "/mnt/d/programming/datasets/VoxCeleb/wav/"
        if os.name == "posix"
        else "D:/programming/datasets/VoxCeleb/wav/"
    )
    EER, minDCF = eval_network(
        init_model(model_names[0], models[model_names[0]]).to(device),
        test_file_dir,
        test_wavs_dir,
        device,
        n_samples=args.n_samples,
    )
    print("EER Full Utterences")
    print(f"model = {model_names[0]},EER = {EER}, minDCF = {minDCF}")
