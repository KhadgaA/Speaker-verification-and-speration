import torch

# import fire
from torchaudio import load
import torchaudio
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

import torch.nn.functional as F

from tqdm import tqdm
from compute_eer import eer
from compute_eer_ecapa import eval_network
import argparse

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


def evaluation(model_names, loader, use_gpu=True, checkpoints=None, n_samples=-1):
    models = torch.nn.ModuleList()
    for model_name in model_names:
        assert model_name in MODEL_LIST, "The model_name should be in {}".format(
            MODEL_LIST
        )
        print(model_name)
        model = init_model(model_name, checkpoints[model_name])
        model.eval()
        models.append(model)

    if use_gpu:
        models = models.cuda()
    test_scores = {model_name: [] for model_name in model_names}
    test_labels = {model_name: [] for model_name in model_names}
    i = 0
    for wav1, wav2, sr, label, *_ in tqdm(loader):

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        if use_gpu:
            wav1 = wav1.cuda()
            wav2 = wav2.cuda()

        for model_name, model in zip(model_names, models):
            with torch.no_grad():
                emb1 = model(wav1)
                emb2 = model(wav2)

            sim = F.cosine_similarity(emb1, emb2)
            test_scores[model_name].append(sim)
            test_labels[model_name].append(label)
        i += 1

        # print(
        #     "The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(
        #         sim[0].item()
        #     )
        # )
        if i == n_samples:
            break
    for model_name in model_names:
        equal_error_rate, threshold = eer(
            test_labels[model_name], test_scores[model_name]
        )
        print(
            f"model = {model_name}, equal error rate = {equal_error_rate}, threshold = {threshold}"
        )


if __name__ == "__main__":
    # fire.Fire(verification)
    models = {
        "wavlm_base_plus": "./model_checkpoints/wavlm_base_plus_nofinetune.pth",
        "wavlm_large": "./model_checkpoints/wavlm_large_nofinetune.pth",
        "hubert_large": "./model_checkpoints/HuBERT_large_SV_fixed.th",
        "wav2vec2_xlsr": "./model_checkpoints/wav2vec2_xlsr_SV_fixed.th",
    }
    # voxceleb_hard = torchaudio.datasets.VoxCeleb1Verification(
    #     r"/mnt/d/programming/datasets/VoxCeleb",
    #     download=True,
    #     meta_url="/mnt/d/programming/datasets/VoxCeleb/list_test_hard2.txt",
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     voxceleb_hard, batch_size=1, shuffle=False
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="wavlm_base_plus", help="Model name"
    )
    parser.add_argument(
        "--n_samples", type=int, default=-1, help="Number of samples to evaluate"
    )
    args = parser.parse_args()
    model_names = args.model.split(" ")

    # print(chkpts)
    # for model_name, chkpt in models.items():
    # evaluation(
    #     model_names=model_names,
    #     checkpoints=models,
    #     loader=test_loader,
    #     n_samples=args.n_samples,
    # )
    EER, minDCF = eval_network(
        init_model(model_names[0], models[model_names[0]]).cuda(),
        "/mnt/d/programming/datasets/VoxCeleb/list_test_hard2.txt",
        "/mnt/d/programming/datasets/VoxCeleb/wav/",
    )

    print(f"model = {model_names[0]},EER = {EER}, minDCF = {minDCF}")
