import torch
import torch.nn as nn
import math
from dataset_kathbadh import *

class AAMSoftmaxLoss(nn.Module):
    def __init__(self, hidden_dim, speaker_num, s=15, m=0.3, easy_margin=False, **kwargs):
        super(AAMSoftmaxLoss, self).__init__()
        import math

        self.test_normalize = True
        
        self.m = m
        self.s = s
        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x_BxH, labels_B):

        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        # cos(theta)
        cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels_B.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, labels_B)
        return loss



import torch
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

import torch.nn.functional as F

from tqdm import tqdm
import argparse
import os

import socket

pc = socket.gethostname()

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

def train(model, train_loader,device, args):
    epochs = args['epochs']
    loss_func = args['loss_func']
    optimizer = args['optimizer']
    history = {"train_loss": [], "EER": [], "minDCF": []}
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        i =0
        for data in tqdm(train_loader,dynamic_ncols=True):
            # print(list(data))
            wavs, utter_idx,labels = list(data)
            wavs = [torch.FloatTensor(wav).to(device) for wav in wavs]
            labels = torch.LongTensor(labels).to(device)

            embedding = model(wavs)
            # embeddings.append(embedding)
            loss =  loss_func(embedding,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            i+=1
            if i%100 == args["n_samples"]:
                break
        epoch_loss /= len(train_loader)

        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        history["train_loss"].append(epoch_loss)
        EER, minDCF = evaluate(model, device)
        history["EER"].append(EER)
        history["minDCF"].append(minDCF)
        print(f"EER: {EER}, minDCF: {minDCF}")
    torch.save(model.state_dict(), f'{args["model"]}_kathbadh_finetune.pth') 
    return history
def evaluate(model, device):
    EER, minDCF = eval_network(
            model,
            test_file_dir,
            test_wavs_dir,
            device,
            n_samples=args.n_samples,
        )
    
    return EER, minDCF

def save_plots(history):
    import matplotlib.pyplot as plt

    plt.plot(history["train_loss"])
    plt.title("Training Loss")
    plt.savefig("train_loss.png")
    plt.close()

    plt.plot(history["EER"])
    plt.title("EER")
    plt.savefig("EER.png")
    plt.close()

    plt.plot(history["minDCF"])
    plt.title("minDCF")
    plt.savefig("minDCF.png")
    plt.close()
    
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="kathbadh",
        help="Dataset name",
        choices=["voxceleb", "kathbadh"],
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    args = parser.parse_args()
    model_names = args.model.split(" ")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print(device)
    else:
        device = torch.device("cpu")
    if args.dataset == "voxceleb":
        from compute_eer_vox import eval_network

        if  "iitj.ac.in" in pc:
            test_file_dir = "/scratch/data/m23csa003/voxceleb/list_test_hard2.txt"
            test_wavs_dir = "/scratch/data/m23csa003/voxceleb/wav/"
        elif pc == "Khadga-Laptop":
            if os.name == "posix":
                test_file_dir = (
                    "/mnt/d/programming/datasets/VoxCeleb/list_test_hard2.txt"
                )
                test_wavs_dir = "/mnt/d/programming/datasets/VoxCeleb/wav/"
            else:
                test_file_dir = "D:/programming/datasets/VoxCeleb/list_test_hard2.txt"
                test_wavs_dir = "D:/programming/datasets/VoxCeleb/wav/"


    elif args.dataset == "kathbadh":
        from compute_eer_kathbadh import eval_network
            
        test_file_dir = (
                "/scratch/data/m23csa003/kathbadh/meta_data/telugu/test_known_data.txt"
            )
        test_wavs_dir = "/scratch/data/m23csa003/kathbadh/kb_data_clean_wav/telugu/test_known/audio/"
        train_file_dir = "/scratch/data/m23csa003/kathbadh/valid_audio/kb_data_clean_wav/telugu/valid/audio"
        if pc == "Khadga-Laptop":
            if os.name == "posix":
                test_file_dir = "/mnt/d/programming/datasets/kathbadh/meta_data/telugu/test_known_data.txt"
                test_wavs_dir = "/mnt/d/programming/datasets/kathbadh/kb_data_clean_wav/telugu/test_known/audio/"
                train_file_dir = "/mnt/d/programming/datasets/kathbadh/valid_audio/kb_data_clean_wav/telugu/valid/audio"
            else:
                test_file_dir = "D:/programming/datasets/kathbadh/meta_data/telugu/test_known_data.txt"
                test_wavs_dir = "D:/programming/datasets/kathbadh/kb_data_clean_wav/telugu/test_known/audio/"
                train_file_dir = "D:/programming/datasets/kathbadh/valid_audio/kb_data_clean_wav/telugu/valid/audio"
    # D:/programming/datasets/kathbadh/valid_audio/kb_data_clean_wav/telugu/valid/audio
    train_config = {
            "vad_config": {'min_sec': 32000},
            "file_path": [train_file_dir],
            "key_list": ["telugu"],
            "meta_data": "",
            "max_timestep": 128000,
        }
    print(train_config)
   
    train_dataset = SpeakerVerifi_train(**train_config)
    train_loader = get_train_dataloader(train_dataset,args.batch_size, 1)
    model = init_model(model_names[0], models[model_names[0]]).to(device)

    train_args = {
        'epochs': args.epochs,
        'loss_func': AAMSoftmaxLoss(256,train_dataset.speaker_num).to(device),
        'optimizer': torch.optim.Adam(model.parameters(), lr=5e-6),
        'n_samples': args.n_samples,
        'model':args.model
    }
    history = train(model, train_loader, device, train_args)
    save_plots(history)
