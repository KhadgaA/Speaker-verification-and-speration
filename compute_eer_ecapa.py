import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *


def eval_network(model, eval_list, eval_path,device):
    model.eval()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()

    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])

    setfiles = list(set(files))
    setfiles.sort()

    for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
        audio, _ = soundfile.read(os.path.join(eval_path, file))
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(device)

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio: 
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), "wrap")
        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf) : int(asf) + max_audio])
        feats = numpy.stack(feats, axis=0).astype(float)
        data_2 = torch.FloatTensor(feats).to(device)
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = model(data_1)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1).detach().cpu()
            embedding_2 = model(data_2)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1).detach().cpu()
        embeddings[file] = [embedding_1, embedding_2]
    scores, labels = [], []
    for line in lines:
        embedding_11, embedding_12 = embeddings[line.split()[1]]
        embedding_21, embedding_22 = embeddings[line.split()[2]]
        # Compute the scores
        score_1 = torch.mean(
            torch.matmul(embedding_11, embedding_21.T)
        )  # higher is positive
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        score = (score_1 + score_2) / 2
        score = score.detach().numpy()
        scores.append(score)
        labels.append(int(line.split()[0]))


    # Coumpute EER and minDCF
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF
