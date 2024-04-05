import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *

score_fn = nn.CosineSimilarity()


import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def EER(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    s = interp1d(fpr, tpr)
    a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
    eer = brentq(a, 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh


def eval_network(model, eval_list, eval_path, device, n_samples=-1):
    model.eval()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()

    i = 0
    for line in lines:
        files.append("/".join(line.split()[1].split("/")[-2:]))
        files.append("/".join(line.split()[2].split("/")[-2:]))
        i += 1
        if i == n_samples:
            break
    setfiles = list(set(files))
    setfiles.sort()

    for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
        audio, _ = soundfile.read(os.path.join(eval_path, file))
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(device)

        with torch.no_grad():
            embedding_1 = model(data_1)            
            embedding_1 = F.normalize(embedding_1, p=2, dim=1).detach().cpu()

        embeddings[file] = embedding_1
    scores, labels = [], []
    i = 0
    for line in lines:

        embedding_11 = embeddings["/".join(line.split()[1].split("/")[-2:])]
        embedding_21 = embeddings["/".join(line.split()[2].split("/")[-2:])]
        # Compute the scores
        # score_1 = torch.mean(
        #     torch.matmul(embedding_11, embedding_21.T)
        # )
        # score = score_1
        score = score_fn(embedding_11, embedding_21)
        score = score.detach().numpy()
        scores.append(score)
        labels.append(int(line.split()[0]))
        i += 1
        if i == n_samples:
            break

    # Coumpute EER and minDCF
    # EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    # fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    # minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    # return EER, minDCF

    return EER(labels, scores)
