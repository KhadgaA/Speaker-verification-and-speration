import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *

def S_Norm(trials, znorm, tnorm):
    '''
    The symmetric normalization (S-norm) computes an average of
    normalized scores from Z-norm and T-norm. S-norm is
    symmetrical as s(e, t) = s(t, e), while the previously mentioned 
    normalizations depend on the order of e and t.
    '''
    # Check if trials are the same in znorm and tnorm
    znorm.sort_values('trials', inplace=True)
    tnorm.sort_values('trials', inplace=True)
    assert all(znorm['trials']) == all(tnorm['trials'])
    
    # Compute average of normalized scores from Z-norm and T-norm
    trials['normalized_score'] = (znorm['normalized_score'] + tnorm['normalized_score'])/2
    
    return trials

def eval_network(model, eval_list, eval_path,device,n_samples=-1):
    model.eval()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()

    i = 0
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
        i+=1
        if i== n_samples:
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

        embeddings[file] = embedding_1 #[embedding_1, embedding_2]
    scores, labels = [], []
    i = 0
    for line in lines:

        embedding_11 = embeddings[line.split()[1]]
        embedding_21 = embeddings[line.split()[2]]
        # Compute the scores
        score_1 = torch.mean(
            torch.matmul(embedding_11, embedding_21.T)
        )  # higher is positive

        score = score_1
        score = score.detach().numpy()
        scores.append(score)
        labels.append(int(line.split()[0]))
        i+=1
        if i== n_samples:
            break

    # Coumpute EER and minDCF
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF
