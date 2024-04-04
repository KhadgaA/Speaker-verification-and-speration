import torch
import tqdm
import numpy
import soundfile
import torch.nn.functional as F
from torch.multiprocessing import spawn, Process, set_start_method
from tools import *


def process_file(eval_path, device, model, file):
    audio, _ = soundfile.read(os.path.join(eval_path, file))
    data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(device)
    with torch.no_grad():
        embedding_1 = model(data_1)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1).detach().cpu()
    return file, embedding_1


def eval_network(model, eval_list, eval_path, device, n_samples=-1):
    model.eval()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()

    i = 0
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
        i += 1
        if i == n_samples:
            break
    setfiles = list(set(files))
    setfiles.sort()

    num_processes = torch.cuda.device_count()  # Number of available GPUs
    results = []
    with spawn(
        fn=process_file,
        args=(eval_path, device, model),
        nprocs=num_processes,
    ) as pool:

        for file in setfiles:
            pool.spawn(process_file, args=(eval_path, device, model, file))

        for future in pool.workers:
            result = future.result()
            results.append(result)

    embeddings = {file: embedding for file, embedding in results}

    scores, labels = [], []
    i = 0
    for line in lines:
        embedding_11 = embeddings[line.split()[1]]
        embedding_21 = embeddings[line.split()[2]]

        score_1 = torch.mean(
            torch.matmul(embedding_11, embedding_21.T)
        )  # higher is positive
        score = score_1
        score = score.detach().numpy()
        scores.append(score)
        labels.append(int(line.split()[0]))
        i += 1
        if i == n_samples:
            break

    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    # eval_path = "your_eval_path_here"
    # eval_list = "your_eval_list_here"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = your_model_definition()  # Replace with your model definition

    # EER, minDCF = evaluate_files(eval_path, device, model, eval_list)
    # print("EER:", EER)
    # print("minDCF:", minDCF)
