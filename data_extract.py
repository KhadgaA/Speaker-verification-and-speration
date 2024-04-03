import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)
voxceleb_hard = torchaudio.datasets.VoxCeleb1Verification(
    r"/mnt/d/programming/datasets/New", download=True
)

# voxceleb = torchaudio.datasets.VoxCeleb1Verification(r'/mnt/d/programming/datasets/VoxCeleb', download=True,meta_url="/mnt/d/programming/datasets/VoxCeleb/list_test_hard2.txt")
print("data_correctly parsed")
