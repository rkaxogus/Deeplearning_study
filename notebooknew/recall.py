import torch
print("GPU 사용 가능:", torch.cuda.is_available())
print("PyTorch 버전:", torch.__version__)
print("CUDA 지원 여부:", torch.version.cuda)
