import torch

if not torch.cuda.is_available():
    print("CUDA driver is not installed.")
else:
    print("CUDA driver is installed.")

if torch.backends.cudnn.is_available():
    print("cuDNN is installed.")
else:
    print("cuDNN is not installed.")
    
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Number of GPUs:", torch.cuda.device_count())