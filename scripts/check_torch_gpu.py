import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("gpu tensor test:", y.shape)
else:
    raise RuntimeError("CUDA is not available")
