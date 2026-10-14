import torch

print("=" * 50)
print("TESTE DA GPU")
print("=" * 50)

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    dispositivo = torch.device("cuda")

    a = torch.randn(2000, 2000, device=dispositivo)
    b = torch.randn(2000, 2000, device=dispositivo)

    c = torch.matmul(a, b)

    torch.cuda.synchronize()

    print("Multiplicação realizada na GPU!")
    print("Resultado:", c.shape)

    memoria = torch.cuda.memory_allocated() / 1024**2

    print(f"Memória GPU utilizada: {memoria:.2f} MB")

else:

    print("CUDA não está disponível!")

print("=" * 50)
