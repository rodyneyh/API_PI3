import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.luna16.unet import UNet


modelo = UNet()

entrada = torch.randn(1, 1, 512, 512)

saida = modelo(entrada)

print("U-Net criada com sucesso!")
print("Entrada:", entrada.shape)
print("Saída:", saida.shape)