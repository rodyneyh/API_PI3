from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.luna16.unet import UNet




CAMINHO_IMAGENS = Path("data/luna16/fatias/imagens.npy")
CAMINHO_MASCARAS = Path("data/luna16/fatias/mascaras.npy")

CAMINHO_MODELO = Path("data/luna16/unet_baseline.pt")

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.001

SEMENTE = 42


# =========================
# DATASET
# =========================

class LungDataset(Dataset):

    def __init__(self, imagens, mascaras):
        self.imagens = imagens
        self.mascaras = mascaras

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, indice):

        imagem = torch.from_numpy(
            self.imagens[indice]
        ).float()

        mascara = torch.from_numpy(
            self.mascaras[indice]
        ).float()

        return imagem, mascara


# =========================
# CARREGAR DADOS
# =========================

print("=" * 70)
print("TREINAMENTO DA U-NET")
print("=" * 70)

print("\nCarregando dados...")

imagens = np.load(CAMINHO_IMAGENS)
mascaras = np.load(CAMINHO_MASCARAS)

print("Imagens:", imagens.shape)
print("Máscaras:", mascaras.shape)


# =========================
# DATASET
# =========================

dataset = LungDataset(
    imagens,
    mascaras
)


# =========================
# DIVISÃO TREINO / VALIDAÇÃO
# =========================

quantidade_treino = int(len(dataset) * 0.8)

quantidade_validacao = (
    len(dataset) - quantidade_treino
)

gerador = torch.Generator().manual_seed(SEMENTE)

treino, validacao = random_split(
    dataset,
    [quantidade_treino, quantidade_validacao],
    generator=gerador
)


print("\nDivisão dos dados:")
print("Treinamento:", len(treino))
print("Validação:", len(validacao))


# =========================
# DATALOADERS
# =========================

loader_treino = DataLoader(
    treino,
    batch_size=BATCH_SIZE,
    shuffle=True
)

loader_validacao = DataLoader(
    validacao,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# DISPOSITIVO
# =========================

dispositivo = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nDispositivo:", dispositivo)


# =========================
# MODELO
# =========================

modelo = UNet().to(dispositivo)

print("\nU-Net criada.")


# =========================
# FUNÇÃO DE PERDA
# =========================

criterio = nn.BCEWithLogitsLoss()


# =========================
# OTIMIZADOR
# =========================

otimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TREINAMENTO
# =========================

for epoca in range(EPOCHS):

    modelo.train()

    perda_treino = 0.0

    for imagens_batch, mascaras_batch in loader_treino:

        imagens_batch = imagens_batch.to(dispositivo)
        mascaras_batch = mascaras_batch.to(dispositivo)

        otimizador.zero_grad()

        saida = modelo(imagens_batch)

        perda = criterio(
            saida,
            mascaras_batch
        )

        perda.backward()

        otimizador.step()

        perda_treino += perda.item()

    perda_treino /= len(loader_treino)


    # =========================
    # VALIDAÇÃO
    # =========================

    modelo.eval()

    perda_validacao = 0.0

    with torch.no_grad():

        for imagens_batch, mascaras_batch in loader_validacao:

            imagens_batch = imagens_batch.to(dispositivo)
            mascaras_batch = mascaras_batch.to(dispositivo)

            saida = modelo(imagens_batch)

            perda = criterio(
                saida,
                mascaras_batch
            )

            perda_validacao += perda.item()

    perda_validacao /= len(loader_validacao)


    print(
        f"Época {epoca + 1}/{EPOCHS} "
        f"- treino: {perda_treino:.4f} "
        f"- validação: {perda_validacao:.4f}"
    )


# =========================
# SALVAR MODELO
# =========================

CAMINHO_MODELO.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    modelo.state_dict(),
    CAMINHO_MODELO
)


print("\n" + "=" * 70)
print("TREINAMENTO FINALIZADO")
print("=" * 70)

print("\nModelo salvo em:")
print(CAMINHO_MODELO)