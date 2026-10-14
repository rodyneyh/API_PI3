from pathlib import Path
import gc

import numpy as np
from scipy.ndimage import zoom


ENTRADA = Path("data/luna16/treino")
SAIDA = Path("data/luna16/fatias")

TAMANHO = 256

SAIDA.mkdir(parents=True, exist_ok=True)

# Pasta temporária para as fatias
TEMP = SAIDA / "temporarias"
TEMP.mkdir(parents=True, exist_ok=True)


pacientes = sorted(
    [p for p in ENTRADA.iterdir() if p.is_dir()]
)

print("=" * 60)
print("PREPARANDO FATIAS DOS EXAMES")
print("=" * 60)
print(f"Exames encontrados: {len(pacientes)}")


total_fatias = 0


for numero, paciente in enumerate(pacientes, start=1):

    print(
        f"\n[{numero}/{len(pacientes)}] "
        f"Processando: {paciente.name}"
    )

    caminho_imagem = paciente / "imagem.npy"
    caminho_mascara = paciente / "mascara.npy"

    imagem = np.load(caminho_imagem)
    mascara = np.load(caminho_mascara)

    fatias_imagem = []
    fatias_mascara = []

    usadas = 0

    for z in range(imagem.shape[0]):

        mascara_slice = mascara[z]

        # Ignora fatias sem máscara
        if mascara_slice.sum() == 0:
            continue

        fatia = imagem[z]

        # ---------------------------------------------
        # CLIP DE HU
        # ---------------------------------------------

        fatia = np.clip(
            fatia,
            -1000,
            400
        )

        # ---------------------------------------------
        # NORMALIZAÇÃO 0-1
        # ---------------------------------------------

        fatia = (
            fatia + 1000
        ) / 1400.0

        # ---------------------------------------------
        # REDIMENSIONAMENTO
        # ---------------------------------------------

        fator_y = TAMANHO / fatia.shape[0]
        fator_x = TAMANHO / fatia.shape[1]

        fatia = zoom(
            fatia,
            (fator_y, fator_x),
            order=1
        )

        mascara_slice = zoom(
            mascara_slice.astype(np.uint8),
            (fator_y, fator_x),
            order=0
        )

        # ---------------------------------------------
        # CANAL
        # ---------------------------------------------

        fatias_imagem.append(
            fatia.astype(np.float32)[None, :, :]
        )

        fatias_mascara.append(
            (mascara_slice > 0).astype(np.uint8)[None, :, :]
        )

        usadas += 1

    # ---------------------------------------------
    # SALVAR ESTE EXAME
    # ---------------------------------------------

    if usadas > 0:

        imagens_np = np.stack(
            fatias_imagem
        )

        mascaras_np = np.stack(
            fatias_mascara
        )

        np.save(
            TEMP / f"{paciente.name}_imagens.npy",
            imagens_np
        )

        np.save(
            TEMP / f"{paciente.name}_mascaras.npy",
            mascaras_np
        )

        total_fatias += usadas

        print(
            f"  Fatias utilizadas: {usadas}"
        )

    # ---------------------------------------------
    # LIBERAR MEMÓRIA
    # ---------------------------------------------

    del imagem
    del mascara
    del fatias_imagem
    del fatias_mascara

    if usadas > 0:
        del imagens_np
        del mascaras_np

    gc.collect()


print("\n" + "=" * 60)
print("EXAMES PROCESSADOS")
print("=" * 60)

print(f"Total de fatias: {total_fatias}")


# ============================================================
# JUNTAR OS ARQUIVOS EM LOTES
# ============================================================

print("\nCriando dataset final...")

arquivos_imagem = sorted(
    TEMP.glob("*_imagens.npy")
)

arquivos_mascara = sorted(
    TEMP.glob("*_mascaras.npy")
)

# Usaremos memmap para não manter tudo na RAM
arquivo_imagens = SAIDA / "imagens.npy"
arquivo_mascaras = SAIDA / "mascaras.npy"


# Descobrir quantidade total
quantidade_total = 0

for arquivo in arquivos_imagem:

    dados = np.load(
        arquivo,
        mmap_mode="r"
    )

    quantidade_total += dados.shape[0]

    del dados


print(
    f"Total de fatias para salvar: "
    f"{quantidade_total}"
)


# Criar arquivos finais usando memmap
imagens_final = np.lib.format.open_memmap(
    arquivo_imagens,
    mode="w+",
    dtype=np.float32,
    shape=(
        quantidade_total,
        1,
        TAMANHO,
        TAMANHO
    )
)

mascaras_final = np.lib.format.open_memmap(
    arquivo_mascaras,
    mode="w+",
    dtype=np.uint8,
    shape=(
        quantidade_total,
        1,
        TAMANHO,
        TAMANHO
    )
)


posicao = 0


for arquivo_img, arquivo_mask in zip(
    arquivos_imagem,
    arquivos_mascara
):

    imgs = np.load(
        arquivo_img,
        mmap_mode="r"
    )

    masks = np.load(
        arquivo_mask,
        mmap_mode="r"
    )

    quantidade = imgs.shape[0]

    imagens_final[
        posicao:posicao + quantidade
    ] = imgs

    mascaras_final[
        posicao:posicao + quantidade
    ] = masks

    posicao += quantidade

    print(
        f"  Adicionado: "
        f"{arquivo_img.stem.replace('_imagens', '')} "
        f"({quantidade} fatias)"
    )

    del imgs
    del masks

    gc.collect()


imagens_final.flush()
mascaras_final.flush()

del imagens_final
del mascaras_final

gc.collect()


print("\n" + "=" * 60)
print("FINALIZADO")
print("=" * 60)

print(
    f"Imagens: {arquivo_imagens}"
)

print(
    f"Máscaras: {arquivo_mascaras}"
)

print(
    f"Total de fatias: {quantidade_total}"
)

print("\nDataset pronto para treinamento!")