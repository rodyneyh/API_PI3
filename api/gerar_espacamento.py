"""Grava data/luna16/treino/<exame>/espacamento.json para o modo real da API.

O api/preparar_treino.py salva os volumes em .npy sem o espacamento do voxel,
que e necessario para calcular o volume pulmonar em mL. Este script le o
espacamento do cabecalho da mascara do LUNA16 (<SeriesInstanceUID>.mhd), sem
carregar a imagem, e confere se o tamanho bate com o imagem.npy.

Uso, a partir da raiz do repositorio:
    python api/gerar_espacamento.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

METADATA = Path(r"D:\DATASET_LUNA16\metadata\metadata.csv")
RAIZ_MASCARAS = Path(r"D:\DATASET_LUNA16\mascaras\seg-lungs-LUNA16\seg-lungs-LUNA16")
PASTA_TREINO = Path("data/luna16/treino")


def ler_cabecalho(caminho):
    leitor = sitk.ImageFileReader()
    leitor.SetFileName(str(caminho))
    leitor.ReadImageInformation()
    return leitor.GetSpacing(), leitor.GetSize()


def main():
    series = {}
    with METADATA.open(newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            series.setdefault(linha["PatientID"], []).append(linha["SeriesInstanceUID"])

    gravados = 0
    for pasta in sorted(p for p in PASTA_TREINO.iterdir() if p.is_dir()):
        forma = np.load(pasta / "imagem.npy", mmap_mode="r").shape  # (z, y, x)
        for uid in series.get(pasta.name, []):
            mhd = RAIZ_MASCARAS / f"{uid}.mhd"
            if not mhd.exists():
                continue
            espacamento, tamanho = ler_cabecalho(mhd)
            if tuple(tamanho[::-1]) == forma:
                (pasta / "espacamento.json").write_text(json.dumps(list(espacamento)))
                gravados += 1
                break
        else:
            print(f"{pasta.name}: série com máscara de tamanho {forma} não encontrada")
    print(f"espacamento.json gravado em {gravados} exames")


if __name__ == "__main__":
    main()
