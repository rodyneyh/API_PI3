"""Gera os dados inventados de api/dados_exemplo/ (etapa 1, valores fixos).

Os arquivos gerados ja estao commitados; este script so existe para
documentar de onde eles vieram e permitir regenerar. Nao e usado pela API.

Uso, a partir da raiz do repositorio:
    python api/gerar_dados_exemplo.py
"""
import json
import struct
import zlib
from pathlib import Path

import numpy as np

PASTA = Path(__file__).parent / "dados_exemplo"
TAMANHO = 128

# (id, nome, indice da fatia, deslocamento do nodulo)
EXAMES = [
    ("demo-001", "Exame de demonstração 1", 120, (0, 0)),
    ("demo-002", "Exame de demonstração 2", 87, (6, -4)),
]

METRICAS = {
    "demo-001": {
        "baseline": {"dice": 0.71, "iou": 0.55, "precisao": 0.62,
                     "revocacao": 0.84, "volume_pulmonar_ml": 4980.0, "tempo_ms": 850},
        "region_growing": {"dice": 0.90, "iou": 0.82, "precisao": 0.92,
                           "revocacao": 0.88, "volume_pulmonar_ml": 4310.0, "tempo_ms": 2400},
        "unet": {"dice": 0.96, "iou": 0.92, "precisao": 0.97,
                 "revocacao": 0.95, "volume_pulmonar_ml": 4420.0, "tempo_ms": 380},
    },
    "demo-002": {
        "baseline": {"dice": 0.68, "iou": 0.52, "precisao": 0.59,
                     "revocacao": 0.81, "volume_pulmonar_ml": 5210.0, "tempo_ms": 910},
        "region_growing": {"dice": 0.88, "iou": 0.79, "precisao": 0.90,
                           "revocacao": 0.86, "volume_pulmonar_ml": 4590.0, "tempo_ms": 2650},
        "unet": {"dice": 0.95, "iou": 0.90, "precisao": 0.96,
                 "revocacao": 0.94, "volume_pulmonar_ml": 4700.0, "tempo_ms": 395},
    },
}


def escrever_png(caminho, matriz):
    """PNG em tons de cinza, 8 bits, escrito so com stdlib."""
    h, w = matriz.shape
    cru = b"".join(b"\x00" + matriz[y].tobytes() for y in range(h))

    def bloco(tipo, dados):
        corpo = tipo + dados
        return (struct.pack(">I", len(dados)) + corpo
                + struct.pack(">I", zlib.crc32(corpo) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + bloco(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + bloco(b"IDAT", zlib.compress(cru, 9))
           + bloco(b"IEND", b""))
    Path(caminho).write_bytes(png)


def elipse(cy, cx, ry, rx):
    y, x = np.mgrid[0:TAMANHO, 0:TAMANHO]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def pulmoes(encolher=0):
    return (elipse(64, 42, 34 - encolher, 22 - encolher)
            | elipse(64, 86, 34 - encolher, 22 - encolher))


def gerar_fatia(desloc):
    img = np.zeros((TAMANHO, TAMANHO), dtype=np.uint8)
    img[elipse(64, 64, 56, 58)] = 150          # corpo
    img[pulmoes()] = 25                        # pulmoes
    img[elipse(100 + desloc[0], 64, 8, 6)] = 230  # coluna
    img[elipse(58 + desloc[0], 40 + desloc[1], 4, 4)] = 140  # nodulo inventado
    return img


def gerar_mascara(metodo):
    if metodo == "baseline":
        m = pulmoes(encolher=3)
        m[:10, :10] = True       # ar fora do corpo "vazando" para a mascara
        return m
    if metodo == "region_growing":
        return pulmoes(encolher=1)
    return pulmoes()             # unet


def main():
    PASTA.mkdir(exist_ok=True)
    exames = []
    for exame_id, nome, indice, desloc in EXAMES:
        pasta = PASTA / exame_id
        pasta.mkdir(exist_ok=True)
        escrever_png(pasta / "fatia.png", gerar_fatia(desloc))
        for metodo in ("baseline", "region_growing", "unet"):
            mascara = gerar_mascara(metodo).astype(np.uint8) * 255
            escrever_png(pasta / f"mascara_{metodo}.png", mascara)
        exames.append({
            "id": exame_id,
            "nome": nome,
            "num_fatias": 240,
            "fatia_exemplo": indice,
            "baixado": True,
        })
    (PASTA / "exames.json").write_text(
        json.dumps(exames, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (PASTA / "metricas.json").write_text(
        json.dumps(METRICAS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dados de exemplo gerados em {PASTA}")


if __name__ == "__main__":
    main()
