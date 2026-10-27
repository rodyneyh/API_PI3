"""Modo real (API_MODO=real): roda os metodos do projeto sobre os volumes locais.

AVISO: src/luna16/inference.py ainda nao existe. Enquanto isso, a funcao unica
de inferencia e `inferir()` aqui embaixo, e e o UNICO ponto da API que chama os
metodos. Quando o inference.py existir, basta trocar `inferir` por um import.

Reaproveita, sem copiar:
  - api/segmentation.py     -> baseline() (limiar -600 HU) e region_growing()
  - src/luna16/unet.py      -> UNet, pesos em data/luna16/unet_baseline.pt
  - api/preparar_fatias.py  -> mesmo pre-processamento do treino (clip -1000..400, /1400, 256x256)

Dados: data/luna16/treino/<exame>/{imagem,mascara}.npy (gerados por
api/preparar_treino.py). O espacamento do voxel vem de espacamento.json na
mesma pasta (gerado por api/gerar_espacamento.py); sem ele,
volume_pulmonar_ml sai null.

Na subida: carrega a U-Net e os volumes dos exames de demonstracao uma vez.
Cada (exame, metodo) e calculado uma vez e fica em cache.
"""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

from api import segmentation
from api.erros import exame_inexistente, exame_nao_baixado
from api.png import codificar_png
from src.luna16.unet import UNet

HU_MIN, HU_MAX = -1000.0, 400.0
TAMANHO_UNET = 256  # mesmo tamanho de api/preparar_fatias.py
LOTE_UNET = 8

PASTA_PADRAO = "data/luna16/treino"
PESOS_PADRAO = "data/luna16/unet_baseline.pt"
EXAMES_PADRAO = "LIDC-IDRI-0001,LIDC-IDRI-0002,LIDC-IDRI-0003"


def carregar_unet(pesos):
    modelo = UNet()
    modelo.load_state_dict(torch.load(pesos, map_location="cpu", weights_only=True))
    modelo.eval()
    return modelo


def segmentar_unet(modelo, volume):
    """U-Net 2D fatia a fatia, de volta na grade original do exame."""
    _, altura, largura = volume.shape
    fy, fx = TAMANHO_UNET / altura, TAMANHO_UNET / largura
    mascara = np.zeros(volume.shape, dtype=bool)
    with torch.inference_mode():
        for inicio in range(0, len(volume), LOTE_UNET):
            entrada = np.stack([
                zoom((np.clip(f, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN), (fy, fx), order=1)
                for f in volume[inicio:inicio + LOTE_UNET]
            ]).astype(np.float32)[:, None]
            previsto = torch.sigmoid(modelo(torch.from_numpy(entrada)))[:, 0].numpy() > 0.5
            for i, p in enumerate(previsto):
                mascara[inicio + i] = zoom(p.astype(np.uint8), (1 / fy, 1 / fx), order=0) > 0
    return mascara


def inferir(metodo, volume, voxel_mm3, modelo_unet):
    """Funcao unica de inferencia: volume em HU (z, y, x) -> mascara booleana."""
    if metodo == "baseline":
        return segmentation.baseline(volume)
    if metodo == "region_growing":
        return segmentation.region_growing(volume, voxel_mm3)
    if metodo == "unet":
        return segmentar_unet(modelo_unet, volume)
    raise ValueError(f"metodo desconhecido: {metodo}")


def comparar(previsto, referencia, voxel_mm3):
    vp = int(np.count_nonzero(previsto & referencia))
    fp = int(np.count_nonzero(previsto & ~referencia))
    fn = int(np.count_nonzero(~previsto & referencia))

    def razao(a, b):
        return round(a / b, 4) if b else 0.0

    return {
        "dice": razao(2 * vp, 2 * vp + fp + fn),
        "iou": razao(vp, vp + fp + fn),
        "precisao": razao(vp, vp + fp),
        "revocacao": razao(vp, vp + fn),
        "volume_pulmonar_ml": (
            round(float(np.count_nonzero(previsto)) * voxel_mm3 / 1000, 1)
            if voxel_mm3 else None
        ),
    }


def janela_uint8(fatia_hu):
    """HU -> 0..255 na mesma janela do treino (-1000 a 400)."""
    return ((np.clip(fatia_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)


class ServicoReal:
    modo = "modelo_real"

    def __init__(self, pasta=None, pesos=None, exames=None):
        self.pasta = Path(pasta or os.getenv("API_PASTA_DADOS", PASTA_PADRAO))
        pesos = Path(pesos or os.getenv("API_PESOS_UNET", PESOS_PADRAO))
        ids = exames or [e.strip() for e in os.getenv("API_EXAMES", EXAMES_PADRAO).split(",") if e.strip()]

        if not pesos.exists():
            raise RuntimeError(f"Pesos da U-Net não encontrados em '{pesos}'. Use API_MODO=fixo.")
        self.modelo = carregar_unet(pesos)

        self.exames = {}   # id -> dict do contrato
        self.volumes = {}  # id -> (volume, referencia, voxel_mm3)
        for exame_id in ids:
            self.exames[exame_id] = self._carregar(exame_id)

        self.cache = {}    # (id, metodo) -> (png, metricas)
        self.travas = {}
        self.trava_geral = threading.Lock()

    def _carregar(self, exame_id):
        pasta = self.pasta / exame_id
        imagem, mascara = pasta / "imagem.npy", pasta / "mascara.npy"
        if not (imagem.exists() and mascara.exists()):
            return {"id": exame_id, "nome": f"Exame {exame_id}",
                    "num_fatias": 0, "fatia_exemplo": 0, "baixado": False}

        volume = np.load(imagem)
        referencia = np.load(mascara).astype(bool)
        espacamento = pasta / "espacamento.json"
        voxel_mm3 = (float(np.prod(json.loads(espacamento.read_text())))
                     if espacamento.exists() else None)
        self.volumes[exame_id] = (volume, referencia, voxel_mm3)

        # Fatia de exemplo: a de maior area de pulmao na mascara de referencia.
        area = referencia.reshape(len(referencia), -1).sum(axis=1)
        return {"id": exame_id, "nome": f"Exame {exame_id}",
                "num_fatias": int(volume.shape[0]),
                "fatia_exemplo": int(np.argmax(area)), "baixado": True}

    def listar_exames(self):
        return list(self.exames.values())

    def _exame(self, exame_id):
        if exame_id not in self.exames:
            raise exame_inexistente(exame_id)
        if exame_id not in self.volumes:
            raise exame_nao_baixado(exame_id, self.pasta / exame_id)
        return self.exames[exame_id]

    def fatia(self, exame_id):
        indice = self._exame(exame_id)["fatia_exemplo"]
        volume = self.volumes[exame_id][0]
        return indice, codificar_png(janela_uint8(volume[indice]))

    def segmentar(self, exame_id, metodo):
        exame = self._exame(exame_id)
        chave = (exame_id, metodo)
        with self.trava_geral:
            trava = self.travas.setdefault(chave, threading.Lock())
        with trava:  # duas requisicoes iguais ao mesmo tempo calculam uma vez so
            if chave not in self.cache:
                volume, referencia, voxel_mm3 = self.volumes[exame_id]
                inicio = time.perf_counter()
                # region_growing usa o voxel so para descartar componentes pequenos.
                previsto = inferir(metodo, volume, voxel_mm3 or 1.0, self.modelo)
                tempo_ms = round((time.perf_counter() - inicio) * 1000)
                metricas = comparar(previsto, referencia, voxel_mm3)
                metricas["tempo_ms"] = tempo_ms
                png = codificar_png(previsto[exame["fatia_exemplo"]].astype(np.uint8) * 255)
                self.cache[chave] = (png, metricas)
        return self.cache[chave]
