"""Modo real (API_MODO=real): os tres metodos rodando sobre os DICOM do HD.

Fluxo por exame (o mesmo de pipeline.run_pipeline_for_uid, usado no treino e
na avaliacao da U-Net):

    DICOM do LIDC-IDRI  --preparar_ct-->  volume HU limpo, suavizado, 1 mm
    mascara LUNA16      --preparar_referencia-->  pulmao (rotulos 3+4), 1 mm
    inferir(metodo, volume)  -> mascara prevista -> metricas contra a referencia

Pecas reaproveitadas:
  - api/servicos/dicom.py        -> acha e le a serie DICOM e a mascara no HD
  - src/luna16/preprocessing.py  -> pre-processamento do treino
  - src/luna16/inference.py      -> inferir(): baseline, region growing, U-Net

Na subida: carrega a U-Net e os exames de API_EXAMES uma vez. Cada
(exame, metodo) e calculado uma vez e fica em cache.
"""
import os
import threading
import time
from pathlib import Path

import numpy as np

from api.erros import exame_inexistente, exame_nao_baixado
from api.png import codificar_png
from api.servicos.dicom import RAIZ_PADRAO, FonteDicom
from src.luna16 import inference
from src.luna16.preprocessing import CLIP_MAX_HU, CLIP_MIN_HU, ESPACAMENTO_ALVO, preparar_ct, preparar_referencia

# Modelo da reexecucao com seed 123 (497 exames de treino, Dice 0,983 nos 107
# do teste_interno), copiado do HD para o repositorio: funciona sem o HD.
# Ver modelos/unet_seed123/README.md.
PESOS_PADRAO = str(Path(__file__).resolve().parents[2] / "modelos" / "unet_seed123" / "unet_novo_final.pt")
CONJUNTO_PADRAO = "teste_interno"
QTD_EXAMES_PADRAO = 3

NOMES_CONJUNTO = {
    "treino": "treino",
    "validacao": "validação",
    "teste_interno": "teste",
    "anterior": "teste antigo",
}


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
        "volume_pulmonar_ml": round(float(np.count_nonzero(previsto)) * voxel_mm3 / 1000, 1),
    }


def janela_uint8(fatia_hu):
    """HU -> 0..255 na janela -1000 a 400 (a mesma do pre-processamento)."""
    return ((np.clip(fatia_hu, CLIP_MIN_HU, CLIP_MAX_HU) - CLIP_MIN_HU)
            / (CLIP_MAX_HU - CLIP_MIN_HU) * 255).astype(np.uint8)


def exames_da_env():
    return [e.strip() for e in os.getenv("API_EXAMES", "").split(",") if e.strip()]


class ServicoReal:
    modo = "modelo_real"
    voxel_mm3 = float(np.prod(ESPACAMENTO_ALVO))

    def __init__(self, fonte=None, pesos=None, exames=None):
        self.fonte = fonte or FonteDicom(os.getenv("API_RAIZ_DATASET", RAIZ_PADRAO))
        pesos = pesos or os.getenv("API_PESOS_UNET", PESOS_PADRAO)
        if not os.path.exists(pesos):
            raise RuntimeError(f"Pesos da U-Net não encontrados em '{pesos}'. Ajuste API_PESOS_UNET ou use API_MODO=fixo.")
        self.modelo = inference.carregar_unet(pesos)

        ids = exames or exames_da_env() or self.fonte.pacientes_do_conjunto(CONJUNTO_PADRAO)[:QTD_EXAMES_PADRAO]
        if not ids:
            raise RuntimeError("Nenhum exame para servir: defina API_EXAMES (ex.: LIDC-IDRI-0013).")

        self.exames = {}   # id -> dict do contrato
        self.volumes = {}  # id -> (volume, referencia)
        self.motivos = {}  # id -> por que nao esta disponivel
        for exame_id in dict.fromkeys(ids):  # sem repetidos, na ordem dada
            self.exames[exame_id] = self._carregar(exame_id)

        self.cache = {}    # (id, metodo) -> (png, metricas)
        self.travas = {}
        self.trava_geral = threading.Lock()

    def _carregar(self, exame_id):
        serie, motivo = self.fonte.serie(exame_id)
        if serie is None:
            self.motivos[exame_id] = motivo
            return {"id": exame_id, "nome": f"Exame {exame_id}",
                    "num_fatias": 0, "fatia_exemplo": 0, "baixado": False}

        inicio = time.perf_counter()
        ct = self.fonte.ler_ct(serie)
        volume = preparar_ct(ct)
        referencia = preparar_referencia(self.fonte.ler_mascara(serie), ct)
        self.volumes[exame_id] = (volume, referencia)
        print(f"[api] {exame_id}: {volume.shape} a 1 mm, carregado em {time.perf_counter() - inicio:.1f} s", flush=True)

        # Fatia de exemplo: a de maior area de pulmao na mascara de referencia.
        area = referencia.reshape(len(referencia), -1).sum(axis=1)
        nome = f"Exame {exame_id}"
        if serie.uid in self.modelo.uids_treino:
            nome += " (usado no treino da U-Net)"
        elif serie.conjunto in NOMES_CONJUNTO:
            nome += f" ({NOMES_CONJUNTO[serie.conjunto]})"
        return {"id": exame_id, "nome": nome,
                "num_fatias": int(volume.shape[0]),
                "fatia_exemplo": int(np.argmax(area)), "baixado": True}

    def listar_exames(self):
        return list(self.exames.values())

    def _exame(self, exame_id):
        if exame_id not in self.exames:
            raise exame_inexistente(exame_id)
        if exame_id not in self.volumes:
            raise exame_nao_baixado(exame_id, self.motivos.get(exame_id, ""))
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
                volume, referencia = self.volumes[exame_id]
                inicio = time.perf_counter()
                previsto = inference.inferir(metodo, volume, self.modelo, self.voxel_mm3)
                tempo_ms = round((time.perf_counter() - inicio) * 1000)
                metricas = comparar(previsto, referencia, self.voxel_mm3)
                metricas["tempo_ms"] = tempo_ms
                png = codificar_png(previsto[exame["fatia_exemplo"]].astype(np.uint8) * 255)
                self.cache[chave] = (png, metricas)
        return self.cache[chave]
