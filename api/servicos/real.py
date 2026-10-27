"""Modo real: roda os métodos que já existem no projeto sobre os volumes preparados.

Reaproveita, sem copiar:
  - api/segmentation.py     -> baseline() e region_growing()
  - src/luna16/unet.py      -> UNet (pesos em data/luna16/unet_baseline.pt)
  - api/preparar_fatias.py  -> o mesmo pré-processamento da U-Net (clip -1000..400, /1400)

Os volumes vêm de data/luna16/treino/<exame>/{imagem,mascara}.npy, gerados por
api/preparar_treino.py. Esses .npy não guardam o espaçamento do voxel, por isso
o volume em mL não é calculado (volume_pulmonar_ml = null).

Cada (exame, método) é calculado uma vez e fica em cache; a U-Net é carregada
uma única vez.
"""
import threading
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from ..config import Config
from ..erros import exame_inexistente, exame_nao_baixado, modelo_indisponivel
from ..metricas import comparar
from ..png import PNG, codificar_png, montar_png
from ..schemas import Exame, Metricas
from .base import ResultadoSegmentacao

HU_MIN, HU_MAX = -1000.0, 400.0
TAMANHO_UNET = 256  # mesmo tamanho usado em preparar_fatias.py
VOXEL_MM3 = 1.0  # só para o region_growing descartar componentes minúsculos


def janela_para_uint8(fatia_hu: np.ndarray) -> np.ndarray:
    """HU -> 0..255 na janela de exibição do projeto (-1000 a 400)."""
    normalizada = (np.clip(fatia_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return (normalizada * 255).astype(np.uint8)


class ServicoReal:
    modo = "modelo_real"

    def __init__(self, config: Config):
        self._config = config
        self._pasta = config.pasta_treino
        self._resultados: dict[tuple[str, str], ResultadoSegmentacao] = {}
        self._fatias: dict[str, PNG] = {}
        self._trava_geral = threading.Lock()
        self._travas: dict[tuple[str, str], threading.Lock] = {}
        self._unet = None
        self._trava_unet = threading.Lock()

    # ---------- exames ----------

    def _caminhos(self, exame_id: str) -> tuple[Path, Path]:
        pasta = self._pasta / exame_id
        return pasta / "imagem.npy", pasta / "mascara.npy"

    def _exame(self, pasta: Path) -> Exame:
        imagem, mascara = pasta / "imagem.npy", pasta / "mascara.npy"
        if not (imagem.exists() and mascara.exists()):
            return Exame(id=pasta.name, nome=pasta.name, baixado=False)
        # mmap lê só o cabeçalho: listar exames não carrega nenhum volume.
        fatias = int(np.load(imagem, mmap_mode="r").shape[0])
        return Exame(id=pasta.name, nome=pasta.name, num_fatias=fatias,
                     fatia_exemplo=fatias // 2, baixado=True)

    def listar_exames(self) -> list[Exame]:
        if not self._pasta.is_dir():
            return []
        return [self._exame(p) for p in sorted(self._pasta.iterdir()) if p.is_dir()]

    def _disponivel(self, exame_id: str) -> Exame:
        pasta = self._pasta / exame_id
        # exame_id vem da URL: nada de sair da pasta de dados.
        if not pasta.is_dir() or pasta.resolve().parent != self._pasta.resolve():
            raise exame_inexistente(exame_id)
        exame = self._exame(pasta)
        if not exame.baixado:
            raise exame_nao_baixado(exame_id)
        return exame

    # ---------- modelos ----------

    def _modelo_unet(self):
        with self._trava_unet:
            if self._unet is None:
                try:
                    import torch

                    from src.luna16.unet import UNet
                except ImportError as e:
                    raise modelo_indisponivel(f"dependência faltando ({e.name}).") from e
                pesos = self._config.pesos_unet
                if not pesos.exists():
                    raise modelo_indisponivel(f"arquivo de pesos não encontrado em '{pesos}'.")
                modelo = UNet()
                modelo.load_state_dict(torch.load(pesos, map_location="cpu", weights_only=True))
                modelo.eval()
                self._unet = modelo
        return self._unet

    def _rodar_unet(self, volume: np.ndarray) -> np.ndarray:
        import torch

        modelo = self._modelo_unet()
        _, altura, largura = volume.shape
        fy, fx = TAMANHO_UNET / altura, TAMANHO_UNET / largura
        mascara = np.zeros(volume.shape, dtype=bool)

        lote = self._config.unet_lote
        with torch.inference_mode():
            for inicio in range(0, len(volume), lote):
                fim = min(inicio + lote, len(volume))
                # mesmo pré-processamento de api/preparar_fatias.py
                entrada = np.stack([
                    zoom((np.clip(f, HU_MIN, HU_MAX) + 1000) / 1400.0, (fy, fx), order=1)
                    for f in volume[inicio:fim]
                ]).astype(np.float32)[:, None]
                probs = torch.sigmoid(modelo(torch.from_numpy(entrada)))[:, 0].numpy() > 0.5
                for i, p in enumerate(probs):
                    mascara[inicio + i] = zoom(p.astype(np.uint8), (1 / fy, 1 / fx), order=0) > 0
        return mascara

    def _rodar_metodo(self, metodo: str, volume: np.ndarray) -> np.ndarray:
        if metodo == "unet":
            return self._rodar_unet(volume)
        try:
            from .. import segmentation  # lazy: exige SimpleITK e scipy
        except ImportError as e:
            raise modelo_indisponivel(f"dependência faltando ({e.name}).") from e

        if metodo == "baseline":
            return segmentation.baseline(volume)
        return segmentation.region_growing(volume, VOXEL_MM3)

    # ---------- respostas ----------

    def fatia(self, exame_id: str) -> PNG:
        exame = self._disponivel(exame_id)
        if exame_id not in self._fatias:
            imagem = np.load(self._caminhos(exame_id)[0], mmap_mode="r")
            png = codificar_png(janela_para_uint8(np.asarray(imagem[exame.fatia_exemplo])))
            self._fatias[exame_id] = montar_png(png)
        return self._fatias[exame_id]

    def segmentar(self, exame_id: str, metodo: str) -> ResultadoSegmentacao:
        exame = self._disponivel(exame_id)
        chave = (exame_id, metodo)

        with self._trava_geral:
            trava = self._travas.setdefault(chave, threading.Lock())
        with trava:  # duas requisições iguais ao mesmo tempo calculam uma vez só
            if chave in self._resultados:
                return self._resultados[chave]

            caminho_imagem, caminho_mascara = self._caminhos(exame_id)
            volume = np.load(caminho_imagem)
            referencia = np.load(caminho_mascara).astype(bool)

            inicio = time.perf_counter()
            previsto = self._rodar_metodo(metodo, volume)
            tempo_ms = round((time.perf_counter() - inicio) * 1000)

            indice = exame.fatia_exemplo
            mascara_png = codificar_png(previsto[indice].astype(np.uint8) * 255)
            resultado = ResultadoSegmentacao(
                fatia_indice=indice,
                fatia=self.fatia(exame_id),
                mascara=montar_png(mascara_png),
                metricas=Metricas(**comparar(previsto, referencia), tempo_ms=tempo_ms),
            )
            self._resultados[chave] = resultado
            return resultado
