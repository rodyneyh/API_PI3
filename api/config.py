"""Configuração da API, lida de variáveis de ambiente.

API_MODO          "fixo" (padrão) ou "real"
API_PASTA_DADOS   dados de exemplo do modo fixo   (padrão: api/dados_exemplo)
API_PASTA_TREINO  exames preparados do modo real  (padrão: data/luna16/treino)
API_PESOS_UNET    pesos da U-Net                  (padrão: data/luna16/unet_baseline.pt)
API_UNET_LOTE     fatias por lote na U-Net        (padrão: 8)
API_CORS_ORIGENS  origens permitidas, separadas por vírgula (padrão: "*")

Os caminhos do modo real são relativos à raiz do repositório: rode sempre de lá.
"""
import os
from dataclasses import dataclass
from pathlib import Path

MODOS = ("fixo", "real")


@dataclass(frozen=True, slots=True)
class Config:
    modo: str = "fixo"
    pasta_dados: Path = Path(__file__).parent / "dados_exemplo"
    pasta_treino: Path = Path("data/luna16/treino")
    pesos_unet: Path = Path("data/luna16/unet_baseline.pt")
    unet_lote: int = 8
    cors_origens: tuple[str, ...] = ("*",)


def carregar_config() -> Config:
    padrao = Config()
    modo = os.environ.get("API_MODO", padrao.modo).strip().lower()
    if modo not in MODOS:
        raise ValueError(f"API_MODO='{modo}' inválido. Use um destes: {', '.join(MODOS)}.")
    origens = os.environ.get("API_CORS_ORIGENS", "*")
    return Config(
        modo=modo,
        pasta_dados=Path(os.environ.get("API_PASTA_DADOS", padrao.pasta_dados)),
        pasta_treino=Path(os.environ.get("API_PASTA_TREINO", padrao.pasta_treino)),
        pesos_unet=Path(os.environ.get("API_PESOS_UNET", padrao.pesos_unet)),
        unet_lote=int(os.environ.get("API_UNET_LOTE", padrao.unet_lote)),
        cors_origens=tuple(o.strip() for o in origens.split(",") if o.strip()),
    )
