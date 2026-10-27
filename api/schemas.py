"""Formato das respostas (o contrato da API). O FastAPI usa estes modelos para
validar, serializar e gerar a documentação em /docs."""
from typing import Literal, get_args

from pydantic import BaseModel, Field

Metodo = Literal["baseline", "region_growing", "unet"]
METODOS: tuple[str, ...] = get_args(Metodo)


class Exame(BaseModel):
    id: str
    nome: str
    # null quando o exame não está baixado (os dados não estão na máquina)
    num_fatias: int | None = Field(default=None, gt=0)
    fatia_exemplo: int | None = Field(default=None, ge=0)
    baixado: bool


class ListaExames(BaseModel):
    exames: list[Exame]


class Metricas(BaseModel):
    dice: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    precisao: float = Field(ge=0, le=1)
    revocacao: float = Field(ge=0, le=1)
    # null no modo real: os .npy do projeto não guardam o espaçamento do voxel
    volume_pulmonar_ml: float | None = Field(default=None, gt=0)
    tempo_ms: int = Field(ge=0)


class Imagem(BaseModel):
    largura: int
    altura: int
    url: str  # PNG cinza de 8 bits, baixado numa segunda requisição (com cache)


class Fatia(Imagem):
    indice: int


class Segmentacao(BaseModel):
    exame_id: str
    metodo: Metodo
    modo: Literal["valores_fixos", "modelo_real"]
    fatia: Fatia
    mascara: Imagem  # 0 = fundo, 255 = pulmão, na mesma grade da fatia
    metricas: Metricas


class DetalheErro(BaseModel):
    codigo: str
    mensagem: str


class Erro(BaseModel):
    detail: DetalheErro
