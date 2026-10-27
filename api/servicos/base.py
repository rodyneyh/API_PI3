"""Contrato entre as rotas e a origem dos dados (valores fixos ou modelos reais)."""
from dataclasses import dataclass
from typing import Protocol

from ..png import PNG
from ..schemas import Exame, Metricas


@dataclass(frozen=True, slots=True)
class ResultadoSegmentacao:
    fatia_indice: int
    fatia: PNG
    mascara: PNG
    metricas: Metricas


class Servico(Protocol):
    modo: str  # "valores_fixos" ou "modelo_real"

    def listar_exames(self) -> list[Exame]: ...

    def fatia(self, exame_id: str) -> PNG: ...

    def segmentar(self, exame_id: str, metodo: str) -> ResultadoSegmentacao: ...
