"""Modo fixo: respostas inventadas, lidas uma única vez de api/dados_exemplo/."""
import json
from pathlib import Path

from ..erros import exame_inexistente, exame_nao_baixado
from ..png import PNG, montar_png
from ..schemas import METODOS, Exame, Metricas
from .base import ResultadoSegmentacao


class ServicoFixo:
    modo = "valores_fixos"

    def __init__(self, pasta: Path):
        exames = json.loads((pasta / "exames.json").read_text(encoding="utf-8"))
        metricas = json.loads((pasta / "metricas.json").read_text(encoding="utf-8"))

        self._exames = [Exame(**e) for e in exames]
        self._por_id = {e.id: e for e in self._exames}
        self._fatias: dict[str, PNG] = {}
        self._mascaras: dict[tuple[str, str], PNG] = {}
        self._metricas: dict[tuple[str, str], Metricas] = {}

        # Tudo é lido aqui, na subida: um arquivo faltando derruba a API logo,
        # em vez de falhar na primeira requisição.
        for exame in self._exames:
            if not exame.baixado:
                continue
            self._fatias[exame.id] = montar_png((pasta / exame.id / "fatia.png").read_bytes())
            for metodo in METODOS:
                chave = (exame.id, metodo)
                arquivo = pasta / exame.id / f"mascara_{metodo}.png"
                self._mascaras[chave] = montar_png(arquivo.read_bytes())
                self._metricas[chave] = Metricas(**metricas[exame.id][metodo])

    def _disponivel(self, exame_id: str) -> Exame:
        exame = self._por_id.get(exame_id)
        if exame is None:
            raise exame_inexistente(exame_id)
        if not exame.baixado:
            raise exame_nao_baixado(exame_id)
        return exame

    def listar_exames(self) -> list[Exame]:
        return self._exames

    def fatia(self, exame_id: str) -> PNG:
        self._disponivel(exame_id)
        return self._fatias[exame_id]

    def segmentar(self, exame_id: str, metodo: str) -> ResultadoSegmentacao:
        exame = self._disponivel(exame_id)
        chave = (exame_id, metodo)
        return ResultadoSegmentacao(
            fatia_indice=exame.fatia_exemplo,
            fatia=self._fatias[exame_id],
            mascara=self._mascaras[chave],
            metricas=self._metricas[chave],
        )
