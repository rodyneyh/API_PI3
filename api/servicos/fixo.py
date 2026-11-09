"""Modo fixo (API_MODO=fixo): respostas inventadas de api/dados_exemplo/.

Serve para a tela rodar sem os dados do LUNA16 nem torch/SimpleITK.
"""
import json
from pathlib import Path

from api.erros import exame_inexistente

PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados_exemplo"


def ler_json(nome):
    return json.loads((PASTA_DADOS / nome).read_text(encoding="utf-8"))


class ServicoFixo:
    modo = "valores_fixos"

    def __init__(self):
        self.exames = ler_json("exames.json")
        self.metricas = ler_json("metricas.json")

    def listar_exames(self):
        return self.exames

    def _exame(self, exame_id):
        for exame in self.exames:
            if exame["id"] == exame_id:
                return exame
        raise exame_inexistente(exame_id)

    def fatia(self, exame_id):
        """(indice, png) da fatia de exemplo."""
        exame = self._exame(exame_id)
        return exame["fatia_exemplo"], (PASTA_DADOS / exame_id / "fatia.png").read_bytes()

    def segmentar(self, exame_id, metodo):
        """(png da mascara, metricas) do metodo na fatia de exemplo."""
        self._exame(exame_id)
        png = (PASTA_DADOS / exame_id / f"mascara_{metodo}.png").read_bytes()
        return png, self.metricas[exame_id][metodo]
