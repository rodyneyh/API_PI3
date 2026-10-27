"""API de segmentacao pulmonar (etapa 1: valores fixos).

Subir, a partir da raiz do repositorio:
    uvicorn api.main:app --reload
"""
import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PASTA_DADOS = Path(__file__).parent / "dados_exemplo"
METODOS = ("baseline", "region_growing", "unet")

app = FastAPI(
    title="API de segmentação pulmonar",
    description="Ponte entre src/luna16/ e a tela. Etapa 1: valores fixos.",
    version="0.1.0",
)

# Liberado para desenvolvimento: a tela do front roda em outra porta.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def erro(status, codigo, mensagem):
    """Todo erro previsivel sai neste formato: {"detail": {"codigo", "mensagem"}}."""
    return HTTPException(
        status_code=status,
        detail={"codigo": codigo, "mensagem": mensagem},
    )


def ler_json(nome):
    return json.loads((PASTA_DADOS / nome).read_text(encoding="utf-8"))


def png_base64(caminho):
    return base64.b64encode(caminho.read_bytes()).decode("ascii")


def largura_altura_png(caminho):
    """Le largura e altura do cabecalho IHDR de um PNG."""
    dados = caminho.read_bytes()[16:24]
    return int.from_bytes(dados[:4], "big"), int.from_bytes(dados[4:], "big")


def buscar_exame(exame_id):
    for exame in ler_json("exames.json"):
        if exame["id"] == exame_id:
            return exame
    raise erro(
        404,
        "exame_inexistente",
        f"Exame '{exame_id}' não existe. Use GET /exames para ver os disponíveis.",
    )


@app.get("/saude")
def saude():
    return {"status": "ok", "modo": "valores_fixos"}


@app.get("/exames")
def listar_exames():
    return {"exames": ler_json("exames.json")}


@app.get("/exames/{exame_id}/segmentacao/{metodo}")
def segmentar(exame_id: str, metodo: str):
    if metodo not in METODOS:
        raise erro(
            400,
            "metodo_invalido",
            f"Método '{metodo}' inválido. Use um destes: {', '.join(METODOS)}.",
        )
    exame = buscar_exame(exame_id)

    pasta = PASTA_DADOS / exame_id
    fatia_png = pasta / "fatia.png"
    mascara_png = pasta / f"mascara_{metodo}.png"
    largura, altura = largura_altura_png(fatia_png)
    mascara_largura, mascara_altura = largura_altura_png(mascara_png)

    return {
        "exame_id": exame_id,
        "metodo": metodo,
        "modo": "valores_fixos",
        "fatia": {
            "indice": exame["fatia_exemplo"],
            "largura": largura,
            "altura": altura,
            "png_base64": png_base64(fatia_png),
        },
        "mascara": {
            "largura": mascara_largura,
            "altura": mascara_altura,
            "png_base64": png_base64(mascara_png),
        },
        "metricas": ler_json("metricas.json")[exame_id][metodo],
    }
