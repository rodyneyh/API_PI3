"""Rotas. Sem lógica de negócio: tudo vem do serviço (fixo ou real)."""
from fastapi import APIRouter, Depends, Request, Response

from .erros import metodo_invalido
from .png import PNG
from .schemas import METODOS, Erro, Fatia, Imagem, ListaExames, Segmentacao
from .servicos.base import Servico

roteador = APIRouter()

CACHE = "public, max-age=300"

ERROS = {
    400: {"model": Erro, "description": "método inválido"},
    404: {"model": Erro, "description": "exame inexistente"},
    409: {"model": Erro, "description": "exame existe, mas não está baixado"},
    503: {"model": Erro, "description": "modelo indisponível (modo real)"},
}
RESPOSTA_PNG = {200: {"content": {"image/png": {}}, "description": "PNG cinza de 8 bits"}}


def obter_servico(request: Request) -> Servico:
    return request.app.state.servico


def validar_metodo(metodo: str) -> str:
    if metodo not in METODOS:
        raise metodo_invalido(metodo, METODOS)
    return metodo


def caminho(request: Request, nome: str, **params: str) -> str:
    return request.url_for(nome, **params).path


def responder_png(request: Request, png: PNG) -> Response:
    """PNG com ETag: se o cliente já tem a imagem, responde 304 sem corpo."""
    cabecalhos = {"ETag": png.etag, "Cache-Control": CACHE}
    enviados = request.headers.get("if-none-match", "")
    etags = {e.strip().removeprefix("W/") for e in enviados.split(",")}
    if png.etag in etags or "*" in etags:
        return Response(status_code=304, headers=cabecalhos)
    return Response(content=png.dados, media_type="image/png", headers=cabecalhos)


@roteador.get("/saude")
def saude(servico: Servico = Depends(obter_servico)) -> dict[str, str]:
    return {"status": "ok", "modo": servico.modo}


@roteador.get("/exames", response_model=ListaExames)
def listar_exames(servico: Servico = Depends(obter_servico)):
    return ListaExames(exames=servico.listar_exames())


@roteador.get(
    "/exames/{exame_id}/segmentacao/{metodo}",
    response_model=Segmentacao,
    responses=ERROS,
)
def segmentar(
    request: Request,
    exame_id: str,
    metodo: str = Depends(validar_metodo),
    servico: Servico = Depends(obter_servico),
):
    r = servico.segmentar(exame_id, metodo)
    return Segmentacao(
        exame_id=exame_id,
        metodo=metodo,
        modo=servico.modo,
        fatia=Fatia(
            indice=r.fatia_indice,
            largura=r.fatia.largura,
            altura=r.fatia.altura,
            url=caminho(request, "fatia_png", exame_id=exame_id),
        ),
        mascara=Imagem(
            largura=r.mascara.largura,
            altura=r.mascara.altura,
            url=caminho(request, "mascara_png", exame_id=exame_id, metodo=metodo),
        ),
        metricas=r.metricas,
    )


@roteador.get(
    "/exames/{exame_id}/fatia.png",
    name="fatia_png",
    response_class=Response,
    responses={**RESPOSTA_PNG, **ERROS},
)
def fatia_png(request: Request, exame_id: str, servico: Servico = Depends(obter_servico)):
    return responder_png(request, servico.fatia(exame_id))


@roteador.get(
    "/exames/{exame_id}/segmentacao/{metodo}/mascara.png",
    name="mascara_png",
    response_class=Response,
    responses={**RESPOSTA_PNG, **ERROS},
)
def mascara_png(
    request: Request,
    exame_id: str,
    metodo: str = Depends(validar_metodo),
    servico: Servico = Depends(obter_servico),
):
    return responder_png(request, servico.segmentar(exame_id, metodo).mascara)
