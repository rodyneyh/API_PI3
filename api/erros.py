"""Erros previsiveis da API, sempre no formato {"detail": {"codigo", "mensagem"}}."""
from fastapi import HTTPException

METODOS = ("baseline", "region_growing", "unet")


def erro(status, codigo, mensagem):
    return HTTPException(
        status_code=status,
        detail={"codigo": codigo, "mensagem": mensagem},
    )


def exame_inexistente(exame_id):
    return erro(
        404,
        "exame_inexistente",
        f"Exame '{exame_id}' não existe. Use GET /exames para ver os disponíveis.",
    )


def metodo_invalido(metodo):
    return erro(
        400,
        "metodo_invalido",
        f"Método '{metodo}' inválido. Use um destes: {', '.join(METODOS)}.",
    )


def exame_nao_baixado(exame_id, pasta):
    return erro(
        409,
        "exame_nao_baixado",
        f"Exame '{exame_id}' está na lista, mas não foi baixado: faltam "
        f"imagem.npy e mascara.npy em '{pasta}'. Gere com api/preparar_treino.py "
        f"ou suba a API com API_MODO=fixo.",
    )
