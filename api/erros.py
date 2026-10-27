"""Erros previsíveis. Todos saem como {"detail": {"codigo", "mensagem"}}."""
from fastapi import Request
from fastapi.responses import JSONResponse


class ErroAPI(Exception):
    def __init__(self, status: int, codigo: str, mensagem: str):
        super().__init__(mensagem)
        self.status = status
        self.codigo = codigo
        self.mensagem = mensagem


def exame_inexistente(exame_id: str) -> ErroAPI:
    return ErroAPI(
        404,
        "exame_inexistente",
        f"Exame '{exame_id}' não existe. Use GET /exames para ver os disponíveis.",
    )


def metodo_invalido(metodo: str, validos: tuple[str, ...]) -> ErroAPI:
    return ErroAPI(
        400,
        "metodo_invalido",
        f"Método '{metodo}' inválido. Use um destes: {', '.join(validos)}.",
    )


def exame_nao_baixado(exame_id: str) -> ErroAPI:
    return ErroAPI(
        409,
        "exame_nao_baixado",
        f"Exame '{exame_id}' existe, mas os dados dele não estão nesta máquina.",
    )


def modelo_indisponivel(motivo: str) -> ErroAPI:
    return ErroAPI(503, "modelo_indisponivel", f"Modelo indisponível: {motivo}")


async def tratar_erro_api(_: Request, erro: ErroAPI) -> JSONResponse:
    return JSONResponse(
        status_code=erro.status,
        content={"detail": {"codigo": erro.codigo, "mensagem": erro.mensagem}},
    )
