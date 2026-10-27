"""API de segmentação pulmonar.

Subir, a partir da raiz do repositório:
    uvicorn api.main:app --reload                  # modo fixo (padrão)
    API_MODO=real uvicorn api.main:app             # modelos reais
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Config, carregar_config
from .erros import ErroAPI, tratar_erro_api
from .rotas import roteador
from .servicos.base import Servico


def criar_servico(config: Config) -> Servico:
    # Import tardio: o modo fixo não precisa de scipy, SimpleITK nem torch.
    if config.modo == "real":
        from .servicos.real import ServicoReal

        return ServicoReal(config)
    from .servicos.fixo import ServicoFixo

    return ServicoFixo(config.pasta_dados)


def criar_app(config: Config | None = None) -> FastAPI:
    config = config or carregar_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.servico = criar_servico(config)  # carregado uma vez, na subida
        yield

    app = FastAPI(
        title="API de segmentação pulmonar",
        description="Ponte entre src/luna16/ e a tela. Modos: fixo (dados de exemplo) e real.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origens),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.add_exception_handler(ErroAPI, tratar_erro_api)
    app.include_router(roteador)
    return app


app = criar_app()
