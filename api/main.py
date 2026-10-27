"""API de segmentacao pulmonar.

Subir, a partir da raiz do repositorio:
    uvicorn api.main:app --reload                       # API_MODO=fixo (padrao)
    API_MODO=real uvicorn api.main:app --reload         # modelos reais

API_MODO=fixo  -> respostas inventadas de api/dados_exemplo/ (sem dados do LUNA16)
API_MODO=real  -> api/servicos/real.py (U-Net e volumes carregados na subida)
"""
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api.erros import METODOS, metodo_invalido
from api.png import largura_altura

_servico = None


def criar_servico():
    modo = os.getenv("API_MODO", "fixo").strip().lower()
    if modo == "fixo":
        from api.servicos.fixo import ServicoFixo
        return ServicoFixo()
    if modo == "real":
        # import aqui dentro: o modo fixo nao precisa de torch nem SimpleITK
        from api.servicos.real import ServicoReal
        return ServicoReal()
    raise RuntimeError(f"API_MODO='{modo}' inválido. Use 'fixo' ou 'real'.")


def obter_servico():
    global _servico
    if _servico is None:
        _servico = criar_servico()
    return _servico


@asynccontextmanager
async def lifespan(app):
    obter_servico()  # carrega modelo e dados uma vez, na subida
    yield


app = FastAPI(
    title="API de segmentação pulmonar",
    description="Ponte entre os métodos de segmentação e a tela. "
                "API_MODO=fixo (valores fixos) ou API_MODO=real (modelos reais).",
    version="0.2.0",
    lifespan=lifespan,
)

# Liberado para desenvolvimento: a tela do front roda em outra porta.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def validar_metodo(metodo):
    if metodo not in METODOS:
        raise metodo_invalido(metodo)


def png(dados):
    return Response(content=dados, media_type="image/png")


@app.get("/saude")
def saude(servico=Depends(obter_servico)):
    return {"status": "ok", "modo": servico.modo}


@app.get("/exames")
def listar_exames(servico=Depends(obter_servico)):
    return {"exames": servico.listar_exames()}


# Rotas sincronas (def): o FastAPI roda em thread, sem travar as outras requisicoes.
@app.get("/exames/{exame_id}/segmentacao/{metodo}")
def segmentar(exame_id: str, metodo: str, servico=Depends(obter_servico)):
    validar_metodo(metodo)
    indice, fatia_png = servico.fatia(exame_id)
    mascara_png, metricas = servico.segmentar(exame_id, metodo)
    largura, altura = largura_altura(fatia_png)
    mascara_largura, mascara_altura = largura_altura(mascara_png)
    return {
        "exame_id": exame_id,
        "metodo": metodo,
        "modo": servico.modo,
        "fatia": {
            "indice": indice,
            "largura": largura,
            "altura": altura,
            "url": f"/exames/{exame_id}/fatia.png",
        },
        "mascara": {
            "largura": mascara_largura,
            "altura": mascara_altura,
            "url": f"/exames/{exame_id}/segmentacao/{metodo}/mascara.png",
        },
        "metricas": metricas,
    }


@app.get("/exames/{exame_id}/fatia.png")
def fatia_png(exame_id: str, servico=Depends(obter_servico)):
    return png(servico.fatia(exame_id)[1])


@app.get("/exames/{exame_id}/segmentacao/{metodo}/mascara.png")
def mascara_png(exame_id: str, metodo: str, servico=Depends(obter_servico)):
    validar_metodo(metodo)
    return png(servico.segmentar(exame_id, metodo)[0])
