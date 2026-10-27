"""Contrato da API no modo fixo, campo a campo.

Rodar, a partir da raiz do repositório:  python -m pytest
"""
import numpy as np
import pytest

from api.config import carregar_config
from api.schemas import METODOS

from .formato import CAMPOS_EXAME, decodificar_png, verificar_segmentacao

COMBINACOES = [(e, m) for e in ("demo-001", "demo-002") for m in METODOS]


def test_saude(cliente):
    r = cliente.get("/saude")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modo": "valores_fixos"}


def test_listar_exames(cliente):
    r = cliente.get("/exames")
    assert r.status_code == 200
    assert set(r.json()) == {"exames"}
    exames = r.json()["exames"]
    assert len({e["id"] for e in exames}) == len(exames)
    for e in exames:
        assert set(e) == CAMPOS_EXAME
        assert isinstance(e["id"], str) and isinstance(e["nome"], str)
        assert isinstance(e["baixado"], bool)
        if e["baixado"]:
            assert isinstance(e["num_fatias"], int) and e["num_fatias"] > 0
            assert 0 <= e["fatia_exemplo"] < e["num_fatias"]
    assert any(e["baixado"] for e in exames)
    assert any(not e["baixado"] for e in exames)


@pytest.mark.parametrize("exame_id,metodo", COMBINACOES)
def test_segmentacao(cliente, exame_id, metodo):
    verificar_segmentacao(cliente, exame_id, metodo, "valores_fixos")


def test_mascara_nao_vazia(cliente):
    url = cliente.get("/exames/demo-001/segmentacao/unet").json()["mascara"]["url"]
    assert (decodificar_png(cliente.get(url).content) == 255).any()


def test_etag_devolve_304_sem_corpo(cliente):
    url = "/exames/demo-001/fatia.png"
    primeira = cliente.get(url)
    segunda = cliente.get(url, headers={"If-None-Match": primeira.headers["etag"]})
    assert segunda.status_code == 304
    assert segunda.content == b""
    assert segunda.headers["etag"] == primeira.headers["etag"]
    # ETag diferente -> imagem completa de novo
    assert cliente.get(url, headers={"If-None-Match": '"outro"'}).status_code == 200


@pytest.mark.parametrize("url", [
    "/exames/xyz/segmentacao/unet",
    "/exames/xyz/segmentacao/unet/mascara.png",
    "/exames/xyz/fatia.png",
])
def test_exame_inexistente(cliente, url):
    r = cliente.get(url)
    assert r.status_code == 404
    assert r.json()["detail"]["codigo"] == "exame_inexistente"
    assert "xyz" in r.json()["detail"]["mensagem"]


@pytest.mark.parametrize("url", [
    "/exames/demo-001/segmentacao/magia",
    "/exames/demo-001/segmentacao/magia/mascara.png",
])
def test_metodo_invalido(cliente, url):
    r = cliente.get(url)
    assert r.status_code == 400
    detalhe = r.json()["detail"]
    assert set(detalhe) == {"codigo", "mensagem"}
    assert detalhe["codigo"] == "metodo_invalido"
    assert all(m in detalhe["mensagem"] for m in METODOS)


@pytest.mark.parametrize("url", [
    "/exames/demo-003/segmentacao/unet",
    "/exames/demo-003/segmentacao/unet/mascara.png",
    "/exames/demo-003/fatia.png",
])
def test_exame_nao_baixado(cliente, url):
    r = cliente.get(url)
    assert r.status_code == 409
    assert r.json()["detail"]["codigo"] == "exame_nao_baixado"


def test_modo_invalido_na_configuracao(monkeypatch):
    monkeypatch.setenv("API_MODO", "magia")
    with pytest.raises(ValueError, match="API_MODO"):
        carregar_config()


def test_configuracao_padrao(monkeypatch):
    for nome in ("API_MODO", "API_PASTA_TREINO", "API_PESOS_UNET", "API_UNET_LOTE"):
        monkeypatch.delenv(nome, raising=False)
    config = carregar_config()
    assert config.modo == "fixo"
    assert str(config.pasta_treino) == "data/luna16/treino"  # relativo, de propósito
    assert str(config.pesos_unet) == "data/luna16/unet_baseline.pt"
