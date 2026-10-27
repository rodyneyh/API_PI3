"""Modo real com um volume sintetico (nao precisa do LUNA16, so dos pesos da U-Net).

Roda os tres metodos de verdade sobre um "torso" de 16x96x96 voxels e confere o
erro 409, o cache e o formato. Pula se faltar torch/SimpleITK ou os pesos.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("SimpleITK")

from api.main import METODOS, app, obter_servico  # noqa: E402
from api.servicos import real  # noqa: E402

PESOS = Path(real.PESOS_PADRAO)
pytestmark = pytest.mark.skipif(not PESOS.exists(), reason=f"sem pesos em {PESOS}")


def volume_sintetico():
    z, y, x = np.mgrid[0:16, 0:96, 0:96]
    volume = np.full((16, 96, 96), -1000, dtype=np.int32)       # ar fora do corpo
    volume[((y - 48) / 40) ** 2 + ((x - 48) / 44) ** 2 <= 1] = 40  # corpo
    pulmao = np.zeros_like(volume, dtype=bool)
    for cx in (28, 68):
        pulmao |= (((y - 48) / 24) ** 2 + ((x - cx) / 13) ** 2 + ((z - 8) / 9) ** 2) <= 1
    volume[pulmao] = -850
    return volume, pulmao


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("treino")
    (pasta / "sintetico").mkdir()
    volume, pulmao = volume_sintetico()
    np.save(pasta / "sintetico" / "imagem.npy", volume)
    np.save(pasta / "sintetico" / "mascara.npy", pulmao)
    (pasta / "sintetico" / "espacamento.json").write_text(json.dumps([2.0, 2.0, 2.0]))

    servico = real.ServicoReal(pasta=pasta, pesos=PESOS, exames=["sintetico", "faltando"])
    app.dependency_overrides[obter_servico] = lambda: servico
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_lista_exames_reais(cliente):
    exames = {e["id"]: e for e in cliente.get("/exames").json()["exames"]}
    assert exames["sintetico"]["baixado"] is True
    assert exames["sintetico"]["num_fatias"] == 16
    assert exames["sintetico"]["fatia_exemplo"] == 8  # maior area de pulmao
    assert exames["faltando"]["baixado"] is False


@pytest.mark.parametrize("rota", [
    "segmentacao/unet", "fatia.png", "segmentacao/baseline/mascara.png",
])
def test_exame_nao_baixado(cliente, rota):
    r = cliente.get(f"/exames/faltando/{rota}")
    assert r.status_code == 409
    detalhe = r.json()["detail"]
    assert set(detalhe) == {"codigo", "mensagem"}
    assert detalhe["codigo"] == "exame_nao_baixado"
    assert "faltando" in detalhe["mensagem"]


@pytest.mark.parametrize("metodo", METODOS)
def test_segmentacao_real(cliente, metodo):
    corpo = cliente.get(f"/exames/sintetico/segmentacao/{metodo}").json()
    assert corpo["modo"] == "modelo_real"
    assert (corpo["fatia"]["largura"], corpo["fatia"]["altura"]) == (96, 96)
    assert (corpo["mascara"]["largura"], corpo["mascara"]["altura"]) == (96, 96)
    m = corpo["metricas"]
    assert isinstance(m["volume_pulmonar_ml"], float) and m["volume_pulmonar_ml"] > 0
    assert isinstance(m["tempo_ms"], int)
    if metodo == "region_growing":
        # caso facil: o crescimento de regiao tem de achar os dois pulmoes
        assert m["dice"] > 0.8


def test_cache_calcula_uma_vez(cliente, monkeypatch):
    chamadas = []
    original = real.inferir
    monkeypatch.setattr(real, "inferir", lambda *a: chamadas.append(a[0]) or original(*a))
    servico = app.dependency_overrides[obter_servico]()
    servico.cache.pop(("sintetico", "baseline"), None)

    primeira = cliente.get("/exames/sintetico/segmentacao/baseline").json()
    cliente.get("/exames/sintetico/segmentacao/baseline/mascara.png")
    segunda = cliente.get("/exames/sintetico/segmentacao/baseline").json()
    assert chamadas == ["baseline"]
    assert primeira["metricas"] == segunda["metricas"]
