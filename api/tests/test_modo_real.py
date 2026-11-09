"""Modo real com um exame sintetico (nao precisa do HD, so dos pesos da U-Net).

Roda os tres metodos de verdade, com o mesmo pre-processamento do treino,
sobre um "torso" sintetico, e confere o erro 409, o cache e o formato. Pula se
faltar torch/SimpleITK/scikit-image ou os pesos.
"""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("skimage")
sitk = pytest.importorskip("SimpleITK")

from api.main import METODOS, app, obter_servico  # noqa: E402
from api.servicos import real  # noqa: E402
from api.servicos.dicom import Serie  # noqa: E402
from src.luna16 import inference  # noqa: E402

PESOS = Path(real.PESOS_PADRAO)
pytestmark = pytest.mark.skipif(not PESOS.exists(), reason=f"sem pesos em {PESOS}")

ESPACAMENTO = (2.0, 2.0, 2.0)  # mm; reamostrado para 1 mm => 32 x 192 x 192


def exame_sintetico():
    """CT (HU) e mascara no formato seg-lungs-LUNA16 (3 = dir., 4 = esq., 5 = traqueia)."""
    z, y, x = np.mgrid[0:16, 0:96, 0:96]
    ct = np.full((16, 96, 96), -1000, dtype=np.int16)              # ar fora do corpo
    ct[((y - 48) / 40) ** 2 + ((x - 48) / 44) ** 2 <= 1] = 40       # corpo
    mascara = np.zeros_like(ct, dtype=np.uint8)
    for rotulo, cx in ((3, 28), (4, 68)):
        pulmao = (((y - 48) / 24) ** 2 + ((x - cx) / 13) ** 2 + ((z - 8) / 9) ** 2) <= 1
        ct[pulmao] = -850
        mascara[pulmao] = rotulo
    traqueia = ((y - 30) ** 2 + (x - 48) ** 2 <= 9)
    ct[traqueia] = -950
    mascara[traqueia] = 5  # nao e parenquima: tem de ficar fora da referencia

    imagens = []
    for array in (ct, mascara):
        img = sitk.GetImageFromArray(array)
        img.SetSpacing(ESPACAMENTO)
        img.SetOrigin((-96.0, -96.0, -16.0))
        imagens.append(img)
    return imagens


class FonteMemoria:
    """Mesma interface de api.servicos.dicom.FonteDicom, sem disco."""

    def __init__(self):
        self.ct, self.mascara = exame_sintetico()

    def serie(self, paciente):
        if paciente != "sintetico":
            return None, "pasta DICOM 'faltando' não existe"
        return Serie(paciente, "1.2.3", Path("."), Path("."), "teste_interno"), None

    def ler_ct(self, serie):
        return self.ct

    def ler_mascara(self, serie):
        return self.mascara


@pytest.fixture(scope="module")
def cliente():
    servico = real.ServicoReal(fonte=FonteMemoria(), pesos=PESOS, exames=["sintetico", "faltando"])
    app.dependency_overrides[obter_servico] = lambda: servico
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_lista_exames_reais(cliente):
    exames = {e["id"]: e for e in cliente.get("/exames").json()["exames"]}
    assert exames["sintetico"]["baixado"] is True
    assert exames["sintetico"]["num_fatias"] == 32        # 16 fatias de 2 mm -> 1 mm
    assert exames["sintetico"]["nome"] == "Exame sintetico (teste)"
    assert 14 <= exames["sintetico"]["fatia_exemplo"] <= 17  # maior area de pulmao (centro)
    assert exames["faltando"]["baixado"] is False


def test_referencia_so_tem_pulmao(cliente):
    servico = app.dependency_overrides[obter_servico]()
    _, referencia = servico.volumes["sintetico"]
    assert referencia.shape == (32, 192, 192)
    # a traqueia (rotulo 5) fica de fora, como na avaliacao do treino
    assert not referencia[:, 56:64, 92:100].any()


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
    assert (corpo["fatia"]["largura"], corpo["fatia"]["altura"]) == (192, 192)
    assert (corpo["mascara"]["largura"], corpo["mascara"]["altura"]) == (192, 192)
    m = corpo["metricas"]
    assert isinstance(m["volume_pulmonar_ml"], float)
    assert isinstance(m["tempo_ms"], int)
    if metodo in ("region_growing", "baseline"):
        # caso facil: tem de achar os dois pulmoes; no baseline, sem o ar
        # fora do corpo (antes o fundo inteiro entrava e o Dice caia)
        assert m["dice"] > 0.8


def test_baseline_ignora_ar_fora_do_corpo(cliente):
    from api.segmentation import baseline

    servico = app.dependency_overrides[obter_servico]()
    volume, referencia = servico.volumes["sintetico"]
    novo, antigo = baseline(volume), baseline(volume, confinar=False, morfologia="erosao")
    assert not novo[:, :4, :].any() and not novo[:, :, :4].any()  # bordas = ar de fora
    assert antigo[:, :4, :].any()                                  # versao antiga pegava
    assert novo.sum() < 2 * referencia.sum()


def test_cache_calcula_uma_vez(cliente, monkeypatch):
    chamadas = []
    original = inference.inferir
    monkeypatch.setattr(inference, "inferir", lambda *a: chamadas.append(a[0]) or original(*a))
    servico = app.dependency_overrides[obter_servico]()
    servico.cache.pop(("sintetico", "baseline"), None)

    primeira = cliente.get("/exames/sintetico/segmentacao/baseline").json()
    cliente.get("/exames/sintetico/segmentacao/baseline/mascara.png")
    segunda = cliente.get("/exames/sintetico/segmentacao/baseline").json()
    assert chamadas == ["baseline"]
    assert primeira["metricas"] == segunda["metricas"]


def test_checkpoint_errado_da_erro_claro(tmp_path):
    import torch

    caminho = tmp_path / "pesos_crus.pt"
    torch.save({"enc1.0.weight": torch.zeros(1)}, caminho)  # state_dict sem envelope
    with pytest.raises(ValueError, match="model_state"):
        inference.carregar_unet(caminho)
