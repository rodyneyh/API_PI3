"""Modo real, com exames sintéticos em pasta temporária (não precisa do LUNA16).

Confere o mesmo formato do contrato e os erros do modo real. Pula sozinho se
scipy/SimpleITK não estiverem instalados; o teste da U-Net também precisa de torch.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("scipy")
pytest.importorskip("SimpleITK")

from api.config import Config  # noqa: E402
from api.main import criar_app  # noqa: E402

from .formato import CAMPOS_EXAME, verificar_segmentacao  # noqa: E402


def elipse(forma, centro, raios):
    z, y, x = np.ogrid[:forma[0], :forma[1], :forma[2]]
    return (((z - centro[0]) / raios[0]) ** 2 + ((y - centro[1]) / raios[1]) ** 2
            + ((x - centro[2]) / raios[2]) ** 2) <= 1


def tomografia_sintetica(forma):
    """Ar fora (-1000 HU), corpo (40 HU), dois pulmões (-800 HU) e a referência."""
    z, y, x = forma
    volume = np.full(forma, -1000, dtype=np.int16)
    volume[elipse(forma, (z / 2, y / 2, x / 2), (z / 2.2, y / 2.1, x / 2.1))] = 40
    pulmoes = (elipse(forma, (z / 2, y / 2, x * 0.32), (z * 0.38, y * 0.30, x * 0.17))
               | elipse(forma, (z / 2, y / 2, x * 0.68), (z * 0.38, y * 0.30, x * 0.17)))
    volume[pulmoes] = -800
    return volume, pulmoes


@pytest.fixture(scope="module")
def pasta_treino(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("treino")
    # exame grande: os métodos clássicos descartam componentes pequenos (50 mL)
    volume, referencia = tomografia_sintetica((40, 128, 128))
    (pasta / "exame-001").mkdir()
    np.save(pasta / "exame-001" / "imagem.npy", volume)
    np.save(pasta / "exame-001" / "mascara.npy", referencia)
    # exame minúsculo: só para a U-Net (que redimensiona cada fatia para 256x256)
    volume, referencia = tomografia_sintetica((3, 32, 32))
    (pasta / "exame-002").mkdir()
    np.save(pasta / "exame-002" / "imagem.npy", volume)
    np.save(pasta / "exame-002" / "mascara.npy", referencia)
    # pasta de exame sem os .npy -> existe, mas não está baixado
    (pasta / "exame-003").mkdir()
    return pasta


@pytest.fixture(scope="module")
def cliente_real(pasta_treino, tmp_path_factory):
    config = Config(
        modo="real",
        pasta_treino=pasta_treino,
        pesos_unet=tmp_path_factory.mktemp("pesos") / "nao_existe.pt",
        unet_lote=2,
    )
    with TestClient(criar_app(config)) as c:
        yield c


def test_saude(cliente_real):
    assert cliente_real.get("/saude").json() == {"status": "ok", "modo": "modelo_real"}


def test_listar_exames(cliente_real):
    exames = {e["id"]: e for e in cliente_real.get("/exames").json()["exames"]}
    assert set(exames) == {"exame-001", "exame-002", "exame-003"}
    for e in exames.values():
        assert set(e) == CAMPOS_EXAME
    assert exames["exame-001"]["baixado"] and exames["exame-001"]["num_fatias"] == 40
    assert exames["exame-001"]["fatia_exemplo"] == 20
    assert exames["exame-003"] == {"id": "exame-003", "nome": "exame-003",
                                   "num_fatias": None, "fatia_exemplo": None, "baixado": False}


@pytest.mark.parametrize("metodo", ["baseline", "region_growing"])
def test_segmentacao_classica(cliente_real, metodo):
    verificar_segmentacao(cliente_real, "exame-001", metodo, "modelo_real")


def test_resultado_vem_do_cache(cliente_real):
    primeira = cliente_real.get("/exames/exame-001/segmentacao/baseline").json()
    segunda = cliente_real.get("/exames/exame-001/segmentacao/baseline").json()
    # o tempo_ms é o do cálculo original, não de uma nova execução
    assert primeira["metricas"] == segunda["metricas"]


def test_dice_real_do_region_growing_no_exame_sintetico(cliente_real):
    # Pulmões bem contrastados: o region growing precisa achar quase tudo.
    dice = cliente_real.get("/exames/exame-001/segmentacao/region_growing").json()["metricas"]["dice"]
    assert dice > 0.8


def test_unet_sem_pesos_da_erro_claro(cliente_real):
    pytest.importorskip("torch")
    r = cliente_real.get("/exames/exame-002/segmentacao/unet")
    assert r.status_code == 503
    assert r.json()["detail"]["codigo"] == "modelo_indisponivel"
    assert "nao_existe.pt" in r.json()["detail"]["mensagem"]


def test_unet_com_pesos(pasta_treino, tmp_path):
    torch = pytest.importorskip("torch")
    from src.luna16.unet import UNet

    pesos = tmp_path / "unet.pt"
    torch.save(UNet().state_dict(), pesos)  # pesos aleatórios: só o formato importa
    config = Config(modo="real", pasta_treino=pasta_treino, pesos_unet=pesos, unet_lote=2)
    with TestClient(criar_app(config)) as c:
        verificar_segmentacao(c, "exame-002", "unet", "modelo_real")


def test_exame_inexistente(cliente_real):
    r = cliente_real.get("/exames/nao-existe/segmentacao/baseline")
    assert r.status_code == 404
    assert r.json()["detail"]["codigo"] == "exame_inexistente"


def test_exame_nao_baixado(cliente_real):
    r = cliente_real.get("/exames/exame-003/segmentacao/baseline")
    assert r.status_code == 409
    assert r.json()["detail"]["codigo"] == "exame_nao_baixado"


def test_id_nao_sai_da_pasta_de_dados(cliente_real):
    r = cliente_real.get("/exames/../segmentacao/baseline")
    assert r.status_code == 404
