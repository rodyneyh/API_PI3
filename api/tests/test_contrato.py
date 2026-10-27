"""Testes do contrato da API, campo a campo.

Valem na etapa 1 (valores fixos) e na etapa 2 (modelos reais): so conferem
o FORMATO das respostas, nunca os valores das metricas.

Rodar, a partir da raiz do repositorio:  pytest
"""
import base64
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from api.main import METODOS, app

cliente = TestClient(app)

ASSINATURA_PNG = b"\x89PNG\r\n\x1a\n"
CAMPOS_EXAME = {"id", "nome", "num_fatias", "fatia_exemplo", "baixado"}
CAMPOS_METRICAS = {
    "dice", "iou", "precisao", "revocacao", "volume_pulmonar_ml", "tempo_ms",
}


def decodificar_png(texto_base64):
    """Decodifica um PNG cinza 8 bits; devolve (largura, altura, linhas de pixels)."""
    dados = base64.b64decode(texto_base64, validate=True)
    assert dados.startswith(ASSINATURA_PNG), "não é um PNG"
    largura, altura, profundidade, tipo_cor = struct.unpack(">IIBB", dados[16:26])
    assert (profundidade, tipo_cor) == (8, 0), "esperado PNG cinza de 8 bits"
    idat, pos = b"", 8
    while pos < len(dados):
        tamanho = struct.unpack(">I", dados[pos:pos + 4])[0]
        if dados[pos + 4:pos + 8] == b"IDAT":
            idat += dados[pos + 8:pos + 8 + tamanho]
        pos += 12 + tamanho
    cru = zlib.decompress(idat)
    linhas = [cru[i * (largura + 1) + 1:(i + 1) * (largura + 1)] for i in range(altura)]
    return largura, altura, linhas


def test_listar_exames_formato():
    r = cliente.get("/exames")
    assert r.status_code == 200
    corpo = r.json()
    assert set(corpo) == {"exames"}
    assert isinstance(corpo["exames"], list) and corpo["exames"]
    for exame in corpo["exames"]:
        assert set(exame) == CAMPOS_EXAME
        assert isinstance(exame["id"], str) and exame["id"]
        assert isinstance(exame["nome"], str) and exame["nome"]
        assert isinstance(exame["num_fatias"], int) and exame["num_fatias"] > 0
        assert isinstance(exame["fatia_exemplo"], int)
        assert 0 <= exame["fatia_exemplo"] < exame["num_fatias"]
        assert isinstance(exame["baixado"], bool)


def test_ids_dos_exames_sao_unicos():
    ids = [e["id"] for e in cliente.get("/exames").json()["exames"]]
    assert len(ids) == len(set(ids))


def pares_exame_metodo():
    exames = cliente.get("/exames").json()["exames"]
    return [(e["id"], m) for e in exames for m in METODOS]


@pytest.mark.parametrize("exame_id,metodo", pares_exame_metodo())
def test_segmentacao_formato(exame_id, metodo):
    r = cliente.get(f"/exames/{exame_id}/segmentacao/{metodo}")
    assert r.status_code == 200
    corpo = r.json()

    assert set(corpo) == {"exame_id", "metodo", "modo", "fatia", "mascara", "metricas"}
    assert corpo["exame_id"] == exame_id
    assert corpo["metodo"] == metodo
    assert corpo["modo"] in ("valores_fixos", "modelo_real")

    assert set(corpo["fatia"]) == {"indice", "largura", "altura", "png_base64"}
    assert isinstance(corpo["fatia"]["indice"], int)
    assert isinstance(corpo["fatia"]["largura"], int)
    assert isinstance(corpo["fatia"]["altura"], int)
    assert set(corpo["mascara"]) == {"largura", "altura", "png_base64"}

    metricas = corpo["metricas"]
    assert set(metricas) == CAMPOS_METRICAS
    for campo in ("dice", "iou", "precisao", "revocacao"):
        assert isinstance(metricas[campo], float)
        assert 0.0 <= metricas[campo] <= 1.0
    assert isinstance(metricas["volume_pulmonar_ml"], float)
    assert metricas["volume_pulmonar_ml"] > 0
    assert isinstance(metricas["tempo_ms"], int) and metricas["tempo_ms"] >= 0


@pytest.mark.parametrize("exame_id,metodo", pares_exame_metodo())
def test_imagens_sao_png_validos_e_batem_com_as_dimensoes(exame_id, metodo):
    corpo = cliente.get(f"/exames/{exame_id}/segmentacao/{metodo}").json()
    fatia, mascara = corpo["fatia"], corpo["mascara"]

    largura, altura, _ = decodificar_png(fatia["png_base64"])
    assert (largura, altura) == (fatia["largura"], fatia["altura"])

    largura, altura, linhas = decodificar_png(mascara["png_base64"])
    assert (largura, altura) == (mascara["largura"], mascara["altura"])
    # A máscara tem de cobrir a mesma grade da fatia para a tela sobrepor.
    assert (largura, altura) == (fatia["largura"], fatia["altura"])
    # Só 0 (fundo) e 255 (pulmão), e com algum pulmão marcado.
    valores = set(b"".join(linhas))
    assert valores <= {0, 255}
    assert 255 in valores


def test_exame_inexistente():
    r = cliente.get("/exames/nao-existe/segmentacao/unet")
    assert r.status_code == 404
    detalhe = r.json()["detail"]
    assert set(detalhe) == {"codigo", "mensagem"}
    assert detalhe["codigo"] == "exame_inexistente"
    assert "nao-existe" in detalhe["mensagem"]


def test_metodo_invalido():
    exame_id = cliente.get("/exames").json()["exames"][0]["id"]
    r = cliente.get(f"/exames/{exame_id}/segmentacao/magia")
    assert r.status_code == 400
    detalhe = r.json()["detail"]
    assert set(detalhe) == {"codigo", "mensagem"}
    assert detalhe["codigo"] == "metodo_invalido"
    for metodo in METODOS:
        assert metodo in detalhe["mensagem"]


def test_saude():
    r = cliente.get("/saude")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "modo"}
    assert r.json()["status"] == "ok"
