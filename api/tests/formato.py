"""Conferência de formato, compartilhada pelos testes dos dois modos.

Só olha a FORMA das respostas, nunca os valores das métricas.
"""
import struct
import zlib

import numpy as np

ASSINATURA_PNG = b"\x89PNG\r\n\x1a\n"
CAMPOS_EXAME = {"id", "nome", "num_fatias", "fatia_exemplo", "baixado"}
CAMPOS_METRICAS = {"dice", "iou", "precisao", "revocacao", "volume_pulmonar_ml", "tempo_ms"}


def decodificar_png(dados: bytes) -> np.ndarray:
    """PNG cinza de 8 bits -> matriz (altura, largura)."""
    assert dados.startswith(ASSINATURA_PNG), "não é um PNG"
    largura, altura, profundidade, tipo_cor = struct.unpack(">IIBB", dados[16:26])
    assert (profundidade, tipo_cor) == (8, 0), "esperado PNG cinza de 8 bits"
    idat, pos = b"", 8
    while pos < len(dados):
        tamanho = struct.unpack(">I", dados[pos:pos + 4])[0]
        if dados[pos + 4:pos + 8] == b"IDAT":
            idat += dados[pos + 8:pos + 8 + tamanho]
        pos += 12 + tamanho
    cru = np.frombuffer(zlib.decompress(idat), dtype=np.uint8)
    return cru.reshape(altura, largura + 1)[:, 1:]


def verificar_segmentacao(cliente, exame_id: str, metodo: str, modo: str) -> None:
    """Confere o JSON de segmentação e as duas imagens que ele aponta."""
    r = cliente.get(f"/exames/{exame_id}/segmentacao/{metodo}")
    assert r.status_code == 200, r.text
    corpo = r.json()

    assert set(corpo) == {"exame_id", "metodo", "modo", "fatia", "mascara", "metricas"}
    assert corpo["exame_id"] == exame_id
    assert corpo["metodo"] == metodo
    assert corpo["modo"] == modo

    fatia, mascara = corpo["fatia"], corpo["mascara"]
    assert set(fatia) == {"indice", "largura", "altura", "url"}
    assert set(mascara) == {"largura", "altura", "url"}
    assert isinstance(fatia["indice"], int)
    for campo in ("largura", "altura"):
        assert isinstance(fatia[campo], int) and isinstance(mascara[campo], int)
    assert isinstance(fatia["url"], str) and isinstance(mascara["url"], str)

    metricas = corpo["metricas"]
    assert set(metricas) == CAMPOS_METRICAS
    for campo in ("dice", "iou", "precisao", "revocacao"):
        assert isinstance(metricas[campo], float)
        assert 0.0 <= metricas[campo] <= 1.0
    volume = metricas["volume_pulmonar_ml"]
    assert volume is None or (isinstance(volume, float) and volume > 0)
    assert isinstance(metricas["tempo_ms"], int) and metricas["tempo_ms"] >= 0

    # As imagens: PNG válido, com as dimensões anunciadas, na mesma grade.
    imagem_fatia = cliente.get(fatia["url"])
    imagem_mascara = cliente.get(mascara["url"])
    for resposta in (imagem_fatia, imagem_mascara):
        assert resposta.status_code == 200
        assert resposta.headers["content-type"] == "image/png"
        assert resposta.headers["etag"]
    pixels_fatia = decodificar_png(imagem_fatia.content)
    pixels_mascara = decodificar_png(imagem_mascara.content)
    assert pixels_fatia.shape == (fatia["altura"], fatia["largura"])
    assert pixels_mascara.shape == (mascara["altura"], mascara["largura"])
    assert pixels_mascara.shape == pixels_fatia.shape
    assert set(np.unique(pixels_mascara)) <= {0, 255}
