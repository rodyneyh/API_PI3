"""PNG cinza de 8 bits, só com a biblioteca padrão (sem Pillow)."""
import hashlib
import struct
import zlib
from dataclasses import dataclass

import numpy as np

ASSINATURA = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class PNG:
    dados: bytes
    etag: str
    largura: int
    altura: int


def codificar_png(matriz: np.ndarray) -> bytes:
    """Codifica uma matriz 2D uint8 como PNG cinza."""
    matriz = np.ascontiguousarray(matriz, dtype=np.uint8)
    altura, largura = matriz.shape
    # Cada linha começa com o byte de filtro 0 (sem filtro).
    linhas = np.zeros((altura, largura + 1), dtype=np.uint8)
    linhas[:, 1:] = matriz
    cru = linhas.tobytes()

    def bloco(tipo: bytes, dados: bytes) -> bytes:
        corpo = tipo + dados
        return (struct.pack(">I", len(dados)) + corpo
                + struct.pack(">I", zlib.crc32(corpo) & 0xFFFFFFFF))

    return (ASSINATURA
            + bloco(b"IHDR", struct.pack(">IIBBBBB", largura, altura, 8, 0, 0, 0, 0))
            + bloco(b"IDAT", zlib.compress(cru, 6))
            + bloco(b"IEND", b""))


def montar_png(dados: bytes) -> PNG:
    """Lê as dimensões do cabeçalho e calcula o ETag de um PNG já pronto."""
    if not dados.startswith(ASSINATURA):
        raise ValueError("os dados não são um PNG")
    largura, altura = struct.unpack(">II", dados[16:24])
    etag = '"' + hashlib.sha1(dados).hexdigest()[:16] + '"'
    return PNG(dados, etag, largura, altura)
