"""PNG cinza de 8 bits, escrito e lido so com a stdlib (sem Pillow)."""
import struct
import zlib

import numpy as np


def codificar_png(matriz):
    """matriz 2D uint8 -> bytes de um PNG cinza de 8 bits."""
    matriz = np.ascontiguousarray(matriz, dtype=np.uint8)
    h, w = matriz.shape
    cru = b"".join(b"\x00" + matriz[y].tobytes() for y in range(h))

    def bloco(tipo, dados):
        corpo = tipo + dados
        return (struct.pack(">I", len(dados)) + corpo
                + struct.pack(">I", zlib.crc32(corpo) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + bloco(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + bloco(b"IDAT", zlib.compress(cru, 6))
            + bloco(b"IEND", b""))


def largura_altura(png):
    """Le largura e altura do cabecalho IHDR."""
    return struct.unpack(">II", png[16:24])
