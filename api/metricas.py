"""Métricas de segmentação contra a máscara de referência (volume 3D inteiro)."""
import numpy as np


def comparar(previsto: np.ndarray, referencia: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precisão e revocação entre duas máscaras booleanas.

    Convenções para casos vazios: se as duas máscaras são vazias, Dice e IoU
    valem 1.0; precisão e revocação valem 0.0 quando não há o que dividir.
    """
    previsto = previsto.astype(bool, copy=False)
    referencia = referencia.astype(bool, copy=False)

    acertos = int(np.count_nonzero(previsto & referencia))
    n_previsto = int(np.count_nonzero(previsto))
    n_referencia = int(np.count_nonzero(referencia))
    uniao = n_previsto + n_referencia - acertos

    return {
        "dice": 2 * acertos / (n_previsto + n_referencia) if n_previsto + n_referencia else 1.0,
        "iou": acertos / uniao if uniao else 1.0,
        "precisao": acertos / n_previsto if n_previsto else 0.0,
        "revocacao": acertos / n_referencia if n_referencia else 0.0,
    }
