import SimpleITK as sitk
import numpy as np
from scipy import ndimage as ndi


def carregar_dicom(pasta):
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(pasta)

    if not series_ids:
        raise ValueError(
            f"Nenhuma série DICOM encontrada em: {pasta}"
        )

    series_id = series_ids[0]

    arquivos = reader.GetGDCMSeriesFileNames(
        pasta,
        series_id
    )

    if not arquivos:
        raise ValueError(
            f"Nenhum arquivo DICOM encontrado em: {pasta}"
        )

    reader.SetFileNames(arquivos)

    imagem = reader.Execute()

    volume = sitk.GetArrayFromImage(imagem)
    spacing = imagem.GetSpacing()

    return volume, spacing




def baseline(volume, limiar=-600, raio=2):
    """
    Threshold de -600 HU + erosão morfológica.
    """

    mascara = volume < limiar

    estrutura = ndi.generate_binary_structure(3, 1)

    if raio:
        mascara = ndi.binary_erosion(
            mascara,
            estrutura,
            iterations=raio
        )

    return mascara



LIMIAR_TECIDO = -320.0
LIMIAR_AR = -400.0
LIMIAR_INFERIOR = -1100.0
RAIO_FECHAMENTO = 3
VOLUME_MINIMO_ML = 50.0

VIZ6 = ndi.generate_binary_structure(3, 1)




def preencher_cavidades(mascara):
    """
    Preenche buracos fatia por fatia.
    """

    return np.stack([
        ndi.binary_fill_holes(f)
        for f in mascara
    ])


def maior_componente(mascara, n=1):
    """
    Mantém os maiores componentes conectados.
    """

    if not mascara.any():
        return mascara

    rotulos, k = ndi.label(mascara, VIZ6)

    tamanhos = ndi.sum_labels(
        mascara,
        rotulos,
        index=np.arange(1, k + 1)
    )

    return np.isin(
        rotulos,
        np.argsort(tamanhos)[::-1][:n] + 1
    )


def remover_pequenos(mascara, vox, min_ml):
    """
    Remove componentes com volume menor que min_ml.
    """

    if not mascara.any():
        return mascara

    rotulos, k = ndi.label(mascara, VIZ6)

    tamanhos = ndi.sum_labels(
        mascara,
        rotulos,
        index=np.arange(1, k + 1)
    )

    manter = np.where(
        tamanhos * vox / 1000 >= min_ml
    )[0] + 1

    return np.isin(rotulos, manter)


def mascara_corpo(hu):
    """
    Cria uma máscara aproximada do corpo do paciente.
    """

    corpo = preencher_cavidades(
        hu > LIMIAR_TECIDO
    )

    corpo = maior_componente(corpo)

    corpo = ndi.binary_closing(
        corpo,
        VIZ6,
        iterations=2
    )

    return preencher_cavidades(corpo)



def achar_sementes(ar_interno):
    """
    Encontra pontos internos de cada pulmão
    em três alturas do volume.
    """

    nz = ar_interno.shape[0]

    areas = ar_interno.reshape(
        nz,
        -1
    ).sum(axis=1)

    if areas.max() == 0:
        return []

    sementes = []

    alturas = sorted({
        int(np.argmax(areas)),
        int(nz * 0.35),
        int(nz * 0.65)
    })

    for z in alturas:

        fatia = ar_interno[z]

        if not fatia.any():
            continue

        meio = int(
            round(
                ndi.center_of_mass(fatia)[1]
            )
        )

        for x0, x1 in (
            (0, meio),
            (meio, fatia.shape[1])
        ):

            metade = np.zeros_like(fatia)

            metade[:, x0:x1] = fatia[:, x0:x1]

            if not metade.any():
                continue

            dist = ndi.distance_transform_edt(
                metade
            )

            if dist.max() < 3:
                continue

            y, x = np.unravel_index(
                np.argmax(dist),
                dist.shape
            )

            semente = (
                int(x),
                int(y),
                int(z)
            )

            if semente not in sementes:
                sementes.append(semente)

    return sementes



def region_growing(
    hu,
    vox,
    confinar=True,
    preencher=True,
    fechar=True,
    retornar_sementes=False
):
    """
    Crescimento de região 3D + reconstrução morfológica.
    """

    corpo = mascara_corpo(hu)

    ar_interno = remover_pequenos(
        (hu < LIMIAR_AR) & corpo,
        vox,
        5.0
    )

    sementes = achar_sementes(
        ar_interno
    )

    if not sementes:

        vazio = np.zeros_like(
            hu,
            dtype=bool
        )

        return (
            (vazio, [])
            if retornar_sementes
            else vazio
        )

 
    entrada = (
        np.where(
            corpo,
            hu,
            1000.0
        ).astype(np.float32)
        if confinar
        else hu.astype(np.float32)
    )

    imagem_sitk = sitk.GetImageFromArray(
        entrada
    )

    crescida = sitk.ConnectedThreshold(
        imagem_sitk,
        seedList=sementes,
        lower=LIMIAR_INFERIOR,
        upper=LIMIAR_AR
    )

    mascara = sitk.GetArrayFromImage(
        crescida
    ).astype(bool)

    if preencher:
        mascara = preencher_cavidades(
            mascara
        )

    if fechar:

        mascara = ndi.binary_closing(
            mascara,
            VIZ6,
            iterations=RAIO_FECHAMENTO
        )

        if preencher:
            mascara = preencher_cavidades(
                mascara
            )

    if confinar:
        mascara &= corpo

    mascara = remover_pequenos(
        mascara,
        vox,
        VOLUME_MINIMO_ML
    )

    if retornar_sementes:
        return mascara, sementes

    return mascara