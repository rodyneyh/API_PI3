"""Pre-processamento usado no treino e na avaliacao da U-Net (e, na API, nos
tres metodos), copiado de D:\\DATASET_LUNA16\\codigo_colab_888 para que a
inferencia veja exatamente o mesmo tipo de entrada do treino:

  1. clean_hu: troca o padding (-2048) por ar e faz clip em -1000..400 HU
  2. denoise: filtro gaussiano 3D leve (sigma 0.5 voxel)
  3. resample_isotropic: reamostra para 1 mm x 1 mm x 1 mm

O volume continua em HU (o limiar do baseline, -600 HU, depende disso).
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

PADDING_HU = -2048
CLIP_MIN_HU = -1000  # ar puro
CLIP_MAX_HU = 400  # acima disso e osso/contraste, irrelevante p/ pulmao
ESPACAMENTO_ALVO = (1.0, 1.0, 1.0)  # mm (x, y, z)
ROTULOS_PULMAO = (3, 4)  # seg-lungs-LUNA16: 3 = pulmao dir., 4 = esq., 5 = traqueia (fora)


def clean_hu(volume_hu: np.ndarray) -> np.ndarray:
    """Substitui o valor de padding por ar (-1000 HU) e faz o clip para a
    faixa relevante ao pulmao. Mantem unidades de HU reais."""
    cleaned = volume_hu.copy()
    cleaned[cleaned <= PADDING_HU + 1] = CLIP_MIN_HU
    return np.clip(cleaned, CLIP_MIN_HU, CLIP_MAX_HU)


def denoise(volume_hu: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Filtro gaussiano 3D leve; sigma pequeno preserva a borda do pulmao."""
    return ndimage.gaussian_filter(volume_hu, sigma=sigma)


def resample_isotropic(sitk_image: sitk.Image, target_spacing=ESPACAMENTO_ALVO, is_mask: bool = False) -> sitk.Image:
    """Reamostra para espacamento isotropico: linear para CT, vizinho mais
    proximo para mascara."""
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0 if is_mask else CLIP_MIN_HU)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    return resampler.Execute(sitk_image)


def _para_sitk(array: np.ndarray, referencia: sitk.Image, dtype: str) -> sitk.Image:
    img = sitk.GetImageFromArray(array.astype(dtype))
    img.CopyInformation(referencia)
    return img


def preparar_ct(ct: sitk.Image) -> np.ndarray:
    """CT em HU (SimpleITK, lido do DICOM ou do .mhd) -> volume (z, y, x) int16
    limpo, suavizado e a 1 mm. Mesma cadeia de pipeline.run_pipeline_for_uid.

    O cast para int16 antes do filtro reproduz o treino: o .mhd do LUNA16 e
    int16, e o gaussian_filter devolve o mesmo tipo da entrada.
    """
    array = sitk.GetArrayFromImage(ct).astype(np.int16)
    suavizado = denoise(clean_hu(array), sigma=0.5)
    return sitk.GetArrayFromImage(
        resample_isotropic(_para_sitk(suavizado, ct, "int16"), is_mask=False)
    )


def preparar_referencia(mascara: sitk.Image, ct: sitk.Image) -> np.ndarray:
    """Mascara seg-lungs-LUNA16 -> parenquima (rotulos 3 e 4) booleano a 1 mm,
    na mesma grade de preparar_ct(ct).

    Se a grade da mascara nao for identica a do CT, ela e reamostrada para a
    grade do CT pelas coordenadas fisicas antes (vizinho mais proximo).
    """
    mesma_grade = (
        mascara.GetSize() == ct.GetSize()
        and np.allclose(mascara.GetSpacing(), ct.GetSpacing(), atol=1e-3)
        and np.allclose(mascara.GetOrigin(), ct.GetOrigin(), atol=1e-2)
        and np.allclose(mascara.GetDirection(), ct.GetDirection(), atol=1e-4)
    )
    if not mesma_grade:
        mascara = sitk.Resample(mascara, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    pulmao = np.isin(sitk.GetArrayFromImage(mascara), ROTULOS_PULMAO)
    return sitk.GetArrayFromImage(
        resample_isotropic(_para_sitk(pulmao, ct, "uint8"), is_mask=True)
    ).astype(bool)
