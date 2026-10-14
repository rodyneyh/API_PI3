from pathlib import Path
import SimpleITK as sitk

from segmentation import carregar_dicom


PASTA_DICOM = Path(
    r"D:\DATASET_LUNA16\lidc_idri\LIDC-IDRI-0001\30178\03192"
)

PASTA_MASCARA = Path(
    r"D:\DATASET_LUNA16\mascaras\seg-lungs-LUNA16\seg-lungs-LUNA16"
    r"\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192.mhd"
)


print("=" * 60)
print("TESTE DICOM + MÁSCARA")
print("=" * 60)


# =========================
# CARREGAR DICOM
# =========================

volume, spacing = carregar_dicom(str(PASTA_DICOM))

print("\nDICOM:")
print("Dimensões:", volume.shape)
print("Spacing:", spacing)


# =========================
# CARREGAR MÁSCARA
# =========================

mascara_sitk = sitk.ReadImage(str(PASTA_MASCARA))

mascara = sitk.GetArrayFromImage(mascara_sitk)

print("\nMáscara:")
print("Dimensões:", mascara.shape)
print("Spacing:", mascara_sitk.GetSpacing())
print("Origem:", mascara_sitk.GetOrigin())
print("Direção:", mascara_sitk.GetDirection())
print("Voxels da máscara:", (mascara > 0).sum())


# =========================
# COMPARAÇÃO
# =========================

print("\nComparação:")

print("Tamanho DICOM:", volume.shape)
print("Tamanho máscara:", mascara.shape)

if volume.shape == mascara.shape:
    print(">>> MESMO TAMANHO")
else:
    print(">>> TAMANHOS DIFERENTES")

print("=" * 60)