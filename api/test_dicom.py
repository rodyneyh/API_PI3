import SimpleITK as sitk
import numpy as np

PASTA = r"D:\DATASET_LUNA16\lidc_idri\LIDC-IDRI-0001\30178\03192"

reader = sitk.ImageSeriesReader()

series_ids = reader.GetGDCMSeriesIDs(PASTA)

print("Séries encontradas:")
print(series_ids)

if not series_ids:
    raise RuntimeError("Nenhuma série DICOM encontrada.")

# Pega a primeira série
series_id = series_ids[0]

arquivos = reader.GetGDCMSeriesFileNames(PASTA, series_id)

print(f"\nQuantidade de arquivos DICOM: {len(arquivos)}")

reader.SetFileNames(arquivos)

imagem = reader.Execute()

volume = sitk.GetArrayFromImage(imagem)

print("\nVolume carregado!")
print("Dimensões:", volume.shape)
print("Spacing:", imagem.GetSpacing())
print("Tipo:", volume.dtype)
print("Valor mínimo:", volume.min())
print("Valor máximo:", volume.max())