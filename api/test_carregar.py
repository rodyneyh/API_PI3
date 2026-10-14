from segmentation import carregar_dicom

PASTA_DICOM = r"D:\DATASET_LUNA16\lidc_idri\LIDC-IDRI-0001\30178\03192"

volume, spacing = carregar_dicom(PASTA_DICOM)

print("Volume carregado!")
print("Dimensões:", volume.shape)
print("Spacing:", spacing)
print("Tipo:", volume.dtype)
print("Mínimo:", volume.min())
print("Máximo:", volume.max())