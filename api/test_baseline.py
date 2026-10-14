from segmentation import carregar_dicom, baseline

PASTA_DICOM = r"D:\DATASET_LUNA16\lidc_idri\LIDC-IDRI-0001\30178\03192"

volume, spacing = carregar_dicom(PASTA_DICOM)

mascara = baseline(volume)

print("Baseline executado!")
print("Dimensões da máscara:", mascara.shape)
print("Tipo:", mascara.dtype)
print("Voxels segmentados:", mascara.sum())