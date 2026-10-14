from segmentation import carregar_dicom, region_growing


PASTA_DICOM = r"D:\DATASET_LUNA16\lidc_idri\LIDC-IDRI-0001\30178\03192"


volume, spacing = carregar_dicom(
    PASTA_DICOM
)


# Volume do voxel em mm³
voxel_volume = (
    spacing[0]
    * spacing[1]
    * spacing[2]
)


print("Volume carregado!")
print("Dimensões:", volume.shape)
print("Voxel:", voxel_volume)


mascara, sementes = region_growing(
    volume,
    voxel_volume,
    retornar_sementes=True
)


print()
print("Region Growing executado!")
print("Dimensões da máscara:", mascara.shape)
print("Tipo:", mascara.dtype)
print("Voxels segmentados:", mascara.sum())
print("Quantidade de sementes:", len(sementes))
print("Sementes:", sementes)