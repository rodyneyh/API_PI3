from pathlib import Path
import SimpleITK as sitk


RAIZ_DICOM = Path(r"D:\DATASET_LUNA16\lidc_idri")

RAIZ_MASCARAS = Path(
    r"D:\DATASET_LUNA16\mascaras\seg-lungs-LUNA16\seg-lungs-LUNA16"
)


# Pega os 5 primeiros pacientes
pacientes = sorted(
    [p for p in RAIZ_DICOM.iterdir() if p.is_dir()]
)[:5]


print("=" * 70)
print("VERIFICAÇÃO DAS MÁSCARAS - 5 PRIMEIROS PACIENTES")
print("=" * 70)


# Cria um conjunto com os UIDs das máscaras disponíveis
mascaras = {
    arquivo.stem: arquivo
    for arquivo in RAIZ_MASCARAS.glob("*.mhd")
}


print(f"\nQuantidade de máscaras .mhd encontradas: {len(mascaras)}")


print("\nPacientes analisados:")

for paciente in pacientes:
    print(f"\n{paciente.name}")

    # Procura todas as pastas que possuem DICOM
    pastas_dicom = []

    for arquivo in paciente.rglob("*.dcm"):
        if arquivo.parent not in pastas_dicom:
            pastas_dicom.append(arquivo.parent)

    for pasta in sorted(pastas_dicom):

        try:
            reader = sitk.ImageSeriesReader()

            series_ids = reader.GetGDCMSeriesIDs(str(pasta))

            if not series_ids:
                continue

            for series_id in series_ids:

                arquivos = reader.GetGDCMSeriesFileNames(
                    str(pasta),
                    series_id
                )

                encontrou = series_id in mascaras

                print("\n  Pasta:", pasta)
                print("  Arquivos DICOM:", len(arquivos))
                print("  SeriesInstanceUID:", series_id)

                if encontrou:
                    print("  >>> MÁSCARA ENCONTRADA!")
                    print("  Máscara:", mascaras[series_id])
                else:
                    print("  >>> Sem máscara")

        except Exception as e:
            print("  ERRO:", e)


print("\n" + "=" * 70)
print("FIM")
print("=" * 70)