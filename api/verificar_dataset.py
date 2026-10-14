from pathlib import Path

RAIZ = Path(r"D:\DATASET_LUNA16\lidc_idri")

pacientes = sorted([
    p for p in RAIZ.iterdir()
    if p.is_dir()
])[:5]

contador = 0

for paciente in pacientes:

    for pasta in paciente.rglob("*"):

        if not pasta.is_dir():
            continue

        arquivos_dcm = list(pasta.glob("*.dcm"))

        if arquivos_dcm:
            contador += 1

            print(f"Paciente: {paciente.name}")
            print(f"Pasta DICOM: {pasta}")
            print(f"Arquivos: {len(arquivos_dcm)}")
            print("-" * 60)

print()
print(f"Total de séries encontradas nos 5 primeiros pacientes: {contador}")