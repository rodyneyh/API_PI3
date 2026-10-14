from pathlib import Path
import SimpleITK as sitk
import numpy as np

from segmentation import carregar_dicom


RAIZ_DICOM = Path(r"D:\DATASET_LUNA16\lidc_idri")

RAIZ_MASCARAS = Path(
    r"D:\DATASET_LUNA16\mascaras\seg-lungs-LUNA16\seg-lungs-LUNA16"
)

SAIDA = Path("data/luna16/treino")

QUANTIDADE_EXAMES = 222


def encontrar_serie_com_mascara(paciente):
    for arquivo in paciente.rglob("*.dcm"):

        pasta = arquivo.parent

        try:
            reader = sitk.ImageSeriesReader()

            series_ids = reader.GetGDCMSeriesIDs(
                str(pasta)
            )

            for series_id in series_ids:

                arquivos = reader.GetGDCMSeriesFileNames(
                    str(pasta),
                    series_id
                )

                caminho_mascara = (
                    RAIZ_MASCARAS / f"{series_id}.mhd"
                )

                if caminho_mascara.exists():

                    return (
                        pasta,
                        series_id,
                        caminho_mascara,
                        len(arquivos)
                    )

        except Exception as e:
            print(f"Erro em {pasta}: {e}")

    return None


# ============================================================
# LOCALIZAR EXAMES
# ============================================================

pacientes = sorted(
    [p for p in RAIZ_DICOM.iterdir() if p.is_dir()]
)


print("=" * 70)
print("PREPARAÇÃO DOS DADOS PARA TREINAMENTO")
print("=" * 70)

print(f"\nExames desejados: {QUANTIDADE_EXAMES}")
print(f"Pacientes disponíveis: {len(pacientes)}")


# ============================================================
# PROCESSAR EXAMES
# ============================================================

processados = 0

for paciente in pacientes:

    if processados >= QUANTIDADE_EXAMES:
        break

    print(
        f"\n[{processados + 1}/{QUANTIDADE_EXAMES}] "
        f"{paciente.name}"
    )

    resultado = encontrar_serie_com_mascara(paciente)

    if resultado is None:
        print("  Sem máscara. Pulando...")
        continue

    (
        pasta_dicom,
        series_id,
        caminho_mascara,
        quantidade_dicom
    ) = resultado

    print(f"  DICOMs: {quantidade_dicom}")
    print(f"  Série: {series_id}")

    # ========================================================
    # CARREGAR CT
    # ========================================================

    volume, spacing = carregar_dicom(
        str(pasta_dicom)
    )

    # ========================================================
    # CARREGAR MÁSCARA
    # ========================================================

    mascara_sitk = sitk.ReadImage(
        str(caminho_mascara)
    )

    mascara = sitk.GetArrayFromImage(
        mascara_sitk
    )

    mascara = mascara > 0

    # ========================================================
    # VERIFICAR DIMENSÕES
    # ========================================================

    if volume.shape != mascara.shape:

        print("  ERRO: dimensões diferentes!")
        print("  CT:", volume.shape)
        print("  Máscara:", mascara.shape)

        continue

    # ========================================================
    # CRIAR PASTA DE SAÍDA
    # ========================================================

    pasta_saida = SAIDA / paciente.name

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SALVAR
    # ========================================================

    np.save(
        pasta_saida / "imagem.npy",
        volume
    )

    np.save(
        pasta_saida / "mascara.npy",
        mascara
    )

    processados += 1

    print("  ✓ CT salvo")
    print("  ✓ Máscara salva")
    print(
        f"  ✓ Voxels da máscara: {mascara.sum()}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PREPARAÇÃO FINALIZADA")
print("=" * 70)

print(f"\nExames processados: {processados}")
print(f"Exames desejados: {QUANTIDADE_EXAMES}")

print("\nDados salvos em:")
print(SAIDA)