"""Leitura dos exames direto do HD (DICOM do LIDC-IDRI + mascara do LUNA16).

Estrutura esperada em API_RAIZ_DATASET (padrao D:\\DATASET_LUNA16):

    metadata/metadata.csv                 PatientID -> SeriesInstanceUID -> pasta
    lidc_idri/<PatientID>/<estudo>/<serie>/*.dcm
    mascaras/seg-lungs-LUNA16/seg-lungs-LUNA16/<SeriesInstanceUID>.mhd
    divisao_exames/split_novo.csv         treino / validacao / teste_interno / anterior

Um paciente do LIDC pode ter mais de uma serie (ex.: CT + raio-X); vale a
que tem mascara no LUNA16 (o mesmo criterio de api/preparar_treino.py).
"""
import csv
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import SimpleITK as sitk

RAIZ_PADRAO = r"D:\DATASET_LUNA16"


@dataclass
class Serie:
    paciente: str
    uid: str
    pasta_dicom: Path
    mascara: Path
    conjunto: str | None  # split_novo.csv; None se o exame nao estiver la


class FonteDicom:
    def __init__(self, raiz=RAIZ_PADRAO):
        self.raiz = Path(raiz)
        if not self.raiz.is_dir():
            raise RuntimeError(
                f"Pasta do dataset '{self.raiz}' não encontrada. O HD externo está "
                "conectado? Ajuste API_RAIZ_DATASET ou use API_MODO=fixo."
            )
        self.pasta_mascaras = self.raiz / "mascaras" / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
        self.conjuntos = self._ler_split()
        self.por_uid = self._ler_metadata()  # SeriesInstanceUID -> Serie
        # paciente -> 1a serie (so o LIDC-IDRI-0332 tem duas com mascara; a
        # outra continua acessivel pelo SeriesInstanceUID)
        self.series = {}
        for serie in self.por_uid.values():
            self.series.setdefault(serie.paciente, serie)

    def _ler_split(self):
        caminho = self.raiz / "divisao_exames" / "split_novo.csv"
        if not caminho.exists():
            return {}
        with caminho.open(newline="", encoding="utf-8") as f:
            return {linha["seriesuid"]: linha["split"] for linha in csv.DictReader(f)}

    def _ler_metadata(self):
        caminho = self.raiz / "metadata" / "metadata.csv"
        if not caminho.exists():
            raise RuntimeError(f"'{caminho}' não encontrado: é ele que liga paciente, série e pasta DICOM.")
        series = {}
        with caminho.open(newline="", encoding="utf-8") as f:
            for linha in csv.DictReader(f):
                uid = linha["SeriesInstanceUID"]
                mascara = self.pasta_mascaras / f"{uid}.mhd"
                if not mascara.exists():
                    continue
                # caminho gravado no download (D:\Projeto_Ronaldo\lidc_idri\<pac>\<est>\<ser>):
                # as 3 ultimas partes valem dentro de <raiz>\lidc_idri
                partes = PureWindowsPath(linha["S5cmdManifestPath"]).parts[-3:]
                series[uid] = Serie(
                    paciente=linha["PatientID"],
                    uid=uid,
                    pasta_dicom=self.raiz.joinpath("lidc_idri", *partes),
                    mascara=mascara,
                    conjunto=self.conjuntos.get(uid),
                )
        return series

    def pacientes_do_conjunto(self, conjunto):
        return sorted(p for p, s in self.series.items() if s.conjunto == conjunto)

    def serie(self, exame_id):
        """(Serie, None) ou (None, motivo). exame_id: PatientID ou SeriesInstanceUID."""
        serie = self.series.get(exame_id) or self.por_uid.get(exame_id)
        if serie is None:
            return None, "paciente sem série com máscara do LUNA16 no metadata.csv"
        if not serie.pasta_dicom.is_dir():
            return None, f"pasta DICOM '{serie.pasta_dicom}' não existe"
        return serie, None

    def ler_ct(self, serie):
        """Le a serie DICOM (HU, RescaleSlope/Intercept aplicados pelo GDCM)."""
        leitor = sitk.ImageSeriesReader()
        arquivos = leitor.GetGDCMSeriesFileNames(str(serie.pasta_dicom), serie.uid)
        if not arquivos:
            raise RuntimeError(f"nenhum DICOM da série {serie.uid} em '{serie.pasta_dicom}'")
        leitor.SetFileNames(arquivos)
        return leitor.Execute()

    def ler_mascara(self, serie):
        return sitk.ReadImage(str(serie.mascara))
