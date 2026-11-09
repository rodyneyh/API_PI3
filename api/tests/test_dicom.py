"""Leitura dos exames no HD (DICOM + mascara do LUNA16). Pula sem o HD."""
from pathlib import Path

import pytest

sitk = pytest.importorskip("SimpleITK")

from api.servicos.dicom import RAIZ_PADRAO, FonteDicom  # noqa: E402

pytestmark = pytest.mark.skipif(not Path(RAIZ_PADRAO).is_dir(), reason=f"HD não conectado ({RAIZ_PADRAO})")


@pytest.fixture(scope="module")
def fonte():
    return FonteDicom(RAIZ_PADRAO)


def test_series_com_mascara(fonte):
    # 888 series do LUNA16 em 887 pacientes (o LIDC-IDRI-0332 tem duas)
    assert len(fonte.por_uid) == 888
    assert len(fonte.series) == 887
    assert all(s.pasta_dicom.is_dir() for s in fonte.por_uid.values())
    assert len(fonte.pacientes_do_conjunto("teste_interno")) == 107


def test_segunda_serie_pelo_uid(fonte):
    uids = [u for u, s in fonte.por_uid.items() if s.paciente == "LIDC-IDRI-0332"]
    assert len(uids) == 2
    for uid in uids:
        serie, motivo = fonte.serie(uid)
        assert motivo is None and serie.uid == uid


def test_dicom_na_mesma_grade_da_mascara(fonte):
    serie, motivo = fonte.serie("LIDC-IDRI-0013")
    assert motivo is None and serie.conjunto == "teste_interno"
    leitor = sitk.ImageFileReader()
    leitor.SetFileName(str(serie.mascara))
    leitor.ReadImageInformation()
    ct = fonte.ler_ct(serie)
    assert ct.GetSize() == leitor.GetSize()
    assert ct.GetSpacing() == pytest.approx(leitor.GetSpacing(), abs=1e-3)


def test_paciente_sem_serie(fonte):
    serie, motivo = fonte.serie("LIDC-IDRI-9999")
    assert serie is None and "metadata.csv" in motivo


def test_hd_ausente_da_erro_claro(tmp_path):
    with pytest.raises(RuntimeError, match="HD externo"):
        FonteDicom(tmp_path / "nao_existe")
