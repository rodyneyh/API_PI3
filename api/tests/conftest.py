import pytest
from fastapi.testclient import TestClient

from api.config import Config
from api.main import criar_app


@pytest.fixture(scope="session")
def cliente():
    """Cliente no modo fixo. Não depende de API_MODO nem de dados locais."""
    with TestClient(criar_app(Config())) as c:
        yield c
