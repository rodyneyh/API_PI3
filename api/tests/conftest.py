import os

# Sem API_MODO definido, os testes rodam no modo fixo (sem dados do LUNA16).
os.environ.setdefault("API_MODO", "fixo")
