from pathlib import Path
import zipfile

ZIP = Path(r"D:\DATASET_LUNA16\mascaras\seg-lungs-LUNA16.zip")
DESTINO = Path(r"D:\DATASET_LUNA16\mascaras\seg-lungs-LUNA16")

if not ZIP.exists():
    print("ERRO: arquivo ZIP não encontrado!")
    print(ZIP)
    raise SystemExit

print("Arquivo encontrado:")
print(ZIP)
print()

print("Extraindo máscaras...")
print("Isso pode demorar um pouco.")

with zipfile.ZipFile(ZIP, "r") as arquivo_zip:
    arquivo_zip.extractall(DESTINO)

print()
print("Extração concluída!")
print("Máscaras em:")
print(DESTINO)