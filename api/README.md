# API de segmentação pulmonar

FastAPI que serve a segmentação de exames por três métodos: `baseline`,
`region_growing` e `unet`. Tem dois modos:

| modo | o que devolve | precisa de |
|---|---|---|
| `fixo` (padrão) | dados inventados de `api/dados_exemplo/` | só `requirements.txt` |
| `real` | roda os métodos do projeto nos exames de `data/luna16/treino/` | `requirements-real.txt` + dados |

## Subir

Sempre da **raiz do repositório** (caminhos relativos, nunca absolutos):

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

Modo real:

```bash
pip install -r api/requirements-real.txt
API_MODO=real uvicorn api.main:app        # Windows: set API_MODO=real
```

Documentação interativa: http://127.0.0.1:8000/docs

## Endpoints

```bash
curl http://127.0.0.1:8000/saude
# {"status":"ok","modo":"valores_fixos"}

curl http://127.0.0.1:8000/exames
# {"exames":[{"id":"demo-001","nome":"...","num_fatias":240,"fatia_exemplo":120,"baixado":true}, ...]}

curl http://127.0.0.1:8000/exames/demo-001/segmentacao/unet
# {"exame_id":"demo-001","metodo":"unet","modo":"valores_fixos",
#  "fatia":   {"indice":120,"largura":128,"altura":128,"url":"/exames/demo-001/fatia.png"},
#  "mascara": {"largura":128,"altura":128,"url":"/exames/demo-001/segmentacao/unet/mascara.png"},
#  "metricas":{"dice":0.96,"iou":0.92,"precisao":0.97,"revocacao":0.95,
#              "volume_pulmonar_ml":4420.0,"tempo_ms":380}}

curl -o fatia.png   http://127.0.0.1:8000/exames/demo-001/fatia.png
curl -o mascara.png http://127.0.0.1:8000/exames/demo-001/segmentacao/unet/mascara.png
```

As imagens são PNG cinza de 8 bits, **binário** (sem base64), com `ETag`: quem já
tem a imagem recebe `304` sem corpo. A máscara é 0 (fundo) / 255 (pulmão) e está
na mesma grade da fatia.

## Erros

Sempre `{"detail": {"codigo": "...", "mensagem": "..."}}`:

| situação | status | `codigo` |
|---|---|---|
| exame não existe | 404 | `exame_inexistente` |
| método fora de baseline/region_growing/unet | 400 | `metodo_invalido` |
| exame existe, mas os dados não estão na máquina | 409 | `exame_nao_baixado` |
| pesos ou dependência do modelo faltando (modo real) | 503 | `modelo_indisponivel` |

## Configuração (variáveis de ambiente)

| variável | padrão |
|---|---|
| `API_MODO` | `fixo` (ou `real`) |
| `API_PASTA_TREINO` | `data/luna16/treino` |
| `API_PESOS_UNET` | `data/luna16/unet_baseline.pt` |
| `API_UNET_LOTE` | `8` (fatias por lote na U-Net) |
| `API_PASTA_DADOS` | `api/dados_exemplo` |
| `API_CORS_ORIGENS` | `*` |

## Modo real: o que reaproveita

Nada foi reescrito, a API só liga o que o projeto já tem:

- `api/segmentation.py`: `baseline()` e `region_growing()`;
- `src/luna16/unet.py`: `UNet`, com os pesos de `api/treinar_unet.py`;
- o pré-processamento de `api/preparar_fatias.py` (clip −1000..400 HU, /1400, 256×256);
- os exames salvos por `api/preparar_treino.py`: `data/luna16/treino/<id>/imagem.npy` e `mascara.npy`.

Como o Dice, o IoU, a precisão e a revocação vêm da máscara de referência do
próprio exame, no volume inteiro, eles são **reais** neste modo. Cada
(exame, método) é calculado uma vez e fica em memória; a U-Net é carregada uma
vez só.

Limites conhecidos:

- `volume_pulmonar_ml` vem `null`: os `.npy` não guardam o espaçamento do voxel.
- A primeira chamada de cada (exame, método) é lenta (volume inteiro); as seguintes são instantâneas.
- Todo o modo real está em `api/servicos/real.py`. Quando existir uma função única de
  inferência (`src/luna16/inference.py`), é só esse arquivo que muda.

## Testes

```bash
python -m pytest
```

`test_contrato.py` confere o formato de cada resposta campo a campo (modo fixo).
`test_modo_real.py` repete a conferência com exames sintéticos e não precisa do
LUNA16; os testes da U-Net só rodam se o `torch` estiver instalado. O
`pytest.ini` aponta só para `api/tests/`, porque os `api/test_*.py` antigos são
scripts soltos que dependem de dados locais.

## Dados de exemplo

`api/dados_exemplo/` tem dois exames inventados (não são tomografias) e um
terceiro, `demo-003`, sem arquivos, para exercitar o erro `exame_nao_baixado`.
Para regenerar: `python api/gerar_dados_exemplo.py`.
