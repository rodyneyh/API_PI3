# API de segmentação pulmonar

Ponte entre os métodos de segmentação e a tela. Tem dois modos, escolhidos pela
variável `API_MODO`:

| modo | o que devolve | precisa de |
|---|---|---|
| `fixo` (padrão) | valores inventados de `api/dados_exemplo/` | só FastAPI |
| `real` | baseline, region growing e U-Net rodando de verdade | `data/luna16/treino/`, pesos da U-Net, torch, SimpleITK, scipy |

O formato das respostas é o mesmo nos dois modos; só o campo `modo` muda
(`"valores_fixos"` ou `"modelo_real"`). Assim a tela roda sem os dados do LUNA16.

## Como subir

Sempre a partir da **raiz do repositório** (caminho relativo, não absoluto:
caminho absoluto quebra a leitura do SimpleITK).

```bash
pip install -r api/requirements.txt
```

**Modo fixo** (padrão):

```bash
uvicorn api.main:app --reload
```

**Modo real**:

```bash
# bash
API_MODO=real uvicorn api.main:app --reload
# PowerShell
$env:API_MODO="real"; uvicorn api.main:app --reload
```

A API fica em http://127.0.0.1:8000, com a documentação interativa em
http://127.0.0.1:8000/docs.

### Configuração do modo real

| variável | padrão | uso |
|---|---|---|
| `API_EXAMES` | `LIDC-IDRI-0001,LIDC-IDRI-0002,LIDC-IDRI-0003` | exames de demonstração (separados por vírgula) |
| `API_PASTA_DADOS` | `data/luna16/treino` | pasta com `<exame>/imagem.npy` e `mascara.npy` |
| `API_PESOS_UNET` | `data/luna16/unet_baseline.pt` | pesos da U-Net |

- Na subida a API carrega a U-Net e os volumes dos exames de `API_EXAMES`
  **uma vez só**. Cada exame ocupa ~200 MB de RAM, por isso a lista é curta.
- Cada par (exame, método) é calculado na primeira chamada e fica em **cache**.
  Na CPU, a primeira chamada da U-Net leva ~1 a 2 min por exame, o region
  growing ~6 a 13 s e o baseline < 1 s; as próximas respondem em ~20 ms.
- As métricas comparam com a máscara de referência do LUNA16 (`mascara.npy`).
- O volume em mL usa `<exame>/espacamento.json`. Os `.npy` não guardam o
  espaçamento do voxel; para gerá-lo a partir das máscaras `.mhd` do LUNA16:
  `python api/gerar_espacamento.py`. Sem esse arquivo, `volume_pulmonar_ml` sai `null`.
- A inferência está isolada em `api/servicos/real.py`, na função `inferir()`,
  o único ponto que chama os métodos (`api/segmentation.py` e `src/luna16/unet.py`).
  Quando existir `src/luna16/inference.py`, basta trocar essa função por um import.

## Como rodar os testes

```bash
python -m pytest                                              # modo fixo
API_MODO=real API_EXAMES=LIDC-IDRI-0001 python -m pytest      # contrato contra os modelos reais
```

- `api/tests/test_contrato.py`: confere o formato de cada resposta campo a
  campo, sem olhar os valores das métricas. Roda no modo de `API_MODO`.
- `api/tests/test_modo_real.py`: roda os três métodos de verdade (com os pesos
  reais) sobre um volume sintético pequeno; testa o erro 409 e o cache. Não
  precisa do LUNA16; pula se faltarem os pesos, torch ou SimpleITK.

O `pytest.ini` aponta só para `api/tests/`, porque os `api/test_*.py` antigos
são scripts soltos que dependem dos dados locais.

## Contrato (PROPOSTA, para levar à reunião)

> Ainda não existe `docs/contrato_api.md` neste repositório. Este é o formato
> que a API implementa hoje; se o grupo decidir outro, basta ajustar
> `api/main.py` e `api/tests/test_contrato.py`.

1. **Imagens como PNG binário, em rotas próprias.** O JSON da segmentação traz
   só a `url` de cada imagem. A fatia é um PNG cinza de 8 bits (janela de −1000
   a 400 HU); a máscara é um PNG cinza com 0 = fundo e 255 = pulmão, **na mesma
   grade da fatia**, para a tela sobrepor. A tela usa direto em
   `<img src="http://127.0.0.1:8000/exames/.../fatia.png">`.
2. **A fatia vem já em janela de exibição (8 bits), não em HU.**
3. **Uma fatia por exame** (a `fatia_exemplo`; no modo real, a de maior área de
   pulmão). As métricas valem para o volume inteiro.
4. **Métricas:** `dice`, `iou`, `precisao`, `revocacao` (0 a 1),
   `volume_pulmonar_ml` (float ou `null`) e `tempo_ms`.
5. **Erros** sempre no formato `{"detail": {"codigo": "...", "mensagem": "..."}}`.

### `GET /exames`

Lista os exames de demonstração.

```bash
curl http://127.0.0.1:8000/exames
```

```json
{
  "exames": [
    {"id": "demo-001", "nome": "Exame de demonstração 1",
     "num_fatias": 240, "fatia_exemplo": 120, "baixado": true}
  ]
}
```

| campo | tipo | significado |
|---|---|---|
| `id` | string | identificador usado nas outras rotas |
| `nome` | string | texto para exibir na tela |
| `num_fatias` | int | total de fatias do exame (0 se não baixado) |
| `fatia_exemplo` | int | índice da fatia devolvida na segmentação |
| `baixado` | bool | `false` quando o exame não está na máquina (só no modo real) |

### `GET /exames/{exame_id}/segmentacao/{metodo}`

`metodo` é `baseline`, `region_growing` ou `unet`.

```bash
curl http://127.0.0.1:8000/exames/demo-001/segmentacao/unet
```

```json
{
  "exame_id": "demo-001",
  "metodo": "unet",
  "modo": "valores_fixos",
  "fatia":   {"indice": 120, "largura": 128, "altura": 128,
              "url": "/exames/demo-001/fatia.png"},
  "mascara": {"largura": 128, "altura": 128,
              "url": "/exames/demo-001/segmentacao/unet/mascara.png"},
  "metricas": {"dice": 0.96, "iou": 0.92, "precisao": 0.97,
               "revocacao": 0.95, "volume_pulmonar_ml": 4420.0, "tempo_ms": 380}
}
```

### `GET /exames/{exame_id}/fatia.png`

```bash
curl -o fatia.png http://127.0.0.1:8000/exames/demo-001/fatia.png
```

### `GET /exames/{exame_id}/segmentacao/{metodo}/mascara.png`

```bash
curl -o mascara.png http://127.0.0.1:8000/exames/demo-001/segmentacao/unet/mascara.png
```

### `GET /saude`

```bash
curl http://127.0.0.1:8000/saude
# {"status":"ok","modo":"valores_fixos"}
```

### Erros

```bash
curl -i http://127.0.0.1:8000/exames/xyz/segmentacao/unet               # 404
# {"detail":{"codigo":"exame_inexistente","mensagem":"Exame 'xyz' não existe. ..."}}

curl -i http://127.0.0.1:8000/exames/demo-001/segmentacao/foo           # 400
# {"detail":{"codigo":"metodo_invalido","mensagem":"Método 'foo' inválido. Use um destes: baseline, region_growing, unet."}}

# modo real, exame listado em API_EXAMES sem imagem.npy/mascara.npy        # 409
curl -i http://127.0.0.1:8000/exames/LIDC-IDRI-9999/segmentacao/unet
# {"detail":{"codigo":"exame_nao_baixado","mensagem":"Exame 'LIDC-IDRI-9999' está na lista, mas não foi baixado: ..."}}
```

## Dados de exemplo (modo fixo)

`api/dados_exemplo/` tem 2 exames inventados (um formato de torso com dois
pulmões desenhados à mão, não é tomografia real): `fatia.png`, uma
`mascara_<metodo>.png` por método e `metricas.json`. As três máscaras são
diferentes de propósito para a tela mostrar diferença entre os métodos. Para
regenerar: `python api/gerar_dados_exemplo.py`.

## Arquivos

| arquivo | papel |
|---|---|
| `api/main.py` | rotas e escolha do modo |
| `api/erros.py` | erros no formato do contrato |
| `api/png.py` | codifica PNG cinza sem Pillow |
| `api/servicos/fixo.py` | modo fixo |
| `api/servicos/real.py` | modo real: `inferir()`, cache, U-Net carregada uma vez |
| `api/gerar_espacamento.py` | grava o espaçamento do voxel de cada exame |
