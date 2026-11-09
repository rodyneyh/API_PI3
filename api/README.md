# API de segmentação pulmonar

Ponte entre os métodos de segmentação e a tela. Tem dois modos, escolhidos pela
variável `API_MODO`:

| modo | o que devolve | precisa de |
|---|---|---|
| `fixo` (padrão) | valores inventados de `api/dados_exemplo/` | só FastAPI |
| `real` | baseline, region growing e U-Net rodando de verdade sobre os DICOM | HD com `DATASET_LUNA16`, torch, SimpleITK, scipy, scikit-image |

O formato das respostas é o mesmo nos dois modos; só o campo `modo` muda
(`"valores_fixos"` ou `"modelo_real"`). Assim a tela roda sem o HD.

## Como subir

Sempre a partir da **raiz do repositório**.

```bash
pip install -r api/requirements.txt
```

**Modo fixo** (padrão):

```bash
uvicorn api.main:app --reload
```

**Modo real** (HD externo conectado):

```bash
# bash
API_MODO=real uvicorn api.main:app --reload
# PowerShell
$env:API_MODO="real"; uvicorn api.main:app --reload
```

A API fica em http://127.0.0.1:8000, com a documentação interativa em
http://127.0.0.1:8000/docs.

### Modo real: de onde vêm os dados

Tudo é lido do HD, sem cópia para o repositório:

| o quê | onde (dentro de `API_RAIZ_DATASET`) |
|---|---|
| tomografia (DICOM do LIDC-IDRI) | `lidc_idri/<paciente>/<estudo>/<série>/*.dcm` |
| paciente → série → pasta | `metadata/metadata.csv` |
| máscara de referência | `mascaras/seg-lungs-LUNA16/seg-lungs-LUNA16/<SeriesInstanceUID>.mhd` |
| conjunto de cada exame | `divisao_exames/split_novo.csv` (treino / validação / teste) |
| pesos da U-Net | `modelos/unet_seed123/unet_novo_final.pt` (no repositório, copiado de `treino_seed123`) |

Cada exame passa pelo **mesmo pipeline usado no treino e na avaliação da
U-Net** (`pipeline.run_pipeline_for_uid`, em `codigo_colab_888`):

1. lê a série DICOM (o paciente pode ter várias; vale a que tem máscara no LUNA16);
2. `src/luna16/preprocessing.py`: troca o padding por ar, clip −1000..400 HU,
   filtro gaussiano σ = 0,5 e reamostragem para **1 mm isotrópico**;
3. referência = **pulmões direito e esquerdo** (rótulos 3 e 4 da máscara; a
   traqueia, rótulo 5, fica de fora), na mesma grade de 1 mm;
4. `src/luna16/inference.py`, `inferir()`: roda o método pedido sobre esse
   volume. Os três métodos recebem a mesma entrada, então a comparação é justa.

A U-Net é a `SmallUNet2D` (base 32 canais, fatias 128×128 em [−1, 1]) treinada
com os 497 exames de treino. Conferido: nos exames LIDC-IDRI-0013, 0022 e 0024
a API dá **exatamente** o Dice e o número de voxels de
`treino_seed123/resultados/avaliacao_novo_final_teste_interno.csv`.

| variável | padrão | uso |
|---|---|---|
| `API_RAIZ_DATASET` | `D:\DATASET_LUNA16` | pasta do dataset no HD |
| `API_PESOS_UNET` | `modelos/unet_seed123/unet_novo_final.pt` | checkpoint da U-Net (qualquer `unet_novo_*.pt`) |
| `API_EXAMES` | os 3 primeiros do conjunto de **teste** (LIDC-IDRI-0013, 0022, 0024) | exames servidos, separados por vírgula: `PatientID` ou `SeriesInstanceUID` |

- Por padrão a API serve exames do **teste**, que a U-Net não viu no treino.
  O `nome` do exame diz o conjunto, por exemplo `"Exame LIDC-IDRI-0013 (teste)"`,
  e avisa `(usado no treino da U-Net)` se o exame estiver entre os de treino do checkpoint.
- Na subida, a API carrega a U-Net (na GPU, se houver) e faz uma passada de
  aquecimento; depois lê os exames de `API_EXAMES` **uma vez só** (~2 a 6 s
  cada, a partir do HD). Cada exame ocupa ~150 MB de RAM, por isso a lista é curta.
- Cada par (exame, método) é calculado na primeira chamada e fica em **cache**.
  Com GPU: U-Net ~1,5 a 4 s por exame, region growing ~5 a 13 s, baseline ~3 a 4 s.
  Depois disso, ~20 ms.
- `volume_pulmonar_ml` = voxels marcados × 1 mm³ (o volume está a 1 mm).
- Exame da lista que não está no HD aparece com `baixado: false` e responde 409.
  Se o HD não estiver conectado, a API não sobe (mensagem clara); use `API_MODO=fixo`.

#### Resultado nos 3 exames padrão (teste)

| exame | método | Dice | IoU | volume (mL) |
|---|---|---|---|---|
| LIDC-IDRI-0013 | baseline | 0,948 | 0,901 | 4 752 |
| | region growing | 0,982 | 0,964 | 5 086 |
| | U-Net | **0,983** | **0,966** | 5 054 |
| LIDC-IDRI-0022 | baseline | 0,928 | 0,866 | 4 991 |
| | region growing | 0,980 | 0,962 | 5 472 |
| | U-Net | **0,984** | **0,968** | 5 442 |
| LIDC-IDRI-0024 | baseline | 0,906 | 0,828 | 4 314 |
| | region growing | 0,974 | 0,948 | 4 582 |
| | U-Net | **0,980** | **0,961** | 4 544 |

Figura: `notebooks/resultados/resultado_LIDC-IDRI-0013.png`
(gerada por `python notebooks/resultados/resultado.py <saida.png>`).

**Baseline corrigido.** Hoje ele é: limiar −600 HU, só dentro do corpo
(`mascara_corpo()`, a mesma do region growing) e abertura morfológica. A versão
antiga (limiar + erosão) tinha dois problemas, medidos nesses 3 exames:

| versão | Dice | volume (mL) | problema |
|---|---|---|---|
| limiar + erosão (antiga) | 0,22 a 0,42 | 13 238 a 24 999 | marcava o **ar em volta do paciente** como pulmão |
| + só dentro do corpo | 0,80 a 0,86 | 3 432 a 3 933 | a erosão corta ~2 voxels da borda inteira (revocação ~0,72) |
| + abertura no lugar da erosão (atual) | 0,91 a 0,95 | 4 314 a 4 991 | — |

Ele continua abaixo dos outros dois porque os vasos (mais densos que −600 HU)
ficam como buracos na máscara. A versão antiga continua disponível para
comparação: `baseline(volume, confinar=False, morfologia="erosao")`.

## Como rodar os testes

```bash
python -m pytest                       # modo fixo (+ testes do modo real que não dependem de API_MODO)
API_MODO=real python -m pytest         # contrato também contra os exames reais do HD
```

- `api/tests/test_contrato.py`: confere o formato de cada resposta campo a
  campo, sem olhar os valores das métricas. Roda no modo de `API_MODO`.
- `api/tests/test_modo_real.py`: roda os três métodos de verdade (pesos reais,
  mesmo pré-processamento) sobre um exame sintético em memória; testa a
  referência sem traqueia, o erro 409 e o cache. Não precisa do HD, só dos pesos.
- `api/tests/test_dicom.py`: leitura no HD (888 séries, grade do DICOM igual à
  da máscara, paciente com duas séries). Pula sem o HD.

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

# modo real, exame listado em API_EXAMES que não está no HD                # 409
curl -i http://127.0.0.1:8000/exames/LIDC-IDRI-9999/segmentacao/unet
# {"detail":{"codigo":"exame_nao_baixado","mensagem":"Exame 'LIDC-IDRI-9999' está na lista, mas não está no HD: ..."}}
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
| `api/servicos/real.py` | modo real: exames carregados na subida, métricas, cache |
| `api/servicos/dicom.py` | acha e lê a série DICOM e a máscara do LUNA16 no HD |
| `api/segmentation.py` | baseline e region growing |
| `src/luna16/preprocessing.py` | pré-processamento do treino (HU, gaussiano, 1 mm) |
| `src/luna16/inference.py` | `inferir()`: único ponto que chama os três métodos; carrega a U-Net |
| `src/luna16/unet.py` | `SmallUNet2D` (pesos do HD) e a `UNet` antiga de `api/treinar_unet.py` |
