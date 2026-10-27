# API de segmentação pulmonar

Ponte entre `src/luna16/` e a tela. **Etapa 1 (estado atual): todas as respostas
são valores fixos** guardados em `api/dados_exemplo/`, no formato do contrato
abaixo. Assim a tela pode ser desenvolvida e testada sem esperar pelos modelos.

## Como subir

Sempre a partir da **raiz do repositório** (caminho relativo, não absoluto):

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

A API fica em http://127.0.0.1:8000. A documentação interativa (gerada pelo
FastAPI) fica em http://127.0.0.1:8000/docs.

## Como rodar os testes

```bash
python -m pytest
```

Os testes (`api/tests/test_contrato.py`) conferem o formato de cada resposta
campo a campo, sem olhar os valores das métricas, e continuam valendo na etapa 2.
O `pytest.ini` aponta só para `api/tests/`, porque os `api/test_*.py` antigos
são scripts soltos que dependem dos dados locais.

## Contrato (PROPOSTA, para levar à reunião)

> O card "Fechar o contrato de formatos" ainda não tem um `docs/contrato_api.md`
> neste repositório. Este é o formato que a API implementa hoje; se o grupo
> decidir outro, basta ajustar `api/main.py` e `api/tests/test_contrato.py`.

Pontos para decidir na reunião:

1. **Imagem e máscara como PNG em base64.** Cada fatia é um PNG cinza de 8 bits;
   a máscara é um PNG cinza com 0 = fundo e 255 = pulmão, **na mesma grade da
   fatia** para a tela sobrepor. A tela usa direto em
   `<img src="data:image/png;base64,...">`. Em JSON puro, uma fatia 512×512
   seriam 262 mil números.
2. **A fatia vem já em janela de exibição (8 bits), não em HU.** Se o Ruan
   precisar dos valores em HU, o contrato muda.
3. **Uma fatia por exame** (a `fatia_exemplo`). Se a tela precisar navegar por
   fatias, o endpoint ganha um parâmetro `?fatia=N`.
4. **Métricas:** `dice`, `iou`, `precisao`, `revocacao` (0 a 1),
   `volume_pulmonar_ml` e `tempo_ms`. Dice/IoU só fazem sentido com máscara de
   referência; sem ela, a etapa 2 precisa decidir o que devolver.
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
| `num_fatias` | int | total de fatias do exame |
| `fatia_exemplo` | int | índice da fatia devolvida na segmentação |
| `baixado` | bool | `false` quando o exame não está na máquina (etapa 2) |

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
  "fatia":   {"indice": 120, "largura": 128, "altura": 128, "png_base64": "iVBORw0KGgo..."},
  "mascara": {"largura": 128, "altura": 128, "png_base64": "iVBORw0KGgo..."},
  "metricas": {"dice": 0.96, "iou": 0.92, "precisao": 0.97,
               "revocacao": 0.95, "volume_pulmonar_ml": 4420.0, "tempo_ms": 380}
}
```

`modo` é `"valores_fixos"` hoje e vira `"modelo_real"` na etapa 2, para a tela
saber o que está mostrando.

### `GET /saude`

```bash
curl http://127.0.0.1:8000/saude
# {"status":"ok","modo":"valores_fixos"}
```

### Erros

```bash
curl -i http://127.0.0.1:8000/exames/xyz/segmentacao/unet        # 404
# {"detail":{"codigo":"exame_inexistente","mensagem":"Exame 'xyz' não existe. ..."}}

curl -i http://127.0.0.1:8000/exames/demo-001/segmentacao/foo    # 400
# {"detail":{"codigo":"metodo_invalido","mensagem":"Método 'foo' inválido. Use um destes: baseline, region_growing, unet."}}
```

Na etapa 2 entra também `exame_nao_baixado` para exame que existe mas não está
em `data/luna16`.

## Dados de exemplo

`api/dados_exemplo/` tem 2 exames inventados (um formato de torso com dois
pulmões desenhados à mão, não é tomografia real): `fatia.png`, uma
`mascara_<metodo>.png` por método e `metricas.json`. As três máscaras são
diferentes de propósito (o baseline "vaza" para o ar fora do corpo, a U-Net é a
mais fiel) para a tela mostrar diferença entre os métodos. Para regenerar:
`python api/gerar_dados_exemplo.py`.

## Etapa 2 (pendente)

- Trocar os valores fixos pela função única de inferência (`src/luna16/inference.py`,
  card "Servir os modelos"). A API não chama os métodos direto nem carrega a
  U-Net a cada requisição.
- Variável de configuração para voltar ao modo de valores fixos.
- Erro `exame_nao_baixado`.
