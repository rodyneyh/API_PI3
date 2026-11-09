# U-Net treinada (seed 123)

Resultado do treino da U-Net copiado do HD (`D:\DATASET_LUNA16\treino_seed123`
e `divisao_exames`), para a API carregar o modelo sem o HD conectado.

| arquivo | o que é |
|---|---|
| `unet_novo_final.pt` | pesos da `SmallUNet2D` (base 32 canais), última época (25ª). É o que a API usa (`API_PESOS_UNET`). |
| `historico_treino.csv` | perda de treino e de validação e Dice por fatia na validação, época a época |
| `avaliacao_novo_final_teste_interno.csv` | Dice, IoU, sensibilidade e precisão em cada um dos 107 exames do teste interno (média de Dice 0,983) |
| `split_novo.csv` | conjunto de cada exame: treino (497), validação (106), teste_interno (107), anterior (177) |
| `split_novo_meta.json` | resumo da divisão (seed 42, 70/15/15) |

O checkpoint é um dicionário com `model_state`, `base_ch` (32), `train_uids`
(os 497 exames de treino) e `epoca`. Para carregar:
`src.luna16.inference.carregar_unet("modelos/unet_seed123/unet_novo_final.pt")`.

Ficaram no HD: `unet_novo_best.pt` (a API não usa), `estado_treino.pt`
(23 MB, só para retomar o treino) e a pasta `cache/` (fatias de imagem).
