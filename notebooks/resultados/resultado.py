import io, os, sys
os.environ["API_MODO"] = "real"; os.environ["API_EXAMES"] = "LIDC-IDRI-0001"
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from fastapi.testclient import TestClient
from api.main import app

saida = sys.argv[1]
with TestClient(app) as c:
    exame = c.get("/exames").json()["exames"][0]
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5.6))
    fatia = None
    for ax, m in zip(eixos, ("baseline", "region_growing", "unet")):
        r = c.get(f"/exames/{exame['id']}/segmentacao/{m}").json()
        if fatia is None:
            fatia = plt.imread(io.BytesIO(c.get(r["fatia"]["url"]).content))
        mask = plt.imread(io.BytesIO(c.get(r["mascara"]["url"]).content)) > 0.5
        mt = r["metricas"]
        print(m, mt)
        ax.imshow(fatia, cmap="gray")
        ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.45)
        ax.set_title(f"{m}\nDice {mt['dice']:.3f} | IoU {mt['iou']:.3f} | "
                     f"{mt['volume_pulmonar_ml']:.0f} mL | {mt['tempo_ms']/1000:.1f} s")
        ax.axis("off")
    fig.suptitle(f"{exame['id']} — fatia {r['fatia']['indice']} de {exame['num_fatias']} (modo {r['modo']})")
    fig.tight_layout(); fig.savefig(saida, dpi=110)
print("salvo", saida)
