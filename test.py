from pathlib import Path
import numpy as np
from PIL import Image

base     = Path("data/processed")
out_base = Path("ll_previews")

for res_dir in sorted(base.glob("resolution_*")):              # ex.: resolution_16, resolution_32…
    out_dir = out_base / res_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for npy_path in res_dir.glob("*.npy"):
        arr = np.load(npy_path, mmap_mode="r")              # (12, res, res)
        ll  = arr[:3]                                       # (3, res, res)

        # mise à l’échelle dynamique [vmin, vmax] → [0, 255]
        vmin, vmax = ll.min(), ll.max()
        if vmax == vmin:
            img = np.zeros_like(ll, dtype=np.uint8)
        else:
            img = np.interp(ll, (vmin, vmax), (0, 255)).astype(np.uint8)

        img = img.transpose(1, 2, 0)                        # (H, W, C)
        Image.fromarray(img, mode="RGB") \
             .save(out_dir / f"{npy_path.stem}_LL.png")