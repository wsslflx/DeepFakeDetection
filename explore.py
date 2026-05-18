from pathlib import Path
from PIL import Image
import numpy as np

data_root = Path("data")

for split in ["train", "val"]:
    folders = list((data_root / split).iterdir())
    print(f"\n{split}:")
    for f in sorted(folders):
        imgs = list(f.glob("*.png"))
        print(f"  {f.name}: {len(imgs)} images")

sample = list((data_root / "train" / "real").glob("*.png"))[0]
img = Image.open(sample)
print(f"\nImage size: {img.size}, mode: {img.mode}")