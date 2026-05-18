import numpy as np
from PIL import Image
from pathlib import Path
from scipy.fft import fft2, fftshift, dctn
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle

LABEL_MAP = {
    "real": 0, "hmar_d20": 1, "hmar_d30": 2,
    "llamagen_B_VQ-16": 3, "llamagen_L_VQ-16": 4,
    "nspvar_20": 5, "nspvar_30": 6,
    "rar_l": 7, "rar_xxl": 8,
}

def fft_radial(gray: np.ndarray, n_bins: int = 128) -> np.ndarray:
    f = np.abs(fftshift(fft2(gray)))
    f = np.log1p(f)
    cy, cx = np.array(f.shape) // 2
    y, x = np.ogrid[:f.shape[0], :f.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    radial = np.bincount(r.ravel(), f.ravel(), minlength=n_bins)[:n_bins]
    counts  = np.bincount(r.ravel(),            minlength=n_bins)[:n_bins]
    return radial / (counts + 1e-8)

def dct_stats(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    h, w = (h // 8) * 8, (w // 8) * 8
    blocks = gray[:h, :w].reshape(h//8, 8, w//8, 8).transpose(0,2,1,3)
    dct = dctn(blocks, axes=(2,3), norm='ortho')
    return dct.mean(axis=(0,1)).ravel()

def extract(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert('L')
    gray = np.array(img, dtype=np.float32) / 255.0
    f1 = fft_radial(gray)
    f2 = dct_stats(gray)
    return np.concatenate([f1, f2])

def process_split(split: str, data_root: Path):
    X, y, paths = [], [], []
    split_dir = data_root / split

    if split == "test":
        img_paths = sorted(split_dir.glob("*.png"))
        feats = Parallel(n_jobs=8)(
            delayed(extract)(p) for p in tqdm(img_paths, desc="test")
        )
        return np.array(feats), None, [str(p) for p in img_paths]

    for class_dir in sorted(split_dir.iterdir()):
        label = LABEL_MAP[class_dir.name]
        img_paths = sorted(class_dir.glob("*.png"))
        feats = Parallel(n_jobs=8)(
            delayed(extract)(p) for p in tqdm(img_paths, desc=class_dir.name)
        )
        X.extend(feats)
        y.extend([label] * len(feats))

    return np.array(X), np.array(y), None

if __name__ == "__main__":
    root = Path("data")
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        X, y, paths = process_split(split, root)
        out = {"X": X, "y": y, "paths": paths}
        with open(f"features_{split}.pkl", "wb") as f:
            pickle.dump(out, f)
        print(f"  Shape: {X.shape}")