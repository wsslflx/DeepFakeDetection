import numpy as np
from PIL import Image
from pathlib import Path
from scipy.fft import fft2, fftshift, dctn
from scipy.ndimage import laplace
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
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

def pixel_stats(arr: np.ndarray, gray: np.ndarray) -> np.ndarray:
    feats = []

    for c in range(3):
        ch = arr[:, :, c].ravel()
        feats += [ch.mean(), ch.std(), skew(ch), kurtosis(ch)]

    lap = laplace(gray).ravel()
    feats += [lap.mean(), lap.std(), np.abs(lap).mean()]
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    feats += hist.tolist()

    dx = np.diff(gray, axis=1).ravel()
    dy = np.diff(gray, axis=0).ravel()
    feats += [dx.std(), dy.std(), np.abs(dx).mean(), np.abs(dy).mean()]

    return np.array(feats)
def extract(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    fft_feats = np.concatenate([
        fft_radial(arr[:, :, c]) for c in range(3)
    ])

    dct_feats = dct_stats(gray)

    slope_feats = np.array([
        np.polyfit(
            np.log1p(np.arange(128)),
            np.log1p(fft_radial(arr[:, :, c])),
            1
        )[0]
        for c in range(3)
    ])

    pix_feats = pixel_stats(arr, gray)
    return np.concatenate([fft_feats, dct_feats, slope_feats, pix_feats])

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