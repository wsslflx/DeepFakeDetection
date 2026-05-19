import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DATA_ROOT   = Path("data")
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-4
NUM_CLASSES = 9
NUM_WORKERS = 4
SAVE_PATH   = Path("best_model.pt")

LABEL_MAP = {
    "real": 0, "hmar_d20": 1, "hmar_d30": 2,
    "llamagen_B_VQ-16": 3, "llamagen_L_VQ-16": 4,
    "nspvar_20": 5, "nspvar_30": 6,
    "rar_l": 7, "rar_xxl": 8,
}
IDX_TO_NAME = {v: k for k, v in LABEL_MAP.items()}


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, split: str, transform):
        self.transform = transform
        self.items = []  # (path, label)

        split_dir = DATA_ROOT / split
        if split == "test":
            for p in sorted(split_dir.glob("*.png")):
                self.items.append((p, -1))
        else:
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.name not in LABEL_MAP:
                    continue
                label = LABEL_MAP[class_dir.name]
                for p in sorted(class_dir.glob("*.png")):
                    self.items.append((p, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels

def per_class_accuracy(preds, labels, num_classes):
    counts   = np.zeros(num_classes, dtype=int)
    corrects = np.zeros(num_classes, dtype=int)
    for p, l in zip(preds, labels):
        counts[l]   += 1
        corrects[l] += int(p == l)
    return corrects / (counts + 1e-8)

if __name__ == "__main__":
    # Datasets
    train_ds = ImageDataset("train", train_transform)
    val_ds   = ImageDataset("val",   val_transform)
    print(f"Train: {len(train_ds)} images")
    print(f"Val:   {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model     = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0
    history = []

    print(f"\nTraining EfficientNet-B0 for {EPOCHS} epochs\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, labels = val_epoch(model, val_loader, criterion)
        scheduler.step()

        # per-class breakdown
        per_class = per_class_accuracy(preds, labels, NUM_CLASSES)

        print(f"  loss: train={train_loss:.4f}  val={val_loss:.4f}")
        print(f"  acc:  train={train_acc:.4f}  val={val_acc:.4f}")
        print("  per-class val acc:")
        for i, acc in enumerate(per_class):
            print(f"    {IDX_TO_NAME[i]:20s}: {acc:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
            "per_class": per_class.tolist(),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
                "label_map": LABEL_MAP,
            }, SAVE_PATH)
            print(f"  ↑ saved best model (val acc: {best_val_acc:.4f})")

        print()

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {SAVE_PATH}")