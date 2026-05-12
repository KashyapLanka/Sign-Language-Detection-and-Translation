import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dual_stream_model import DualStreamASLModel, DualStreamConfig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


VERSION = "v1.5 FINAL + GRAPHS"
print(f"\n🚀 Running TRAIN VERSION: {VERSION}\n")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class RAMDataset(Dataset):
    def __init__(self, imgs, lms, labels):
        self.imgs = imgs.float()
        self.lms = lms.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.imgs[i]

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3,1,1)
        x = (x - mean) / std

        lm = self.lms[i]

        lm = lm.view(-1, 3)
        lm = lm - lm[0]
        lm = lm / (torch.norm(lm) + 1e-6)

        lm = lm + torch.randn_like(lm) * 0.02
        scale = 1 + torch.randn(1) * 0.1
        lm = lm * scale
        lm = lm + torch.randn_like(lm) * 0.02

        lm = lm.view(-1)

        return x, lm, self.labels[i]


# ─────────────────────────────────────────────
def split(labels):
    indices = list(range(len(labels)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=labels.numpy(),
        random_state=42
    )

    return train_idx, val_idx


# ─────────────────────────────────────────────
def train():
    print("Loading dataset...")
    data = torch.load("../dataset.pt")

    imgs = data["images"]
    lms = data["landmarks"]
    labels = data["labels"]
    classes = data["classes"]

    tr, vl = split(labels)

    train_loader = DataLoader(
        RAMDataset(imgs[tr], lms[tr], labels[tr]),
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        RAMDataset(imgs[vl], lms[vl], labels[vl]),
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DualStreamASLModel(
        DualStreamConfig(len(classes), (160,160))
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # 🔴 METRICS STORAGE
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []

    best_val = 0

    for epoch in range(12):

        model.train()
        tc, ts = 0, 0

        for x, l, y in train_loader:
            x, l, y = x.to(device), l.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x, l)
            loss = loss_fn(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tc += (out.argmax(1) == y).sum().item()
            ts += y.size(0)

        train_acc = tc / ts

        model.eval()
        vc, vs = 0, 0
        vloss = 0

        with torch.no_grad():
            for x, l, y in val_loader:
                x, l, y = x.to(device), l.to(device), y.to(device)

                out = model(x, l)
                loss = loss_fn(out, y)

                vloss += loss.item()
                vc += (out.argmax(1) == y).sum().item()
                vs += y.size(0)

        val_acc = vc / vs
        vloss /= len(val_loader)

        # 🔴 STORE METRICS
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        val_loss_list.append(vloss)

        # 🔴 SAVE BEST MODEL
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state_dict": model.state_dict()}, "model.pt")

        print(
            f"Epoch {epoch:02d} | "
            f"Train Acc {train_acc:.3f} | "
            f"Val Acc {val_acc:.3f} | "
            f"Val Loss {vloss:.4f}"
        )

    print("Best model saved!")

    # ─────────────────────────────
    # 📊 PLOTTING
    # ─────────────────────────────
    import matplotlib.pyplot as plt

    epochs = list(range(len(train_acc_list)))

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(epochs, val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss vs Epoch")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.show()
    
    #confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, l, y in val_loader:
            x, l = x.to(device), l.to(device)

            out = model(x, l)
            preds = out.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    train()