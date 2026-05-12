import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from background_aware_preprocessor import BackgroundAwarePreprocessor

DATASET = "../Dataset"
OUTPUT = "../dataset.pt"


# ─────────────────────────────────────────────
# Worker (runs in separate process)
# ─────────────────────────────────────────────
def process_image(args):
    path, label = args

    try:
        # IMPORTANT: create preprocessor inside process
        pre = BackgroundAwarePreprocessor()

        img = cv2.imread(path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = pre(img)
        if not res.success:
            return None

        roi = res.roi.astype(np.float32) / 255.0
        lm = res.landmarks.astype(np.float32)

        # Center (make relative)
        lm = lm - lm.mean(axis=0)

        # Scale normalization (optional but recommended)
        scale = np.linalg.norm(lm)
        if scale > 0:
            lm = lm / scale

        lm = lm.flatten()

        return roi.transpose(2, 0, 1), lm, label

    except Exception:
        return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Building dataset with multiprocessing...")

    classes = sorted(os.listdir(DATASET))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Build task list
    tasks = []
    for cls in classes:
        cls_path = os.path.join(DATASET, cls)
        for img_name in os.listdir(cls_path):
            path = os.path.join(cls_path, img_name)
            tasks.append((path, class_to_idx[cls]))

    print(f"Total images: {len(tasks)}")
    print(f"Using {cpu_count()} CPU cores")

    images = []
    landmarks = []
    labels = []

    skipped = 0

    # Multiprocessing
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))

    for r in results:
        if r is None:
            skipped += 1
            continue

        roi, lm, label = r
        images.append(roi)
        landmarks.append(lm)
        labels.append(label)

    print("\nProcessing complete")
    print(f"Valid samples: {len(labels)}")
    print(f"Skipped: {skipped}")

    # Convert to tensors (efficient)
    images = torch.tensor(np.array(images), dtype=torch.float32)
    landmarks = torch.tensor(np.array(landmarks), dtype=torch.float32)
    labels = torch.tensor(labels)

    torch.save({
        "images": images,
        "landmarks": landmarks,
        "labels": labels,
        "classes": classes
    }, OUTPUT)

    print(f"\nDataset saved to: {OUTPUT}")


# ─────────────────────────────────────────────
# Windows-safe entry
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()