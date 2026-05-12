import torch
import torch.nn as nn


class DualStreamConfig:
    def __init__(self, num_classes, image_size):
        self.num_classes = num_classes
        self.image_size = image_size


class DualStreamASLModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ───── Landmark Branch (ONLY) ─────
        self.landmark_branch = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        # ───── Classifier ─────
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, config.num_classes)
        )

    def forward(self, img, lm):
        # 🔴 IGNORE image branch completely
        lm = self.landmark_branch(lm)
        return self.classifier(lm)