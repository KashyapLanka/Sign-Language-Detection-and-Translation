import cv2
import torch
import time
import numpy as np

from background_aware_preprocessor import BackgroundAwarePreprocessor
from dual_stream_model import DualStreamASLModel, DualStreamConfig
from temporal_head import RealTimeDebouncer


class LiveInference:
    def __init__(self, checkpoint="model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset labels
        data = torch.load("../dataset.pt")
        self.labels = data["classes"]

        # Build model (landmark-only)
        self.model = DualStreamASLModel(
            DualStreamConfig(len(self.labels), (160, 160))
        ).to(self.device)

        # Load weights
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        # Preprocessor
        self.pre = BackgroundAwarePreprocessor()

        # ✅ FIX: match your actual class signature
        self.debouncer = RealTimeDebouncer(class_labels=self.labels)

        # Confidence threshold (slightly relaxed)
        self.conf_thresh = 0.3

    # ─────────────────────────────
    # Landmark normalization
    # ─────────────────────────────
    def normalize_landmarks(self, lm):
        lm = lm.astype(np.float32)

        # Center at wrist (landmark 0)
        wrist = lm[0]
        lm = lm - wrist

        # Scale normalization
        scale = np.linalg.norm(lm)
        if scale > 0:
            lm = lm / scale

        return lm.flatten()

    # ─────────────────────────────
    # Main loop
    # ─────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received")
                continue

            start = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pre(rgb)

            if not res.success:
                cv2.imshow("Output", frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # 🔴 Normalize landmarks
            lm = self.normalize_landmarks(res.landmarks)
            lm = torch.tensor(lm, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 🔴 Dummy image (model ignores it)
            dummy_img = torch.zeros((1, 3, 160, 160), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                out = self.model(dummy_img, lm)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            # Optional debug
            # print(np.argmax(probs), np.max(probs))

            deb = self.debouncer.step(probs)

            label = deb["predicted_label"]
            conf = deb["confidence"]

            if conf < self.conf_thresh:
                label = "..."

            latency = (time.time() - start) * 1000

            # Display
            cv2.putText(frame, f"{label} ({conf:.2f})",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.putText(frame, f"{latency:.1f} ms",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            cv2.imshow("Output", frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LiveInference().run()