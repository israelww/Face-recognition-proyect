"""
app_gui.py
==========
Stable desktop GUI for face recognition using a trained PyTorch CNN. 
Main goals:
- Reliable real-time camera mode.
- Strong UNKNOWN filtering to reduce false positives.
- Robust Windows Unicode path handling.
- Default recognition scope focused on students (Alumnos).

Run examples:
  python Scripts/app_gui.py
  python Scripts/app_gui.py --model models/actores_alumnos/best_model.pth
  python Scripts/app_gui.py --scope alumnos_actores --threshold_sim 70 --min_margin 4
"""

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import models, transforms

import tkinter as tk
from tkinter import filedialog, messagebox


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"


COLORS = {
    "bg": "#10131a",
    "panel": "#1a1f2b",
    "panel2": "#141923",
    "text": "#f2f5ff",
    "muted": "#9da7bd",
    "accent": "#38a3ff",
    "good": "#1fc88f",
    "bad": "#ff5b6e",
    "border": "#2a3144",
}


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """Robust image reading for Windows Unicode paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        try:
            return cv2.imread(path, flags)
        except Exception:
            return None


def normalize_csv_path(raw_path: str) -> str:
    """Normalize old absolute CSV paths to current project root."""
    p = Path(str(raw_path))
    if p.exists():
        return str(p)

    raw = str(raw_path).replace("\\", "/")
    anchor = "Face-Recognition Proyect"
    if anchor in raw:
        idx = raw.index(anchor)
        rel = raw[idx + len(anchor):].lstrip("/")
        return str(PROJECT_ROOT / rel)

    return str(p)


def find_best_model() -> str | None:
    if not MODELS_DIR.exists():
        return None
    candidates = list(MODELS_DIR.rglob("best_model.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def classes_from_groups(train_csv: Path, groups: list[str]) -> list[str]:
    if not train_csv.exists():
        return []
    target = {g.strip().lower() for g in groups}
    classes = set()
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("grupo", "").strip().lower() in target:
                c = row.get("clase", "").strip()
                if c:
                    classes.add(c)
    return sorted(classes)


class FaceDetector:
    """MTCNN detector if available, Haar fallback."""

    def __init__(self, mtcnn_conf: float = 0.88):
        self.mtcnn_conf = mtcnn_conf
        self.backend = "none"
        self.mtcnn = None

        try:
            from mtcnn import MTCNN

            self.mtcnn = MTCNN()
            self.backend = "mtcnn"
            print("  [Detector] MTCNN loaded")
        except Exception:
            print("  [Detector] MTCNN unavailable, using Haar")

        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.haar.empty() and self.backend == "none":
            raise RuntimeError("No face detector available (MTCNN/Haar).")
        if self.backend == "none":
            self.backend = "haar"

    def detect(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        boxes = []
        if self.mtcnn is not None:
            try:
                dets = self.mtcnn.detect_faces(frame_rgb)
                for d in dets:
                    conf = float(d.get("confidence", 0.0))
                    if conf < self.mtcnn_conf:
                        continue
                    x, y, w, h = d["box"]
                    boxes.append((int(x), int(y), int(x + w), int(y + h)))
                if boxes:
                    return boxes
            except Exception:
                pass

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes


class CNNRecognizer:
    """
    Uses trained CNN checkpoint + embedding centroid gallery.
    Decision:
      - similarity threshold
      - margin (top1 sim - top2 sim)
      - softmax threshold
    """

    def __init__(
        self,
        model_path: str,
        threshold_sim: float = 70.0,
        min_margin: float = 4.0,
        threshold_softmax: float = 0.0,
        allowed_classes: list[str] | None = None,
        gallery_npz: str | None = None,
        max_gallery_per_class: int = 180,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.threshold_sim = threshold_sim
        self.min_margin = min_margin
        self.threshold_softmax = threshold_softmax
        self.max_gallery_per_class = max_gallery_per_class

        print(f"  [Model] Loading checkpoint: {model_path}")
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(model_path, map_location=self.device)

        self.model_name = ckpt.get("model_name", "resnet18")
        self.num_classes = int(ckpt.get("num_classes", 0))
        self.img_size = int(ckpt.get("img_size", 160))
        self.class_names = ckpt.get("class_names", [f"Class_{i}" for i in range(self.num_classes)])

        self.model = self._build_model(self.model_name, self.num_classes)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(self.device)
        self.embedder = self._build_embedder().to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self.allowed_indices = self._resolve_allowed_indices()

        self.gallery_npz = (
            Path(gallery_npz) if gallery_npz else Path(model_path).resolve().parent / "gallery_embeddings_v2.npz"
        )
        self.gallery = self._load_or_build_gallery()

        print(
            "  [Model] "
            f"name={self.model_name} classes={self.num_classes} "
            f"th_sim={self.threshold_sim:.1f}% margin={self.min_margin:.1f}% "
            f"th_softmax={self.threshold_softmax:.0%}"
        )
        print(f"  [Gallery] file={self.gallery_npz}")
        if self.allowed_indices is not None:
            names = [self.class_names[i] for i in self.allowed_indices]
            print("  [Scope] Allowed classes: " + ", ".join(names))

    def _build_model(self, model_name: str, num_classes: int) -> nn.Module:
        if model_name == "resnet18":
            m = models.resnet18(weights=None)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_f, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )
            return m

        if model_name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_f, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )
            return m

        raise ValueError(f"Unsupported model_name: {model_name}")

    def _build_embedder(self) -> nn.Module:
        if self.model_name == "resnet18":

            class ResnetEmbed(nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.backbone = nn.Sequential(*list(base.children())[:-1])
                    self.proj = base.fc[1]
                    self.act = base.fc[2]

                def forward(self, x):
                    x = self.backbone(x)
                    x = torch.flatten(x, 1)
                    x = self.proj(x)
                    x = self.act(x)
                    return x

            return ResnetEmbed(self.model)

        if self.model_name == "efficientnet_b0":

            class EffEmbed(nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.features = base.features
                    self.avgpool = base.avgpool
                    self.proj = base.classifier[1]
                    self.act = base.classifier[2]

                def forward(self, x):
                    x = self.features(x)
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.proj(x)
                    x = self.act(x)
                    return x

            return EffEmbed(self.model)

        raise ValueError(f"Unsupported model for embeddings: {self.model_name}")

    def _resolve_allowed_indices(self) -> list[int] | None:
        if self.allowed_classes is None:
            return None
        idxs = [i for i, c in enumerate(self.class_names) if c in self.allowed_classes]
        if not idxs:
            print("  [Scope] Warning: no allowed classes matched model classes. Using all.")
            return None
        return idxs

    @torch.no_grad()
    def _tensor_and_embedding(self, face_rgb: np.ndarray):
        pil = Image.fromarray(face_rgb).convert("RGB")
        t = self.transform(pil).unsqueeze(0).to(self.device)
        emb = self.embedder(t)
        emb = F.normalize(emb, p=2, dim=1)
        return t, emb.squeeze(0).cpu().numpy().astype(np.float32)

    def _load_or_build_gallery(self) -> dict:
        if self.gallery_npz.exists():
            try:
                data = np.load(str(self.gallery_npz), allow_pickle=True)
                names = data["names"].tolist()
                class_indices = data["class_indices"].astype(np.int32)
                centroids = data["centroids"].astype(np.float32)
                counts = data["counts"].astype(np.int32)
                norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                centroids = centroids / np.clip(norms, 1e-9, None)
                return {
                    "names": names,
                    "class_indices": class_indices,
                    "centroids": centroids,
                    "counts": counts,
                }
            except Exception as e:
                print(f"  [Gallery] Could not load existing gallery, rebuilding. Reason: {e}")

        return self._build_gallery_from_train_csv()

    def _build_gallery_from_train_csv(self) -> dict:
        if not TRAIN_CSV.exists():
            raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")

        per_class = {c: [] for c in self.class_names}

        with open(TRAIN_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        keep = []
        for r in rows:
            cls = r.get("clase", "")
            if cls not in per_class:
                continue
            if self.allowed_classes is not None and cls not in self.allowed_classes:
                continue
            keep.append(r)

        loaded = 0
        for r in keep:
            cls = r["clase"]
            if len(per_class[cls]) >= self.max_gallery_per_class:
                continue

            p = normalize_csv_path(r.get("ruta", ""))
            if not Path(p).exists():
                continue

            img_bgr = imread_unicode(p)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _, emb = self._tensor_and_embedding(img_rgb)
            per_class[cls].append(emb)
            loaded += 1

        names = []
        class_indices = []
        centroids = []
        counts = []

        for idx, cls in enumerate(self.class_names):
            if self.allowed_classes is not None and cls not in self.allowed_classes:
                continue
            vectors = per_class[cls]
            if not vectors:
                continue
            mat = np.vstack(vectors).astype(np.float32)
            c = mat.mean(axis=0)
            c /= np.clip(np.linalg.norm(c), 1e-9, None)

            names.append(cls)
            class_indices.append(idx)
            centroids.append(c)
            counts.append(len(vectors))

        if not centroids:
            raise RuntimeError("No gallery could be built. Check train.csv paths and dataset.")

        centroids = np.vstack(centroids).astype(np.float32)
        class_indices = np.array(class_indices, dtype=np.int32)
        counts = np.array(counts, dtype=np.int32)

        np.savez(
            str(self.gallery_npz),
            names=np.array(names),
            class_indices=class_indices,
            centroids=centroids,
            counts=counts,
        )

        print(f"  [Gallery] Built from train.csv with {loaded} embeddings.")
        return {
            "names": names,
            "class_indices": class_indices,
            "centroids": centroids,
            "counts": counts,
        }

    @torch.no_grad()
    def predict(self, face_rgb: np.ndarray, top_k: int = 3) -> dict:
        t, emb = self._tensor_and_embedding(face_rgb)
        logits = self.model(t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

        centroids = self.gallery["centroids"]
        sims = centroids @ emb
        sims_pct = ((sims + 1.0) / 2.0) * 100.0

        order = sims_pct.argsort()[::-1]
        top = order[:top_k]

        top_results = []
        for rank_idx in top:
            cls_name = self.gallery["names"][rank_idx]
            cls_model_idx = int(self.gallery["class_indices"][rank_idx])
            top_results.append(
                {
                    "name": cls_name,
                    "similarity": float(sims_pct[rank_idx]),
                    "softmax": float(probs[cls_model_idx] * 100.0),
                }
            )

        best = top_results[0]
        second_sim = top_results[1]["similarity"] if len(top_results) > 1 else 0.0
        margin = best["similarity"] - second_sim

        softmax_ok = True
        if self.threshold_softmax > 0:
            softmax_ok = best["softmax"] >= self.threshold_softmax * 100.0

        is_known = (
            best["similarity"] >= self.threshold_sim
            and margin >= self.min_margin
            and softmax_ok
        )

        return {
            "name": best["name"] if is_known else "UNKNOWN",
            "is_known": is_known,
            "similarity": best["similarity"],
            "softmax": best["softmax"],
            "margin": margin,
            "top_k": top_results,
        }


class FaceApp:
    W, H = 1080, 760

    def __init__(self, detector: FaceDetector, recognizer: CNNRecognizer, camera_id: int = 0):
        self.detector = detector
        self.recognizer = recognizer
        self.camera_id = camera_id

        self.root = tk.Tk()
        self.root.title("Face Recognition - Stable")
        self.root.geometry(f"{self.W}x{self.H}")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.cap = None
        self.cam_thread = None
        self.cam_running = False

        self._camera_screen()

    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()

    @staticmethod
    def _expand_box(x1, y1, x2, y2, img_w, img_h, margin_ratio=0.20):
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        m = int(max(bw, bh) * margin_ratio)
        return (
            max(0, x1 - m),
            max(0, y1 - m),
            min(img_w, x2 + m),
            min(img_h, y2 + m),
        )

    def _header(self, title: str, color: str):
        h = tk.Frame(self.root, bg=COLORS["panel"], height=56)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Label(h, text=title, bg=COLORS["panel"], fg=color, font=("Segoe UI", 15, "bold")).pack(
            side="left", padx=16
        )
        tk.Button(
            h,
            text="Exit",
            command=self._close,
            bg=COLORS["accent"],
            fg="white",
            relief="flat",
            padx=14,
            pady=4,
            cursor="hand2",
        ).pack(side="right", padx=16)

    def _camera_screen(self):
        self._clear()
        self._header("Live Camera", COLORS["accent"])

        body = tk.Frame(self.root, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=12)

        self.video_label = tk.Label(body, bg="black", width=860, height=620)
        self.video_label.pack(side="left", padx=(0, 12))

        side = tk.Frame(body, bg=COLORS["panel"], width=260, highlightbackground=COLORS["border"], highlightthickness=1)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="Result", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 12, "bold")).pack(pady=(16, 8))

        self.cam_name = tk.Label(side, text="Waiting...", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 21, "bold"), wraplength=230)
        self.cam_name.pack(pady=(10, 8))

        self.cam_score = tk.Label(side, text="", bg=COLORS["panel"], fg=COLORS["text"], font=("Consolas", 11), justify="left")
        self.cam_score.pack(pady=(4, 8))

        tk.Frame(side, bg=COLORS["border"], height=1).pack(fill="x", padx=12, pady=(8, 10))

        self.cam_top = []
        for _ in range(3):
            lbl = tk.Label(side, text="", bg=COLORS["panel"], fg=COLORS["text"], font=("Consolas", 10), anchor="w", justify="left", wraplength=230)
            lbl.pack(fill="x", padx=14, pady=3)
            self.cam_top.append(lbl)

        self.cam_status = tk.Label(self.root, text="Starting camera...", bg=COLORS["bg"], fg=COLORS["muted"], font=("Consolas", 9), anchor="w")
        self.cam_status.pack(fill="x", padx=14, pady=(0, 8))

        self.cam_running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _camera_loop(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Camera", "Could not open webcam."))
            self.root.after(0, self._close)
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_idx = 0
        while self.cam_running:
            ok, frame_bgr = self.cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_idx += 1

            if frame_idx % 2 == 0:
                faces = self.detector.detect(frame_rgb)
                draw = frame_rgb.copy()
                if faces:
                    faces = sorted(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        h, w = frame_rgb.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        x1, y1, x2, y2 = self._expand_box(x1, y1, x2, y2, w, h, margin_ratio=0.20)

                        crop = frame_rgb[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        result = self.recognizer.predict(crop, top_k=3)
                        if i == 0:
                            self.root.after(0, self._update_cam_result, result)

                        color = (31, 200, 143) if result["is_known"] else (255, 91, 110)
                        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                        label = f"{result['name']} | sim {result['similarity']:.1f}%"
                        tsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
                        cv2.rectangle(draw, (x1, y1 - 24), (x1 + tsz[0] + 8, y1), color, -1)
                        cv2.putText(draw, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

                    frame_rgb = draw
                else:
                    self.root.after(0, self._update_cam_result, None)

            try:
                img = Image.fromarray(frame_rgb).resize((860, 620), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                self.root.after(0, self._show_video, imgtk)
            except Exception:
                pass

            status_text = (
                f"Detector: {self.detector.backend} | "
                f"th_sim={self.recognizer.threshold_sim:.1f}% | "
                f"margin={self.recognizer.min_margin:.1f}%"
            )
            self.root.after(0, lambda t=status_text: self.cam_status.configure(text=t))
            time.sleep(0.02)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _show_video(self, imgtk):
        try:
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception:
            pass

    def _update_cam_result(self, result):
        if result is None:
            self.cam_name.configure(text="No face", fg=COLORS["muted"])
            self.cam_score.configure(text="")
            for lbl in self.cam_top:
                lbl.configure(text="")
            return

        name_color = COLORS["good"] if result["is_known"] else COLORS["bad"]
        self.cam_name.configure(text=result["name"], fg=name_color)
        self.cam_score.configure(
            text=(
                f"sim:     {result['similarity']:.1f}%\n"
                f"softmax: {result['softmax']:.1f}%\n"
                f"margin:  {result['margin']:.1f}%"
            ),
            fg=name_color,
        )

        for i in range(3):
            if i < len(result["top_k"]):
                item = result["top_k"][i]
                self.cam_top[i].configure(
                    text=f"{i+1}. {item['name']} | sim {item['similarity']:.1f}% | soft {item['softmax']:.1f}%"
                )
            else:
                self.cam_top[i].configure(text="")

    def _stop_camera(self):
        self.cam_running = False
        if self.cam_thread is not None:
            self.cam_thread.join(timeout=2)
            self.cam_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _close(self):
        self._stop_camera()
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Stable GUI for face recognition with trained CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam id")

    parser.add_argument(
        "--scope",
        type=str,
        default="alumnos_actores",
        choices=["alumnos", "actores", "alumnos_actores", "all", "custom"],
        help="Recognition scope",
    )
    parser.add_argument("--groups", nargs="+", default=["Alumnos"], help="Groups for custom scope")

    parser.add_argument("--threshold_sim", type=float, default=70.0, help="Similarity threshold in percent")
    parser.add_argument("--min_margin", type=float, default=4.0, help="Minimum margin (top1-top2 similarity)")
    parser.add_argument(
        "--threshold_softmax",
        type=float,
        default=0.0,
        help="Softmax threshold (0-1). Use 0 to disable this filter.",
    )

    parser.add_argument("--gallery", type=str, default=None, help="Optional gallery .npz path")
    parser.add_argument("--max_gallery_per_class", type=int, default=180, help="Max train samples per class for gallery")

    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path:
        print("\nERROR: No best_model.pth found under models/\n")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"\nERROR: Model not found: {model_path}\n")
        sys.exit(1)

    allowed_classes = None
    if args.scope == "alumnos":
        allowed_classes = classes_from_groups(TRAIN_CSV, ["Alumnos"])
        if not allowed_classes:
            print("  [Scope] No Alumnos classes found in train.csv, using all classes.")
            allowed_classes = None
    elif args.scope == "actores":
        allowed_classes = classes_from_groups(TRAIN_CSV, ["Actores"])
        if not allowed_classes:
            print("  [Scope] No Actores classes found in train.csv, using all classes.")
            allowed_classes = None
    elif args.scope == "alumnos_actores":
        allowed_classes = classes_from_groups(TRAIN_CSV, ["Alumnos", "Actores"])
        if not allowed_classes:
            print("  [Scope] No Alumnos/Actores classes found in train.csv, using all classes.")
            allowed_classes = None
    elif args.scope == "custom":
        allowed_classes = classes_from_groups(TRAIN_CSV, args.groups)
        if not allowed_classes:
            print("  [Scope] No classes found for custom groups, using all classes.")
            allowed_classes = None

    print("\n" + "=" * 70)
    print("FACE RECOGNITION - STABLE REBUILD")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(
        f"Config: scope={args.scope} th_sim={args.threshold_sim:.1f}% "
        f"margin={args.min_margin:.1f}% th_softmax={args.threshold_softmax:.0%}"
    )

    detector = FaceDetector(mtcnn_conf=0.88)
    recognizer = CNNRecognizer(
        model_path=model_path,
        threshold_sim=args.threshold_sim,
        min_margin=args.min_margin,
        threshold_softmax=args.threshold_softmax,
        allowed_classes=allowed_classes,
        gallery_npz=args.gallery,
        max_gallery_per_class=args.max_gallery_per_class,
        device="cpu",
    )

    app = FaceApp(detector=detector, recognizer=recognizer, camera_id=args.camera_id)
    app.run()


if __name__ == "__main__":
    main()
