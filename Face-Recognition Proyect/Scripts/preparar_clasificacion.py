"""
preparar_clasificacion.py
=========================
Fase 4 — Preparación para Clasificación Multiclase
Divide Dataset_aumentado/ en conjuntos train / val / test
y genera los archivos de configuración necesarios para entrenar
con PyTorch (DataLoader) o Keras (ImageDataGenerator).
 
Splits estándar:
  - Train:      70%
  - Validación: 15%
  - Test:       15%
 
Genera:
  - splits/train.csv, val.csv, test.csv
  - splits/clases.json   ← mapeo índice → nombre de clase
  - splits/estadisticas.json
 
Uso:
    python scripts/preparar_clasificacion.py
    python scripts/preparar_clasificacion.py --train 0.8 --val 0.1 --test 0.1
"""
 
import cv2
import json
import random
import argparse
import csv
from pathlib import Path
from collections import Counter
 
BASE_DIR  = Path(__file__).parent.parent / "Dataset_aumentado"
SPLIT_DIR = Path(__file__).parent.parent / "splits"
EXTS      = {".jpg", ".jpeg", ".png"}
SEED      = 42
 
 
# ─── Recolectar todas las imágenes con sus etiquetas ────────────────────────────
 
def recolectar_muestras(base: Path):
    """
    Recorre Dataset_aumentado/ y genera una lista de (ruta, clase, grupo).
    La clase se define por el nombre de la carpeta (Alumno1, Famoso1, etc.)
    """
    muestras = []
    clases   = set()
 
    grupos = sorted([d for d in base.iterdir() if d.is_dir()])
    for grupo in grupos:
        categorias = sorted([d for d in grupo.iterdir() if d.is_dir()])
        for categoria in categorias:
            nombre_clase = categoria.name
            clases.add(nombre_clase)
            for img in categoria.iterdir():
                if img.suffix.lower() in EXTS:
                    muestras.append({
                        "ruta":   str(img),
                        "clase":  nombre_clase,
                        "grupo":  grupo.name,   # Alumnos o Famosos
                    })
 
    return muestras, sorted(clases)
 
 
# ─── División train/val/test estratificada ───────────────────────────────────────
 
def dividir_estratificado(muestras, train_r, val_r, test_r, seed=SEED):
    """
    División estratificada: mantiene la proporción de cada clase en los tres splits.
    Esto es crítico para clasificación multiclase — evita que una clase
    quede subrepresentada en validación o test.
    """
    random.seed(seed)
 
    # Agrupar por clase
    por_clase = {}
    for m in muestras:
        por_clase.setdefault(m["clase"], []).append(m)
 
    train, val, test = [], [], []
 
    for clase, items in por_clase.items():
        random.shuffle(items)
        n     = len(items)
        n_tr  = int(n * train_r)
        n_val = int(n * val_r)
        # El resto va a test
 
        train.extend(items[:n_tr])
        val.extend(items[n_tr:n_tr + n_val])
        test.extend(items[n_tr + n_val:])
 
    # Mezclar dentro de cada split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
 
    return train, val, test
 
 
# ─── Guardar CSVs y configuración ────────────────────────────────────────────────
 
def guardar_csv(muestras, ruta, mapeo_clases):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ruta", "clase", "clase_idx", "grupo"])
        for m in muestras:
            writer.writerow([
                m["ruta"],
                m["clase"],
                mapeo_clases[m["clase"]],
                m["grupo"],
            ])
 
 
def verificar_imagen(ruta: str) -> bool:
    img = cv2.imread(ruta)
    return img is not None and img.shape[:2] == (160, 160)
 
 
# ─── Reporte ─────────────────────────────────────────────────────────────────────
 
def imprimir_reporte(train, val, test, clases):
    total = len(train) + len(val) + len(test)
    print(f"\n{'='*62}")
    print(f"  SPLITS GENERADOS")
    print(f"{'─'*62}")
    print(f"  {'Split':<10} {'Imágenes':>10}  {'%':>6}")
    print(f"  {'─'*30}")
    for nombre, split in [("Train", train), ("Val", val), ("Test", test)]:
        pct = len(split) / total * 100
        print(f"  {nombre:<10} {len(split):>10,}  {pct:>5.1f}%")
    print(f"  {'─'*30}")
    print(f"  {'TOTAL':<10} {total:>10,}  100.0%")
    print(f"\n  Clases ({len(clases)}):")
    conteos = Counter(m["clase"] for m in train + val + test)
    for clase in clases:
        n = conteos[clase]
        barra = "█" * min(n // 10, 30)
        print(f"    {clase:20s}  {n:5d}  {barra}")
    print(f"{'='*62}\n")
 
 
# ─── Entry point ─────────────────────────────────────────────────────────────────
 
def preparar(train_r=0.70, val_r=0.15, test_r=0.15):
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Los ratios deben sumar 1.0"
 
    if not BASE_DIR.exists():
        print(f"\n  ERROR: No se encontró {BASE_DIR}")
        print("  Ejecuta primero: python scripts/preprocesar.py")
        print("                   python scripts/aumentar.py\n")
        return
 
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
 
    print(f"\n  Recolectando imágenes de {BASE_DIR}...")
    muestras, clases = recolectar_muestras(BASE_DIR)
    print(f"  Total: {len(muestras):,} imágenes  |  {len(clases)} clases")
 
    # Mapeo clase → índice (para PyTorch/Keras)
    mapeo_clases  = {c: i for i, c in enumerate(clases)}
    mapeo_inverso = {i: c for c, i in mapeo_clases.items()}
 
    print(f"\n  Dividiendo dataset (train={train_r:.0%} / val={val_r:.0%} / test={test_r:.0%})...")
    train, val, test = dividir_estratificado(muestras, train_r, val_r, test_r)
 
    # Guardar CSVs
    guardar_csv(train, SPLIT_DIR / "train.csv", mapeo_clases)
    guardar_csv(val,   SPLIT_DIR / "val.csv",   mapeo_clases)
    guardar_csv(test,  SPLIT_DIR / "test.csv",  mapeo_clases)
    print(f"  ✓ train.csv, val.csv, test.csv  →  {SPLIT_DIR}")
 
    # Guardar configuración de clases
    config = {
        "n_clases":       len(clases),
        "clases":         mapeo_inverso,
        "clases_nombre":  clases,
        "resolucion":     "160x160",
        "total_imagenes": len(muestras),
        "splits": {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        }
    }
    with open(SPLIT_DIR / "clases.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ clases.json  →  {SPLIT_DIR / 'clases.json'}")
 
    imprimir_reporte(train, val, test, clases)
 
    # ── Snippet de uso para PyTorch ──────────────────────────────────────────
    print("  ─── Cómo cargar en PyTorch ───────────────────────────────────")
    print("""
  import pandas as pd
  from torch.utils.data import Dataset, DataLoader
  from torchvision import transforms
  from PIL import Image
 
  class FaceDataset(Dataset):
      def __init__(self, csv_path, transform=None):
          self.df        = pd.read_csv(csv_path)
          self.transform = transform
 
      def __len__(self):
          return len(self.df)
 
      def __getitem__(self, idx):
          row   = self.df.iloc[idx]
          img   = Image.open(row["ruta"]).convert("RGB")
          label = int(row["clase_idx"])
          if self.transform:
              img = self.transform(img)
          return img, label
 
  transform = transforms.Compose([
      transforms.Resize((160, 160)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ])
 
  train_loader = DataLoader(
      FaceDataset("splits/train.csv", transform=transform),
      batch_size=32, shuffle=True
  )
    """)
    print("  ──────────────────────────────────────────────────────────────\n")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    args = parser.parse_args()
    preparar(train_r=args.train, val_r=args.val, test_r=args.test)