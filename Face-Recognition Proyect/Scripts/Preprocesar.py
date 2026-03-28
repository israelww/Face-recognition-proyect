"""
preprocesar.py
==============
Fase 2 — Preprocesamiento
Aplica MTCNN (o Haar como fallback) para:
  1. Detectar el rostro en cada imagen
  2. Alinear geométricamente usando los landmarks oculares (MTCNN)
  3. Recortar (crop) al área del rostro
  4. Redimensionar a 160×160 px
 
Procesa TODO el Dataset (Alumnos + Famosos) y guarda en Dataset_procesado/.
 
Uso:
    python scripts/preprocesar.py
    python scripts/preprocesar.py --confianza 0.90   # umbral MTCNN
"""
 
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
 
try:
    from mtcnn import MTCNN
    TIENE_MTCNN = True
except ImportError:
    TIENE_MTCNN = False
 
BASE_DIR  = Path(__file__).parent.parent / "Dataset"
OUT_DIR   = Path(__file__).parent.parent / "Dataset_procesado"
LOG_DIR   = Path(__file__).parent.parent / "logs"
TAMANO    = (160, 160)
EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
# ─── Detección y alineación con MTCNN ───────────────────────────────────────────
 
class ProcesadorMTCNN:
    """
    MTCNN devuelve 5 landmarks: ojo_izq, ojo_der, nariz, boca_izq, boca_der.
    Usamos los dos ojos para alinear la imagen horizontalmente antes de recortar.
    """
    def __init__(self, umbral_confianza=0.90):
        self.model     = MTCNN()
        self.umbral    = umbral_confianza
 
    def procesar(self, img_bgr):
        """Retorna (rostro_alineado_160x160, metadata) o (None, None)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resultados = self.model.detect_faces(img_rgb)
        if not resultados:
            return None, None
 
        # Mejor detección por confianza
        det = max(resultados, key=lambda r: r["confidence"])
        if det["confidence"] < self.umbral:
            return None, None
 
        # ── Alineación por landmarks oculares ──────────────────────────────
        kp        = det["keypoints"]
        ojo_izq   = np.array(kp["left_eye"],  dtype=np.float32)
        ojo_der   = np.array(kp["right_eye"], dtype=np.float32)
 
        # Ángulo entre los ojos respecto a la horizontal
        dy     = ojo_der[1] - ojo_izq[1]
        dx     = ojo_der[0] - ojo_izq[0]
        angulo = np.degrees(np.arctan2(dy, dx))
 
        # Centro entre los ojos → punto de rotación
        centro = tuple(((ojo_izq + ojo_der) / 2).astype(int))
        H, W   = img_bgr.shape[:2]
        M      = cv2.getRotationMatrix2D(centro, angulo, scale=1.0)
        img_alineada = cv2.warpAffine(img_bgr, M, (W, H),
                                      flags=cv2.INTER_LANCZOS4,
                                      borderMode=cv2.BORDER_REFLECT)
 
        # ── Recorte con margen ─────────────────────────────────────────────
        x, y, w, h = det["box"]
        margen = int(0.20 * max(w, h))
        x1 = max(0, x - margen)
        y1 = max(0, y - margen)
        x2 = min(W, x + w + margen)
        y2 = min(H, y + h + margen)
 
        recorte = img_alineada[y1:y2, x1:x2]
        if recorte.size == 0:
            return None, None
 
        recorte_160 = cv2.resize(recorte, TAMANO, interpolation=cv2.INTER_LANCZOS4)
 
        meta = {
            "confianza": round(det["confidence"], 4),
            "angulo_correccion": round(angulo, 2),
            "bbox_original": [x, y, w, h],
        }
        return recorte_160, meta
 
 
class ProcesadorHaar:
    """Fallback sin alineación cuando MTCNN no está disponible."""
    def __init__(self, **kwargs):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml)
 
    def procesar(self, img_bgr):
        gris    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rostros = self.cascade.detectMultiScale(gris, 1.1, 5, minSize=(60, 60))
        if len(rostros) == 0:
            return None, None
        x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
        margen = int(0.20 * max(w, h))
        H, W   = img_bgr.shape[:2]
        x1 = max(0, x - margen)
        y1 = max(0, y - margen)
        x2 = min(W, x + w + margen)
        y2 = min(H, y + h + margen)
        recorte = img_bgr[y1:y2, x1:x2]
        if recorte.size == 0:
            return None, None
        recorte_160 = cv2.resize(recorte, TAMANO, interpolation=cv2.INTER_LANCZOS4)
        meta = {"confianza": 1.0, "angulo_correccion": 0.0, "bbox_original": [x, y, w, h]}
        return recorte_160, meta
 
 
# ─── Pipeline ────────────────────────────────────────────────────────────────────
 
def procesar_dataset(umbral: float = 0.90):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
 
    if TIENE_MTCNN:
        print(f"\n  Detector: MTCNN  (umbral confianza: {umbral})")
        procesador = ProcesadorMTCNN(umbral_confianza=umbral)
    else:
        print(f"\n  Detector: Haar Cascades  (instala 'mtcnn' para mejor calidad)")
        procesador = ProcesadorHaar()
 
    print(f"  Resolución de salida: {TAMANO[0]}×{TAMANO[1]} px")
    print(f"  Fuente: {BASE_DIR}")
    print(f"  Destino: {OUT_DIR}\n")
 
    log_global = {
        "timestamp": datetime.now().isoformat(),
        "detector":  "MTCNN" if TIENE_MTCNN else "Haar",
        "resolucion": f"{TAMANO[0]}x{TAMANO[1]}",
        "categorias": {}
    }
 
    total_ok  = 0
    total_skip = 0
 
    grupos = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    for grupo in grupos:
        categorias = sorted([d for d in grupo.iterdir() if d.is_dir()])
        for categoria in categorias:
            imagenes = [f for f in categoria.iterdir() if f.suffix.lower() in EXTS]
            if not imagenes:
                continue
 
            carpeta_out = OUT_DIR / grupo.name / categoria.name
            carpeta_out.mkdir(parents=True, exist_ok=True)
 
            ok = 0
            skip = 0
            metas = []
 
            for ruta in tqdm(imagenes, desc=f"  {grupo.name}/{categoria.name}", leave=False):
                img = cv2.imread(str(ruta))
                if img is None:
                    skip += 1
                    continue
 
                rostro, meta = procesador.procesar(img)
                if rostro is None:
                    skip += 1
                    continue
 
                nombre_out = ruta.stem + "_proc.jpg"
                cv2.imwrite(str(carpeta_out / nombre_out), rostro,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                metas.append({**meta, "archivo": ruta.name})
                ok += 1
 
            total_ok   += ok
            total_skip += skip
            estado = "✓" if skip == 0 else f"✓ ({skip} sin rostro detectado)"
            print(f"  {estado}  {grupo.name}/{categoria.name:22s}  "
                  f"{ok:4d} procesadas  {skip:3d} omitidas")
 
            log_global["categorias"][f"{grupo.name}/{categoria.name}"] = {
                "procesadas": ok, "omitidas": skip, "muestras": metas[:5]
            }
 
    # Guardar log
    log_path = LOG_DIR / "preprocesamiento.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_global, f, indent=2, ensure_ascii=False)
 
    print(f"\n{'='*55}")
    print(f"  PREPROCESAMIENTO COMPLETO")
    print(f"  Imágenes procesadas: {total_ok}")
    print(f"  Imágenes omitidas:   {total_skip}  (sin rostro detectable)")
    print(f"  Log guardado en:     {log_path}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confianza", type=float, default=0.90,
                        help="Umbral de confianza MTCNN (default: 0.90)")
    args = parser.parse_args()
    procesar_dataset(umbral=args.confianza)