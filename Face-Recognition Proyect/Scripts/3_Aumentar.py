"""
aumentar.py
===========
Fase 3 — Aumentación de Datos
Aplica transformaciones a Dataset_procesado/ para multiplicar las muestras
sin necesidad de nuevas capturas.
 
Técnicas implementadas (requeridas en metodología):
  ✓ Rotación           (±15°)
  ✓ Cambio de brillo   (oscurecer / aclarar)
  ✓ Espejo horizontal  (flip)
 
Técnicas adicionales recomendadas:
  + Ruido gaussiano
  + Recorte y zoom aleatorio
  + Ajuste de contraste
  + Blur leve (simula desenfoque de cámara)
  + Cambio de tono (simula distintas iluminaciones de color)
 
Uso:
    python scripts/aumentar.py --factor 6
    python scripts/aumentar.py --factor 6 --solo_basicas
"""
 
import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm
 
BASE_DIR = Path(__file__).parent.parent / "Dataset_procesado"
OUT_DIR  = Path(__file__).parent.parent / "Dataset_aumentado"
TAMANO   = (160, 160)
EXTS     = {".jpg", ".jpeg", ".png"}
 
 
# ─────────────────────────────────────────────────────────────────────────────────
# TÉCNICAS REQUERIDAS
# ─────────────────────────────────────────────────────────────────────────────────
 
def rotacion(img: np.ndarray) -> np.ndarray:
    """
    Rota la imagen entre -15° y +15°.
    Usa reflexión en los bordes para no dejar áreas negras.
    """
    angulo = random.uniform(-15, 15)
    h, w   = img.shape[:2]
    M      = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)
 
 
def cambio_brillo(img: np.ndarray) -> np.ndarray:
    """
    Simula distintas condiciones de iluminación:
    - factor < 1.0 → imagen más oscura
    - factor > 1.0 → imagen más clara
    """
    factor = random.uniform(0.50, 1.60)
    img_f  = img.astype(np.float32) * factor
    return np.clip(img_f, 0, 255).astype(np.uint8)
 
 
def espejo(img: np.ndarray) -> np.ndarray:
    """
    Volteo horizontal. Duplica muestras simétricamente.
    Importante: el espejo NO es válido para texto/asimetría de accesorios,
    pero sí es útil para rostros.
    """
    return cv2.flip(img, 1)
 
 
# ─────────────────────────────────────────────────────────────────────────────────
# TÉCNICAS ADICIONALES (recomendadas)
# ─────────────────────────────────────────────────────────────────────────────────
 
def ruido_gaussiano(img: np.ndarray) -> np.ndarray:
    """Simula ruido de sensor de cámara de baja calidad."""
    sigma = random.uniform(5, 20)
    ruido = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
 
 
def recorte_zoom(img: np.ndarray) -> np.ndarray:
    """Recorte aleatorio y redimensionado — simula distintos encuadres."""
    h, w   = img.shape[:2]
    margen = int(random.uniform(0.05, 0.15) * min(h, w))
    top    = random.randint(0, margen)
    left   = random.randint(0, margen)
    bot    = random.randint(0, margen)
    right  = random.randint(0, margen)
    recortada = img[top:h - bot, left:w - right]
    return cv2.resize(recortada, (w, h), interpolation=cv2.INTER_LANCZOS4)
 
 
def ajuste_contraste(img: np.ndarray) -> np.ndarray:
    """CLAHE con parámetros aleatorios."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = random.uniform(1.0, 4.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
 
 
def blur_leve(img: np.ndarray) -> np.ndarray:
    """Desenfoque suave que simula cámaras de baja resolución."""
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)
 
 
def cambio_tono(img: np.ndarray) -> np.ndarray:
    """
    Ajuste en espacio HSV — simula distintas temperaturas de color
    (luz incandescente, fluorescente, luz de día, etc.)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-18, 18)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 1.25), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
 
 
# ─────────────────────────────────────────────────────────────────────────────────
# Grupos de técnicas
# ─────────────────────────────────────────────────────────────────────────────────
 
BASICAS    = [rotacion, cambio_brillo, espejo]
ADICIONALES = [ruido_gaussiano, recorte_zoom, ajuste_contraste, blur_leve, cambio_tono]
 
 
def generar_variantes(img: np.ndarray, n: int, solo_basicas: bool) -> list:
    """
    Genera exactamente `n` variantes únicas de la imagen.
    Siempre incluye:
      - 1× espejo puro (flip)
      - n-1× combinaciones aleatorias del resto
    """
    pool = BASICAS if solo_basicas else BASICAS + ADICIONALES
    variantes = [espejo(img)]  # Siempre incluir espejo
 
    for _ in range(n - 1):
        resultado = img.copy()
        # Aplicar 1 a 3 transformaciones distintas al azar
        ops = random.sample([f for f in pool if f != espejo],
                            k=random.randint(1, min(3, len(pool) - 1)))
        for op in ops:
            resultado = op(resultado)
        variantes.append(resultado)
 
    return variantes
 
 
# ─────────────────────────────────────────────────────────────────────────────────
 
def aumentar_dataset(factor: int = 6, solo_basicas: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    tecnicas_usadas = "Rotación + Brillo + Espejo" if solo_basicas \
        else "Rotación + Brillo + Espejo + Ruido + Zoom + Contraste + Blur + Tono"
 
    print(f"\n{'='*60}")
    print(f"  AUMENTACIÓN DE DATOS")
    print(f"  Factor: ×{factor}  (1 original + {factor-1} sintéticas)")
    print(f"  Técnicas: {tecnicas_usadas}")
    print(f"  Fuente:  {BASE_DIR}")
    print(f"  Destino: {OUT_DIR}")
    print(f"{'='*60}\n")
 
    total_orig = 0
    total_gen  = 0
 
    grupos = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    for grupo in grupos:
        categorias = sorted([d for d in grupo.iterdir() if d.is_dir()])
        for categoria in categorias:
            imagenes = [f for f in categoria.iterdir() if f.suffix.lower() in EXTS]
            if not imagenes:
                continue
 
            carpeta_out = OUT_DIR / grupo.name / categoria.name
            carpeta_out.mkdir(parents=True, exist_ok=True)
 
            conteo = 0
            for ruta in tqdm(imagenes, desc=f"  {grupo.name}/{categoria.name}", leave=False):
                img = cv2.imread(str(ruta))
                if img is None:
                    continue
 
                # Copiar original
                cv2.imwrite(str(carpeta_out / ruta.name), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                conteo += 1
 
                # Generar variantes
                variantes = generar_variantes(img, factor - 1, solo_basicas)
                for i, var in enumerate(variantes):
                    nombre_var = f"{ruta.stem}_aug{i+1:02d}.jpg"
                    cv2.imwrite(str(carpeta_out / nombre_var), var,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    conteo += 1
 
            orig = len(imagenes)
            gen  = conteo - orig
            total_orig += orig
            total_gen  += gen
            print(f"  ✓ {grupo.name}/{categoria.name:22s}  "
                  f"{orig:3d} originales  +{gen:4d} sintéticas  = {conteo:4d} total")
 
    print(f"\n{'='*60}")
    print(f"  Imágenes originales:  {total_orig}")
    print(f"  Imágenes generadas:   {total_gen}")
    print(f"  TOTAL en dataset:     {total_orig + total_gen}")
    print(f"  Factor real:          ×{(total_orig + total_gen) / total_orig:.1f}")
    print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=6,
                        help="Multiplicador de imágenes (default: 6)")
    parser.add_argument("--solo_basicas", action="store_true",
                        help="Usar solo rotación, brillo y espejo (metodología base)")
    args = parser.parse_args()
    aumentar_dataset(factor=args.factor, solo_basicas=args.solo_basicas)