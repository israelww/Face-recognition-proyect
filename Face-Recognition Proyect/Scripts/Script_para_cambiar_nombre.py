# ------------------------------------------------------------
# INSTRUCCIONES DE USO
#
# 1) Abre una terminal en la carpeta donde esta este archivo.
# 2) Ejecuta:
#    python script.py "RUTA_DE_LA_CARPETA" NOMBRE_BASE
#
# Ejemplo:
#    python script.py "C:\Users\joeyk\OneDrive\Desktop\CNN\DataSet\Famosos\MeganFox" meganfox
#
# Resultado esperado:
#    meganfox_1.jpg, meganfox_2.png, meganfox_3.jpg, etc.
#    (conserva la extension original de cada imagen)
#
# Opciones utiles:
#    --simular      Muestra los cambios sin renombrar realmente.
#    --inicio 10    Empieza la numeracion desde 10.
#
# Ejemplos con opciones:
#    python script.py "C:\MiCarpeta" imagen --simular
#    python script.py "C:\MiCarpeta" imagen --inicio 10
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSIONES_IMAGEN = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".heic",
}


def obtener_imagenes(carpeta: Path) -> list[Path]:
    archivos = [p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN]
    return sorted(archivos, key=lambda p: p.name.lower())


def renombrar_imagenes(carpeta: Path, nombre_base: str, inicio: int, simular: bool) -> None:
    imagenes = obtener_imagenes(carpeta)
    if not imagenes:
        print("No se encontraron imágenes para renombrar.")
        return

    # Paso 1: mover a nombres temporales para evitar conflictos.
    temporales: list[Path] = []
    for i, imagen in enumerate(imagenes, start=1):
        temp = carpeta / f"__tmp_renombre_{i:06d}{imagen.suffix.lower()}"
        temporales.append(temp)
        if simular:
            print(f"[SIMULACION] {imagen.name} -> {temp.name}")
        else:
            imagen.rename(temp)

    # Paso 2: nombre final (ej. imagen_1.png).
    for i, temp in enumerate(temporales, start=inicio):
        nuevo_nombre = f"{nombre_base}_{i}{temp.suffix.lower()}"
        destino = carpeta / nuevo_nombre
        if simular:
            print(f"[SIMULACION] {temp.name} -> {destino.name}")
        else:
            temp.rename(destino)
            print(f"{temp.name} -> {destino.name}")

    if not simular:
        print(f"\nListo. Se renombraron {len(temporales)} imagen(es).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renombra imágenes de una carpeta con formato: nombre_base_1.png, nombre_base_2.jpg, etc."
    )
    parser.add_argument("carpeta", type=Path, help="Ruta de la carpeta que contiene las imágenes.")
    parser.add_argument("nombre_base", help="Nombre base deseado. Ejemplo: imagen")
    parser.add_argument(
        "--inicio",
        type=int,
        default=1,
        help="Número inicial de la secuencia (por defecto: 1).",
    )
    parser.add_argument(
        "--simular",
        action="store_true",
        help="Muestra los cambios sin renombrar archivos realmente.",
    )
    args = parser.parse_args()

    carpeta = args.carpeta.resolve()
    if not carpeta.exists() or not carpeta.is_dir():
        raise SystemExit(f"La carpeta no existe o no es válida: {carpeta}")

    if args.inicio < 0:
        raise SystemExit("El valor de --inicio no puede ser negativo.")

    renombrar_imagenes(carpeta, args.nombre_base.strip(), args.inicio, args.simular)


if __name__ == "__main__":
    main()
