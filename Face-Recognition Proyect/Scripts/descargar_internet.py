# pip install icrawler imagehash Pillow

import os
import hashlib
import logging
from pathlib import Path
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler
from icrawler.builtin.google import GoogleParser

# ── Silenciar logs internos de icrawler ──────────────────────────────
logging.getLogger("icrawler").setLevel(logging.ERROR)


# ────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ────────────────────────────────────────────────────────────────────
CELEBRITY_NAME = "Justin Bieber"   # 👈 Cambia aquí
N_IMAGES       = 50                   # Total de imágenes deseadas
OUTPUT_PATH    = "./dataset"            # Carpeta raíz
ENABLE_GOOGLE  = False                  # Google desactivado temporalmente por cambios/bloqueos en su HTML
# ────────────────────────────────────────────────────────────────────


class SafeGoogleParser(GoogleParser):
    """Evita crash cuando Google no devuelve URLs y parse() retornaría None."""

    def parse(self, response):
        try:
            return super().parse(response) or []
        except Exception as e:
            logging.getLogger(__name__).warning("Google parser falló y se omitirá la página: %s", e)
            return []


# Fuentes con sus keywords y cuántas imágenes aporta cada una
SOURCES = [
    {
        "crawler_cls": BingImageCrawler,
        "keyword":     f"{CELEBRITY_NAME} face portrait",
        "max_num":     60,
        "filters":     {"type": "photo"},
    },
    {
        "crawler_cls": BaiduImageCrawler,
        "keyword":     f"{CELEBRITY_NAME} face portrait",
        "max_num":     40,
        "filters":     {},
    },
]

if ENABLE_GOOGLE:
    SOURCES.insert(
        1,
        {
            "crawler_cls": GoogleImageCrawler,
            "keyword":     f"{CELEBRITY_NAME} portrait headshot",
            "max_num":     50,
            "filters":     {"type": "face"},   # Google tiene filtro específico de cara
        },
    )


def get_file_hash(filepath: Path) -> str:
    """Genera un hash MD5 del contenido de la imagen para detectar duplicados."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(folder: Path) -> int:
    """Elimina imágenes duplicadas comparando hashes MD5. Retorna cuántas eliminó."""
    seen_hashes = {}
    removed = 0
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            img_hash = get_file_hash(img_path)
            if img_hash in seen_hashes:
                img_path.unlink()
                removed += 1
            else:
                seen_hashes[img_hash] = img_path
        except Exception:
            pass
    return removed


def download_celebrity(name: str, n_images: int, base_path: str):
    safe_name = name.replace(" ", "_")
    save_dir  = Path(base_path) / safe_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  🎯 Objetivo : {name}")
    print(f"  📁 Destino  : {save_dir}")
    print(f"  🖼️  Total    : {n_images} imágenes")
    print(f"{'='*55}\n")

    active_sources = [dict(source) for source in SOURCES]
    if not ENABLE_GOOGLE:
        print("  ℹ️ Google está desactivado temporalmente para evitar el error de parser.\n")

    # ── Ajustar proporciones si n_images difiere del total por defecto ──
    total_default = sum(s["max_num"] for s in active_sources)
    for source in active_sources:
        source["max_num"] = max(1, round(source["max_num"] * n_images / total_default))

    # ── Descargar desde cada fuente ──────────────────────────────────
    for i, source in enumerate(active_sources):
        crawler_cls = source["crawler_cls"]
        source_name = crawler_cls.__name__.replace("ImageCrawler", "")
        print(f"  [{i+1}/{len(active_sources)}] 🔍 {source_name} → {source['max_num']} imágenes ...")

        try:
            crawler_kwargs = dict(
                downloader_threads=4,
                storage={"root_dir": str(save_dir)},
            )
            if crawler_cls is GoogleImageCrawler:
                crawler_kwargs["parser_cls"] = SafeGoogleParser
            crawler = crawler_cls(**crawler_kwargs)
            crawler.crawl(
                keyword=source["keyword"],
                filters=source["filters"] if source["filters"] else None,
                max_num=source["max_num"],
                file_idx_offset="auto",   # Numeración consecutiva sin solapamientos
            )
            print(f"       ✅ {source_name} completado")
        except Exception as e:
            print(f"       ⚠️  {source_name} falló: {e}")

    # ── Eliminar duplicados ──────────────────────────────────────────
    print("\n  🔎 Eliminando duplicados por hash MD5...")
    removed = remove_duplicates(save_dir)

    # ── Reporte final ────────────────────────────────────────────────
    final_count = len(list(save_dir.glob("*.*")))
    print(f"\n{'─'*55}")
    print(f"  📊 Imágenes descargadas : {final_count + removed}")
    print(f"  🗑️  Duplicados eliminados: {removed}")
    print(f"  ✅ Imágenes finales      : {final_count}")
    print(f"  📁 Guardadas en          : {save_dir.resolve()}")
    print(f"{'─'*55}\n")


# ── EJECUCIÓN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_celebrity(
        name=CELEBRITY_NAME,
        n_images=N_IMAGES,
        base_path=OUTPUT_PATH,
    )
