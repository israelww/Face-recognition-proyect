from icrawler.builtin import BingImageCrawler
import os
from pathlib import Path

# 1. Configuración de Rutas Dinámicas
ROOT_DIR = Path(__file__).resolve().parent.parent
grupo    = "Cantantes"
artista  = "Sakira Cantante"
save_path = str(ROOT_DIR / "Dataset" / grupo / artista)

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"✅ Directorio creado: {save_path}")

# 2. Configuración del Crawler
bing_crawler = BingImageCrawler(storage={'root_dir': save_path})

# 3. Lista de búsquedas para obtener más variedad y cantidad
busquedas = [
    f"{artista} face portrait",
    f"{artista} headshot",
    f"{artista} closeup face",
    f"{artista} photoshoot face",
    f"{artista} interview 2026"
]

print(f"🚀 Iniciando descarga de imágenes para: {artista}...")

# 4. Bucle de descarga
for query in busquedas:
    print(f"🔍 Buscando: {query}")
    # max_num=30 por cada búsqueda para no saturar y evitar bloqueos
    bing_crawler.crawl(keyword=query, max_num=30, overwrite=False)

print(f"\n✨ Proceso finalizado. Revisa la carpeta: {save_path}")