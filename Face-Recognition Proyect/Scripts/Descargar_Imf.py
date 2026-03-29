from icrawler.builtin import BingImageCrawler
import os
from pathlib import Path

# 1. Configuración de Rutas Dinámicas
# Subimos dos niveles desde 'Face-Recognition Proyect/Scripts/' para llegar a la raíz
ROOT_DIR = Path(__file__).resolve().parent.parent
grupo    = "Cantantes"
artista  = "Sakira Cantante"
# Ruta final: .../Face-Recognition Proyect/Dataset/Cantantes/Sabrina_Carpenter
save_path = str(ROOT_DIR / "Dataset" / grupo / artista)

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"✅ Directorio creado: {save_path}")

# 2. Configuración del Crawler (Usamos Bing para evitar bloqueos de Google)
# storage indica dónde se guardarán físicamente los archivos
bing_crawler = BingImageCrawler(storage={'root_dir': save_path})

# 3. Lista de búsquedas para obtener más variedad y cantidad
# Agregar términos como 'portrait' o 'face' ayuda a tu script de MTCNN
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