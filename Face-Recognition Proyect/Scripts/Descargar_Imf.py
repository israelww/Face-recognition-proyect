from bing_image_downloader import downloader
import os

# Ruta exacta a tu carpeta de Actores
ruta_dataset = "../Dataset/Actores/Cillian Murphy"

downloader.download(
    "Cillian Murphy face portrait", 
    limit=70, 
    output_dir="../Dataset/Actores", 
    adult_filter_off=True, 
    force_replace=False, 
    timeout=60,
    verbose=True
)