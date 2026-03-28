import os
import cv2

# 1. Configuración inicial
nombre_persona = 'MarioRios'
path_datos = 'dataset' # Carpeta donde se guardarán las fotos
path_completo = os.path.join(path_datos, nombre_persona)

if not os.path.exists(path_completo):
    print(f'Carpeta creada: {path_completo}')
    os.makedirs(path_completo)

# 2. Cargar el clasificador de rostros (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Iniciar la captura de video
cap = cv2.VideoCapture(0)
count = 0

print("Capturando rostro... Presiona 'q' para salir antes de tiempo.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Convertir a escala de grises para que el algoritmo trabaje mejor
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo en el rostro (opcional, para visualización)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Recortar el rostro y redimensionarlo (ej. 60x60 píxeles)
        rostro_recortado = frame[y:y + h, x:x + w]
        rostro_recortado = cv2.resize(rostro_recortado, (60, 60), interpolation=cv2.INTER_CUBIC)
        
        # Guardar la imagen
        cv2.imwrite(f'{path_completo}/rostro_{count}.jpg', rostro_recortado)
        count += 1

    # Mostrar el video en vivo
    cv2.imshow('Creando Dataset', frame)

    # Detener si llegamos a 100 fotos o si presionamos 'q'
    k = cv2.waitKey(1)
    if k == ord('q') or count >= 100:
        break

print(f"Dataset finalizado. Se guardaron {count} imágenes en {path_completo}")
cap.release()
cv2.destroyAllWindows()