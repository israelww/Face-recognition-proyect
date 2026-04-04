import os
import cv2
import time
import uuid

nombre_persona = 'Joey'
path_datos     = r'.\Face-Recognition Proyect\Dataset\Alumnos'
path_completo  = os.path.join(path_datos, nombre_persona)
META           = 120    
TAMANO         = (160, 160)  
INTERVALO_MIN  = 0.4     
 
if not os.path.exists(path_completo):
    os.makedirs(path_completo)
    print(f'Carpeta creada: {path_completo}')

try:
    from mtcnn import MTCNN
    detector_mtcnn = MTCNN()
    usar_mtcnn = True
    print("Detector: MTCNN ✓")
except ImportError:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    usar_mtcnn = False
    print("Detector: Haar Cascades (instala mtcnn para mejor calidad)")
 

VARIACIONES = [
    "Frente, expresion neutra",
    "Frente, sonriendo",
    "Cabeza inclinada derecha",
    "Cabeza inclinada izquierda",
    "Con lentes (si aplica)",
    "Con mascarilla",
    "Iluminacion lateral",
]
 
def variacion_actual(count):
    idx = min(count // (META // len(VARIACIONES)), len(VARIACIONES) - 1)
    return VARIACIONES[idx]
 

cap   = cv2.VideoCapture(0)
count = 0
ultimo_guardado = 0 
 
print(f"\nCapturando {META} fotos para: {nombre_persona}")
print("Presiona 'q' para salir antes de tiempo.\n")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)  
    ahora = time.time()
 
   
    rostro_recortado = None
 
    if usar_mtcnn:
        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = detector_mtcnn.detect_faces(frame_rgb)
        if resultados:
            det = max(resultados, key=lambda r: r['confidence'])
            if det['confidence'] > 0.88:
                x, y, w, h = det['box']
                x, y = max(0, x), max(0, y)
                
                margen = int(0.20 * max(w, h))
                x1 = max(0, x - margen)
                y1 = max(0, y - margen)
                x2 = min(frame.shape[1], x + w + margen)
                y2 = min(frame.shape[0], y + h + margen)
                rostro_recortado = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
    else:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
           
            margen = int(0.20 * max(w, h))
            x1 = max(0, x - margen)
            y1 = max(0, y - margen)
            x2 = min(frame.shape[1], x + w + margen)
            y2 = min(frame.shape[0], y + h + margen)
            rostro_recortado = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
 
   
    if rostro_recortado is not None and (ahora - ultimo_guardado) >= INTERVALO_MIN:
        
        rostro_160 = cv2.resize(rostro_recortado, TAMANO, interpolation=cv2.INTER_LANCZOS4)
        nombre_archivo = f'rostro_{count:04d}_{uuid.uuid4().hex[:5]}.jpg'
        cv2.imwrite(os.path.join(path_completo, nombre_archivo), rostro_160)
        count += 1
        ultimo_guardado = ahora
 
   
    cv2.putText(frame, f'{nombre_persona}  {count}/{META}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, variacion_actual(count),
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 230, 255), 1)
 
   
    prog = int((count / META) * frame.shape[1])
    cv2.rectangle(frame, (0, frame.shape[0] - 7), (prog, frame.shape[0]),
                  (0, 210, 80), -1)
 
    cv2.imshow('Creando Dataset', frame)
 
    k = cv2.waitKey(1)
    if k == ord('q') or count >= META:
        break
 
print(f"\nDataset finalizado. {count} imágenes guardadas en '{path_completo}'")
cap.release()
cv2.destroyAllWindows()