import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

ALTO_VENTANA, ANCHO_VENTANA = 640, 960
ANCHO_CAMARA = 240
MARGEN = 16
ARCHIVO = 'Imagen-algebra-2.webp'

imagen = cv2.imread(ARCHIVO)
if imagen is None:
    raise SystemExit(f'No se pudo abrir {ARCHIVO}. Convierte tu imagen a .png o .jpg si persiste.')

# Descargar modelo de manos si no existe localmente
MODELO_PATH = 'hand_landmarker.task'
if not os.path.exists(MODELO_PATH):
    print('Descargando modelo de detección de manos...')
    url_modelo = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url_modelo, MODELO_PATH)
    print('¡Modelo descargado con éxito!')

# Configurar MediaPipe Hands
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELO_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = HandLandmarker.create_from_options(options)

# Inicializar el detector de rostros de OpenCV (Ultrarrápido y nativo)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------------------------------------------------- funciones matematicas y efectos
def transformar(imagen, kx, ky):
    h, w = imagen.shape[:2]
    base = min(ALTO_VENTANA / h, ANCHO_VENTANA / w)
    
    M_lin = np.array([[base * kx, 0.0],
                      [0.0, base * ky]])
    c_img = np.array([w / 2, h / 2], dtype=np.float32)
    c_win = np.array([ANCHO_VENTANA / 2, ALTO_VENTANA / 2], dtype=np.float32)

    t = c_win - M_lin @ c_img
    M = np.hstack([M_lin, t.reshape(2, 1)])
    
    lienzo = cv2.warpAffine(imagen, M, (ANCHO_VENTANA, ANCHO_VENTANA),
                            flags=cv2.INTER_LINEAR, borderValue=(30, 30, 30))
    return lienzo, M

def pixelar_rostro(frame, faces, blocks=12):
    """
    Aplica efecto de mosaico (pixeleado) dinámico en las coordenadas de los rostros detectados.
    """
    h_img, w_img = frame.shape[:2]
    for (x, y, w, h) in faces:
        # Dar un pequeño margen para cubrir bien el rostro completo
        x_m = max(0, x - int(w * 0.1))
        y_m = max(0, y - int(h * 0.1))
        w_m = min(w_img - x_m, int(w * 1.2))
        h_m = min(h_img - y_m, int(h * 1.2))
        
        rostro = frame[y_m:y_m+h_m, x_m:x_m+w_m]
        if rostro.size == 0:
            continue
            
        # Reducir y reampliar la región con interpolación por vecino más cercano
        sw = max(1, w_m // blocks)
        sh = max(1, h_m // blocks)
        
        rostro_pequeno = cv2.resize(rostro, (sw, sh), interpolation=cv2.INTER_LINEAR)
        rostro_pixelado = cv2.resize(rostro_pequeno, (w_m, h_m), interpolation=cv2.INTER_NEAREST)
        
        frame[y_m:y_m+h_m, x_m:x_m+w_m] = rostro_pixelado
    return frame

def es_puno(hand_landmarks):
    muñeca = hand_landmarks[0]
    puntas = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]
    doblados = 0
    for p, b in zip(puntas, bases):
        if hand_landmarks[p].y > hand_landmarks[b].y:
            doblados += 1
    return doblados >= 3

def pegar_camara(lienzo, frame):
    fh, fw = frame.shape[:2]
    alto_cam = int(ANCHO_CAMARA * fh / fw)
    pequeno = cv2.resize(frame, (ANCHO_CAMARA, alto_cam))
    f1 = ALTO_VENTANA - alto_cam - MARGEN
    c1 = ANCHO_VENTANA - ANCHO_CAMARA - MARGEN
    lienzo[f1:f1 + alto_cam, c1:c1 + ANCHO_CAMARA] = pequeno
    cv2.rectangle(lienzo, (c1, f1), (c1 + ANCHO_CAMARA, f1 + alto_cam), (255, 255, 255), 2)

# ---------------------------------------------------------------- programa principal
if __name__ == '__main__':
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise SystemExit('No se pudo abrir la cámara.')
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    kx_actual, ky_actual = 1.0, 1.0
    suavizado = 0.2
    pixelar_activo = False  # Estado inicial: Apagado

    print('Controles activos:')
    print(' - 1 Mano abierta: Tamaño normal')
    print(' - 1 Puño cerrado: Achicar imagen')
    print(' - 2 Manos: Estirar ancho (X) y alto (Y)')
    print(' - Tecla P: Activar / Desactivar pixeleado del rostro')
    print(' - Tecla Q o ESC: Salir')

    while True:
        ok, frame = camara.read()
        if not ok:
            break
            
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        
        # 1. Detección de manos con MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resultado = detector.detect(mp_image)

        kx_objetivo, ky_objetivo = 1.0, 1.0
        estado_texto = "Esperando mano..."

        if resultado.hand_landmarks:
            num_manos = len(resultado.hand_landmarks)

            for hand_lms in resultado.hand_landmarks:
                for lm in hand_lms:
                    cx_lm, cy_lm = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame, (cx_lm, cy_lm), 4, (0, 255, 0), -1)

            if num_manos == 1:
                lms = resultado.hand_landmarks[0]
                if es_puno(lms):
                    kx_objetivo, ky_objetivo = 0.4, 0.4
                    estado_texto = "Gesto: Puño (Achicar)"
                else:
                    kx_objetivo, ky_objetivo = 1.0, 1.0
                    estado_texto = "Gesto: Mano Abierta (Normal)"

            elif num_manos >= 2:
                h1 = resultado.hand_landmarks[0][9]
                h2 = resultado.hand_landmarks[1][9]
                p1 = np.array([h1.x * W, h1.y * H])
                p2 = np.array([h2.x * W, h2.y * H])

                dist_x = abs(p1[0] - p2[0]) / W
                kx_objetivo = float(np.interp(dist_x, [0.15, 0.65], [0.5, 2.2]))

                dist_y = abs(p1[1] - p2[1]) / H
                ky_objetivo = float(np.interp(dist_y, [0.05, 0.50], [0.5, 2.2]))

                estado_texto = f"Dos manos | Ancho: {kx_objetivo:.2f} | Alto: {ky_objetivo:.2f}"

        # 2. Detección de rostros y aplicación del pixeleado en tiempo real
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if pixelar_activo:
            frame = pixelar_rostro(frame, rostros, blocks=14)

        # 3. Transformación suave de la imagen de álgebra
        kx_actual += (kx_objetivo - kx_actual) * suavizado
        ky_actual += (ky_objetivo - ky_actual) * suavizado

        ventana, M = transformar(imagen, kx_actual, ky_actual)
        pegar_camara(ventana, frame)

        # 4. Interfaz visual en pantalla
        cv2.putText(ventana, estado_texto, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(ventana, f'Escala X: {kx_actual:.2f} | Escala Y: {ky_actual:.2f}', (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        estado_pix = "ACTIVADO" if pixelar_activo else "DESACTIVADO"
        color_pix = (0, 255, 255) if pixelar_activo else (150, 150, 150)
        cv2.putText(ventana, f'Pixeleado [P]: {estado_pix}', (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pix, 1, cv2.LINE_AA)

        cv2.imshow('Imagen Interactiva con MediaPipe', ventana)
        cv2.imshow('Camara y Esqueletos', frame)

        # Control de teclado
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord('q'), 27):
            break
        elif tecla in (ord('p'), ord('P')):
            pixelar_activo = not pixelar_activo
            print(f"Pixeleado de rostro cambiado a: {'ACTIVADO' if pixelar_activo else 'DESACTIVADO'}")

    camara.release()
    cv2.destroyAllWindows()