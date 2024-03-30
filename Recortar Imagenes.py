import cv2
import os

# Ruta de la carpeta con las imágenes
ruta_carpeta_imagenes = "D:\FACULTAD\MATERIAS PENDIENTES\92 Inteligencia Artificial\Proyecto\Entrenamiento"

# Ruta de la carpeta donde guardar los recortes
ruta_carpeta_recortes = "D:\FACULTAD\MATERIAS PENDIENTES\92 Inteligencia Artificial\Proyecto\Entrenamiento\Recortes"

# Comprueba si existe la carpeta de recortes y, si no, la crea
if not os.path.exists(ruta_carpeta_recortes):
    os.makedirs(ruta_carpeta_recortes)

# Obtiene una lista de todos los archivos en la carpeta de imágenes
archivos = os.listdir(ruta_carpeta_imagenes)

# Procesa cada archivo
for nombre_archivo in archivos:
    # Solo procesa archivos con extensión .jpeg
    if nombre_archivo.endswith('.jpg'):
        # Lee la imagen
        ruta_imagen = os.path.join(ruta_carpeta_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"No se pudo abrir la imagen {nombre_archivo}. Por favor, intenta de nuevo.")
            continue

        # Divide la imagen en 4 partes iguales verticalmente
        altura, ancho = imagen.shape[:2]
        imgs = [imagen[i*altura//4:(i+1)*altura//4, :] for i in range(4)]
        imgs_recortadas = []
        imgs_recortadas.append(imgs[0])
        imgs_recortadas.append(imgs[1])
        imgs_recortadas.append(imgs[2])
        imgs_recortadas.append(imgs[3])


        # Guarda cada recorte en una nueva carpeta
        for i, img_recortada in enumerate(imgs_recortadas):
            nombre_recorte = os.path.splitext(nombre_archivo)[0] + f"_recorte_{i+1}.jpeg"
            ruta_recorte = os.path.join(ruta_carpeta_recortes, nombre_recorte)
            cv2.imwrite(ruta_recorte, img_recortada)