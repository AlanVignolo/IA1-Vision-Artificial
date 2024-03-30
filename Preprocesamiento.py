import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
from Knn import Knn
from Kmeans import KMeans
from Hu_Moments import Hu_Moments

class Preprocesamiento:
    def __init__(self): 

        self.Carpeta = 'D:\FACULTAD\MATERIAS PENDIENTES\92 Inteligencia Artificial\Proyecto\Base de datos'

        self.Categorias = ['Tornillos', 'Tuercas', 'Clavos', 'Arandelas']
        self.Etiquetas = ['Tornillos'] * 8 + ['Tuercas'] * 8 + ['Clavos'] * 8 + ['Arandelas'] * 8
        
        self.caracteristicas_hog = []
        self.caracteristicas_hu = []

        self.models_kmeans = []
        self.models_knn = []

    def preprocesar_datos(self):

        self.caracteristicas_hog = []
        self.caracteristicas_hu = []

        self.models_kmeans = [] 
        self.models_knn = []

        kernel = np.ones((5,5),np.uint8) # Matriz de 1's de 5x5 para usar en la operación de dilatación. 

        for categoria in self.Categorias:
            Subcarpeta = os.path.join(self.Carpeta, categoria)
            for NombreImagen in os.listdir(Subcarpeta):

                print("Procesando imagen: ", NombreImagen)
                imag = cv2.imread(os.path.join(Subcarpeta, NombreImagen))

                imag = imag[20:-20,20:-20]

                imag = cv2.resize(imag, (512, 312))
                img_blur = cv2.GaussianBlur(imag, (5, 5), 0) # Aplicar un filtro Gaussiano para eliminar el ruido.

                img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY) # Convertir la imagen a escala de grises.

                img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV) # Convertir la imagen a HSV: Matiz, Saturación y Valor.

                lower_range = np.array([140, 50, 50]) # Rango inferior de los colores a detectar.
                upper_range = np.array([160, 255, 255]) # Rango superior de los colores a detectar.

                mask = cv2.inRange(img_hsv, lower_range, upper_range) # Crear una máscara con los colores detectados.

                mask = cv2.bitwise_not(mask) # Invertir la máscara. 

                img_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask) # Aplicar la máscara a la imagen.

                img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel) # Aplicar una operación morfológica de cierre. 

                contours, _ = cv2.findContours(img_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontrar los contornos de la imagen. 

                img_new = np.ones(img_gray.shape, dtype=np.uint8)*255 # Crear una imagen en blanco.

                cv2.drawContours(img_new, contours, -1, (0), thickness=cv2.FILLED) # Dibujar los contornos en la imagen en blanco.

                M = cv2.moments(contours[0]) # Calcular los momentos de la imagen.
                try: 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                except:
                    try:
                        M = cv2.moments(contours[1])
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    except:
                        M = cv2.moments(contours[2])
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                height, width = img_new.shape[:2] # Obtener el alto y ancho de la imagen.
                dX = (width // 2) - cX # Calcular la distancia en X entre el centro de la imagen y el centroide del objeto.
                dY = (height // 2) - cY # Calcular la distancia en Y entre el centro de la imagen y el centroide del objeto.

                T = np.float32([[1, 0, dX], [0, 1, dY]]) # Crear una matriz de transformación.
                img_new = cv2.warpAffine(img_new, T, (width, height), borderValue = 255) # Aplicar la matriz de transformación a la imagen.

                moments = cv2.moments(img_new)
                hu_moments = Hu_Moments(moments)

                bounding_rect = cv2.boundingRect(contours[0]) # Obtener el rectángulo delimitador del objeto detectado.
                imag = imag[bounding_rect[1]:bounding_rect[1]+bounding_rect[3], bounding_rect[0]:bounding_rect[0]+bounding_rect[2]] # Recortar la imagen con el rectángulo delimitador.
                imag = cv2.resize(imag, (128, 128), interpolation=cv2.INTER_LINEAR) # Redimensionar la imagen a 128x128 píxeles.

                img_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY) # Convertir la imagen a escala de grises.
                hog_caracteristicas = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

                self.caracteristicas_hog.append(hog_caracteristicas)
                self.caracteristicas_hu.append(hu_moments)

                NombreImagen = "\ " + NombreImagen + ".jpeg"
                direccion1 = "D:\FACULTAD\MATERIAS PENDIENTES\92 Inteligencia Artificial\Proyecto\Fotos viejas" + NombreImagen
                cv2.imwrite(direccion1, img_new)
                
        self.caracteristicas_hog = np.array(self.caracteristicas_hog)
        self.caracteristicas_hu = np.array(self.caracteristicas_hu)

        self.models_kmeans = []
        self.models_knn = []

        for i, caracteristicas in enumerate([self.caracteristicas_hog, self.caracteristicas_hu]): # Crear un modelo de K-Means y un modelo de K-NN para cada tipo de característica.

            caracteristicas = np.array(caracteristicas)
            model_kmeans = KMeans()
            model_kmeans.fit(caracteristicas)
            
            nombres_centroides = []
            
            for centroide in model_kmeans.centroides_: # Para cada centroide.

                distancias = [np.linalg.norm(centroide - punto) for punto in caracteristicas] # Calcular la distancia entre el centroide y cada punto.
                indices_mas_cercanos = np.argsort(distancias)[:5] # Obtener los índices de los 5 puntos más cercanos al centroide.
                etiquetas_mas_cercanas = [self.Etiquetas[i] for i in indices_mas_cercanos] # Obtener el nombre de los 5 puntos más cercanos al centroide.
                nombre_centroide = max(set(etiquetas_mas_cercanas), key=etiquetas_mas_cercanas.count) # Obtener el nombre más frecuente entre los 5 puntos más cercanos al centroide.
                nombres_centroides.append(nombre_centroide) 

            model_kmeans.labels_ = nombres_centroides

            model_knn = Knn()
            model_knn.fit(list(zip(caracteristicas, self.Etiquetas)))

            self.models_kmeans.append(model_kmeans)
            self.models_knn.append(model_knn)

        file_paths = ['knn_models.p','kmeans_models.p', 'caracteristicas_hog.p', 'caracteristicas_hu.p']

        
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Guardar los modelos y las características en archivos .p. (Serialización de objetos)
        with open('knn_models.p', 'wb') as f:
            pickle.dump(self.models_knn, f)
        with open('kmeans_models.p', 'wb') as f:
            pickle.dump(self.models_kmeans, f)
        with open('caracteristicas_hog.p', 'wb') as f:
            pickle.dump(self.caracteristicas_hog, f)
        with open('caracteristicas_hu.p', 'wb') as f:
            pickle.dump(self.caracteristicas_hu, f)
    
    def cargar_datos(self):

        self.caracteristicas_hog = []
        self.caracteristicas_hu = []
        self.models_kmeans = []
        self.models_knn = []
        with open('caracteristicas_hog.p', 'rb') as f:
            self.caracteristicas_hog = pickle.load(f)
        with open('caracteristicas_hu.p', 'rb') as f:
            self.caracteristicas_hu = pickle.load(f)
        with open('kmeans_models.p', 'rb') as f:
            self.models_kmeans = pickle.load(f)
        with open('knn_models.p', 'rb') as f:
            self.models_knn = pickle.load(f)    

    def clasificar_imagen(self, imag, method):

        hog_caracteristicas, hu_moments = self.preprocesamiento_simple(imag)

        colors = ['b', 'g', 'r', 'c','m']
        resultados = []

        if method == 'kmeans':
            for i, (caracteristicas_nuevas, caracteristicas_existentes, metodo) in enumerate(zip([hog_caracteristicas, hu_moments], [self.caracteristicas_hog, self.caracteristicas_hu], ['HOG', 'Hu Moments'])):
                        
                distancias = [np.linalg.norm(caracteristicas_nuevas - centroide) for centroide in (self.models_kmeans[i]).centroides_] # Calcular la distancia euclidiana entre el centroide y cada punto.
                indice_centroide_cercano = np.argmin(distancias) # Obtener el índice del centroide más cercano.
                caracteristicas_nuevas_nombre = (self.models_kmeans[i]).labels_[indice_centroide_cercano] # Obtener el nombre del centroide más cercano.
                resultados.append(caracteristicas_nuevas_nombre) # Agregar el nombre del centroide más cercano a la lista de resultados.
                        
                pca = PCA(n_components=3)
                pca.fit(np.array(caracteristicas_existentes))

                reduced_caracteristicas = pca.transform(caracteristicas_existentes) 
                reduced_caracteristicas_nuevas = pca.transform(caracteristicas_nuevas.reshape(1, -1))
                reduced_centroides = pca.transform((self.models_kmeans[i]).centroides_)

                fig = plt.figure(figsize=(10, 6)) # Crear una figura.
                ax = fig.add_subplot(111, projection='3d') # Crear un subplot 3D.

                for j, (centroid, color) in enumerate(zip(reduced_centroides, colors)):

                    points = reduced_caracteristicas[(self.models_kmeans[i]).clusters == j] # Obtener los puntos que pertenecen al cluster j.
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.5) # Graficar los puntos del cluster j.
                    ax.scatter(centroid[0], centroid[1], centroid[2], color='k', marker='x', s=100) # Graficar el centroide del cluster j.

                ax.scatter(reduced_caracteristicas_nuevas[0, 0], reduced_caracteristicas_nuevas[0, 1], reduced_caracteristicas_nuevas[0, 2], color='magenta', marker='*', s=200, label=f"Predicción: {caracteristicas_nuevas_nombre}") # Graficar el punto a clasificar.
                plt.title(f"Kmeans para {metodo}")
                plt.legend()
                plt.show()

            return resultados

        elif method == 'knn':
            for i, (caracteristicas_nuevas, caracteristicas_existentes, metodo) in enumerate(zip([hog_caracteristicas, hu_moments],[self.caracteristicas_hog, self.caracteristicas_hu], ['HOG', 'Hu Moments'])):

                label_asignado = (self.models_knn[i]).predict(caracteristicas_nuevas) # Obtener el nombre de la categoría a la que pertenece el punto.
                resultados.append(label_asignado) 

                caracteristicas_existentes_np = np.array(caracteristicas_existentes)  

                pca = PCA(n_components=3)
                pca.fit(caracteristicas_existentes_np)

                reduced_caracteristicas = pca.transform(caracteristicas_existentes_np) 
                reduced_caracteristicas_nuevas = pca.transform(caracteristicas_nuevas.reshape(1, -1)) 

                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')

                cantidad_puntos = 8 # Cantidad de puntos por categoría.

                inicio = 0
                for color, categoria in zip(colors, self.Categorias):
                    puntos = reduced_caracteristicas[inicio: inicio + cantidad_puntos] # Obtener los puntos de la categoría.
                    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], alpha=0.5, color=color, label=categoria) # Graficar los puntos de la categoría.
                    inicio += cantidad_puntos 

                for label in set(self.Etiquetas[inicio:]):
                    points = reduced_caracteristicas[self.Etiquetas[inicio:] == label] # Obtener los puntos que pertenecen al cluster j.
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5, label=label) # Graficar los puntos del cluster j.

                ax.scatter(reduced_caracteristicas_nuevas[0][0], reduced_caracteristicas_nuevas[0][1], reduced_caracteristicas_nuevas[0][2], color='magenta', marker='*', s=200, label=f"Predicción: {label_asignado}")
                plt.title(f"KNN para {metodo}")
                plt.legend()
                plt.show()

            return resultados


    def preprocesamiento_simple(self, imag):

        kernel = np.ones((5,5),np.uint8)
        imag = imag[20:-20,20:-20]

        imag = cv2.resize(imag, (512, 312))
        imag_blur = cv2.GaussianBlur(imag, (5, 5), 0)   
        imag_gray = cv2.cvtColor(imag_blur, cv2.COLOR_BGR2GRAY)
                
        img_hsv = cv2.cvtColor(imag_blur, cv2.COLOR_BGR2HSV)
                
        lower_range = np.array([140, 50, 50])
        upper_range = np.array([160, 255, 255])

        mask = cv2.inRange(img_hsv, lower_range, upper_range)
        mask = cv2.bitwise_not(mask)
        img_masked = cv2.bitwise_and(imag_gray, imag_gray, mask=mask)
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(img_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_new = np.ones(imag_gray.shape, dtype=np.uint8)*255
        cv2.drawContours(img_new, contours, -1, (0), thickness=cv2.FILLED)
        img_new = cv2.dilate(img_new, kernel, iterations = 1)
        img_new = cv2.erode(img_new, kernel, iterations = 1)
        M = cv2.moments(contours[0])

        try:
            cX = int(M["m10"] / M["m00"])
        except:
            try:
                M = cv2.moments(contours[1])
                cX = int(M["m10"] / M["m00"])
            except:
                M = cv2.moments(contours[2])
                cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        height, width = img_new.shape[:2]
        dX = (width // 2) - cX
        dY = (height // 2) - cY

        T = np.float32([[1, 0, dX], [0, 1, dY]])
        img_new = cv2.warpAffine(img_new, T, (width, height), borderValue = 255)

        moments = cv2.moments(img_new)
        hu_moments = Hu_Moments(moments)


        bounding_rect = cv2.boundingRect(contours[0])
        imag = imag[bounding_rect[1]:bounding_rect[1]+bounding_rect[3], bounding_rect[0]:bounding_rect[0]+bounding_rect[2]]
        imag = cv2.resize(imag, (128, 128), interpolation=cv2.INTER_LINEAR)

        imag_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        hog_caracteristicas = hog(imag_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

        return np.array(hog_caracteristicas), np.array(hu_moments)

    def Preprocesar_multiple(self, imagen,metodo):

        altura, ancho = imagen.shape[:2]

        imagen_ver = cv2.resize(imagen, (400, 600))
        cv2.imshow(f"Foto a predecir", imagen_ver)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        imgs = [imagen[i*altura//4:(i+1)*altura//4, :] for i in range(4)]
        imgs_recortadas = []
        imgs_recortadas.append(imgs[0][10:-10,10:-10])
        imgs_recortadas.append(imgs[1][10:-10,10:-10])
        imgs_recortadas.append(imgs[2][10:-10,10:-10])
        imgs_recortadas.append(imgs[3][10:-10,10:-10])

        cv2.imwrite("D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\Caja-1.jpeg", imgs_recortadas[0])
        cv2.imwrite("D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\Caja-2.jpeg", imgs_recortadas[1])
        cv2.imwrite("D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\Caja-3.jpeg", imgs_recortadas[2])
        cv2.imwrite("D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\Caja-4.jpeg", imgs_recortadas[3])
        
        predicciones = []
        i=1
        for imag in imgs_recortadas:
            
            cv2.imshow(f"Caja {i}", imag)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            predicciones.append(self.clasificar_imagen(imag, metodo))
            i+=1
        return predicciones
