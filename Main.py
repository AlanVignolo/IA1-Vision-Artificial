from Preprocesamiento import Preprocesamiento
import cv2

import warnings
warnings.filterwarnings("ignore")

def main():
    preprocessor = Preprocesamiento()

    while True:
        print("\n----------------------------------")
        print("1. Preprocesar y guardar datos.")
        print("2. Cargar datos existentes.")
        print("3. Clasificar imagen con Kmeans.")
        print("4. Clasificar imagen con KNN.")
        print("5. Clasificar una imagen compuesta con KMeans o KNN.")
        print("6. Salir.")
        print("----------------------------------")

        opcion = int(input("Ingrese el número de la opción que desea realizar: "))
   
        if opcion == 1:
            preprocessor.preprocesar_datos()
            print("Los datos han sido preprocesados y guardados con éxito.")

        elif opcion == 2:
            preprocessor.cargar_datos()
            print("Los datos han sido cargados con éxito.")

        elif opcion == 3:
            imagen_path = "D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\TUERCA.jpeg"
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                print("No se pudo abrir la imagen. Por favor, intenta de nuevo.")
                continue
            preprocessor.clasificar_imagen(imagen, 'kmeans')

        elif opcion == 4:

            imagen_path = "D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\TUERCA.jpeg"
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                print("No se pudo abrir la imagen. Por favor, intenta de nuevo.")
                continue
            preprocessor.clasificar_imagen(imagen, 'knn')

        elif opcion == 5:
            metodo = input("Ingrese el método a usar (kmeans/knn): ")

            imagen_path = "D:\\FACULTAD\\MATERIAS PENDIENTES\\92 Inteligencia Artificial\\Proyecto\\Base de datos Test\\FINAL.jpg"
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                print("No se pudo abrir la imagen. Por favor, intenta de nuevo.")
                continue
            predicciones = preprocessor.Preprocesar_multiple(imagen, metodo)

            print("\nPredicciones hog: ")
            for pred in [predicciones[i][0] for i in range(4)]:
                print(pred)

            print("\nPredicciones hu: ")
            for pred in [predicciones[i][1] for i in range(4)]:
                print(pred)

        elif opcion == 6:
            print("Saliendo...")
            break

if __name__ == "__main__":
    main()