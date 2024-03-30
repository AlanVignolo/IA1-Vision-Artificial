import numpy as np
from collections import Counter

class Knn:
    def __init__(self):
        self.k = 5

    def fit(self, data):
        self.data = data

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2)) # Distancia euclidiana

    def predict(self, img):
        y_pred = self._predict(img)
        return y_pred

    def _predict(self, img):
        distances = [self.euclidean_distance(img, data[0]) for data in self.data] # Calcula la distancia entre la imagen y cada punto del dataset
        k_indices = np.argsort(distances)[:self.k] # Ordena las distancias de menor a mayor y elige los índices de los k puntos más cercanos
        k_nearest_labels = [self.data[i][1] for i in k_indices] # Obtiene las etiquetas de los k puntos más cercanos
        most_common = Counter(k_nearest_labels).most_common(1) # Obtiene la etiqueta más común
        return most_common[0][0] # Devuelve la etiqueta más común