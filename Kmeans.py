import numpy as np

class KMeans:
    def __init__(self, n_clusters=4):

        self.n_clusters = n_clusters
        self.centroides_ = None
        self.labels_ = None
        self.clusters = None

    def fit(self, X):

        self.centroides_ = self._kmeans_plusplus(X)
        self.centroides_ = [X[4], X[12], X[20], X[28]]
        prev_centroids = np.zeros_like(self.centroides_)

        while np.linalg.norm(self.centroides_ - prev_centroids) > 1e-4: 
            self.clusters = np.argmin([np.linalg.norm(X - c, axis=1) for c in self.centroides_], axis=0) # Asigna cada punto al cluster más cercano
            prev_centroids = self.centroides_.copy()

            self.centroides_ = np.array([X[self.clusters == k].mean(axis=0) for k in range(self.n_clusters)]) # Recalcula los centroides de cada cluster
        
        print(self.clusters)
        return self

    def _kmeans_plusplus(self, X):
        n, _ = X.shape # n = número de puntos
        centroides = np.zeros((self.n_clusters, X.shape[1])) # Inicializa los centroides
        initial_idx = np.random.choice(n) # Elige un punto al azar
        centroides[0] = X[initial_idx] # El primer centroide es el punto elegido al azar
        for i in range(1, self.n_clusters):
            dist_sq = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroides[:i]], axis=0) # Calcula la distancia al cuadrado de cada punto al centroide más cercano
            probs = dist_sq / np.sum(dist_sq) # Calcula las probabilidades de elegir cada punto como centroide
            cumulative_probs = np.cumsum(probs) # Calcula las probabilidades acumuladas
            idx = np.where(cumulative_probs >= np.random.rand())[0][0] # Elige un punto al azar según las probabilidades acumuladas
            centroides[i] = X[idx] # Elige el punto correspondiente al índice elegido al azar
        return centroides 