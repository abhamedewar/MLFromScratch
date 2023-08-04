import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, K, epoch, plot_steps):
        self.K = K
        self.epoch = epoch
        self.plot_steps = plot_steps

        #list of list each list will store indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        #will store K centroids
        self.centroids = []

    def predict(self, X):
        self.X = X
        n_samples, n_features = X.shape
        #generate indices for initial centroids, replace is set to false so that we do not consider same element twice
        random_idx = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_idx]

        for _ in range(self.epoch):
            #update clusters
            self.clusters = self._create_clusters(self.centroids)
            #update centroids
            centroids_old = self.centroids
            self.centroids = self._update_centroids()
            #check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()
        #return cluster labels
        labels = [i for i in range(n_samples)]
        for idx, cluster in enumerate(self.clusters):
            for c in cluster:
                labels[c] = idx

        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):
            distances = [self.euclidean_distance(sample, c) for c in centroids]
            centroid_idx = np.argmin(distances)
            clusters[centroid_idx].append(idx)
        
        return clusters

    def _is_converged(self, old_centroids, new_centroids):
        dist = [self.euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.K)]

        return sum(dist) == 0

    def _update_centroids(self):
        centroids = []
        for c in self.clusters:
            cluster_mean = np.mean(self.X[c], axis=0)
            centroids.append(cluster_mean)

        return centroids

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2) ** 2))
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

# Testing
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, epoch=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()