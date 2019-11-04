import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_K(X, k_values, plot_chart=False):
    scores = []
    print('Starting search...')
    for k in k_values:
        print(f'k = {k}')
        max_iter = 100
        kmeans = KMeans(k, max_iter=max_iter)
        labels = kmeans.fit_predict(X, plot_clusters=True, plot_step=max_iter)
        scores.append(silhouette_score(X, labels))

    if plot_chart:
        fig, ax = plt.subplots()
        ax.plot(k_values, scores)
        ax.set_title('Silhouette avg trend for K-Means with variable K value')
        plt.show()

    return k_values[np.argmax(scores)]


def plot_scatter(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, iter: int):
    fig, ax = plt.subplots()
    ax.set_title(f'Iteration {iter} with K={centroids.shape[0]}')
    # for i in range(centroids.shape[0]):
    #     idx = labels == i
    #     ax.scatter(X[idx, 0], X[idx, 1], label=i)
    ax.scatter(X[:, 0], X[:, 1], c=labels)

    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, c='red')
    plt.show()


def silhouette_samples(X: np.ndarray, labels: np.ndarray, plot_chart=False) -> np.ndarray:
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """
    n_clusters = np.unique(labels).size

    s = np.zeros(X.shape[0])  # samples silhouette array

    A = s.copy()  # samples similarity score array

    singletons = []

    for i in range(X.shape[0]):
        idx = np.arange(0, X.shape[0])

        # mask values are True if the label is the same of that of i AND if the index is not i
        mask = np.all(np.stack((labels == labels[i], idx != i)), axis=0)
        # sum of all True values gives the size of the cluster
        cl_i_size = mask.sum()
        if cl_i_size > 0:
            coeff = 1 / mask.sum()
            A[i] = coeff * np.sum(np.linalg.norm(X[mask, :] - X[i, :], axis=1), axis=0)
        else:
            singletons.append(i)

    B = s.copy()
    for i in range(X.shape[0]):
        l_i = labels[i]
        diss = np.zeros(n_clusters)
        others = [l for l in range(n_clusters) if l != l_i]
        for k in others:
            mask = labels == k
            cl_i_size = mask.sum()
            if cl_i_size != 0:
                coeff = 1 / cl_i_size
                diss[k] = coeff * np.sum(np.linalg.norm(X[mask, :] - X[i, :], axis=1), axis=0)
            else:
                others.remove(k)

        B[i] = diss[others].min()

    # compute s
    s = (B - A) / np.max(np.stack((A, B)), axis=0)
    s[singletons] = 0

    if plot_chart:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, X.shape[0]), np.sort(s))
        ax.set_title(f'Samples silhouettes in ascending order (K={n_clusters})')
        plt.show()

    return s


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Evaluate the silhouette for each point and return the mean.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float
    """
    return silhouette_samples(X, labels).mean()


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit_predict(self, X: np.ndarray, plot_clusters: bool = False, plot_step: int = 5) -> np.ndarray:
        """
        Run the K-means clustering on dataset X

        :param X: input data points, array, shape = (N, C)
        :param plot_clusters: boolean, if True the method plots the clusters every <plot_step> iterations
        :param plot_step: step for every scatter-plot
        :return labels: array, shape = N
        """
        # generate n_clusters random points as starting centroids
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        x_centroids = np.random.randint(min_vals[0], max_vals[0], size=self.n_clusters)
        y_centroids = np.random.randint(min_vals[1], max_vals[1], size=self.n_clusters)

        self.centroids = np.stack((x_centroids, y_centroids), axis=-1)
        dist = np.zeros((X.shape[0], self.n_clusters))

        changed = True
        it = 1

        while changed and it <= self.max_iter:
            # compute distances between each datapoint and each centroid
            for i in range(self.n_clusters):
                dist[:, i] = np.linalg.norm(X - self.centroids[i, :], axis=1)

            self.labels = np.argmin(dist, axis=1)

            # backup and update centroids
            old_centroids = self.centroids.copy()
            for i in range(self.n_clusters):
                mask = self.labels == i
                if np.any(mask):
                    self.centroids[i, :] = np.mean(X[mask], axis=0)
                else:
                    self.reset_centroid(i)

            changed = not np.allclose(self.centroids, old_centroids)
            if plot_clusters and (it % plot_step == 0 or not changed):
                plot_scatter(X, self.centroids, self.labels, it)

            it += 1

        return self.labels

    def dump_to_file(self, filename: str):
        """
        Dump the evaluated labels to a CSV file
        """
        s = pd.Series(self.labels)
        df = pd.DataFrame({'Id': s.index, 'ClusterId': s.values})
        df.to_csv(filename, index=False)

    def reset_centroid(self, i):
        """
        Re-initialize a centroid if its cluster is empty. It selects a random point within the current centroids area
        :param i: label of the centroid
        """
        min_vals = np.min(self.centroids[np.arange(0,self.n_clusters) != i, :], axis=0)
        max_vals = np.max(self.centroids[np.arange(0,self.n_clusters) != i, :], axis=0)
        self.centroids[i, :] = np.array([np.random.randint(min_vals[0], max_vals[0]), np.random.randint(min_vals[1], max_vals[1])])

