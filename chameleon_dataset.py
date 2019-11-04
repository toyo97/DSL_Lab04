import numpy as np
from clustering import KMeans, silhouette_samples, silhouette_score, find_K
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'output/ch_output.csv')

# load the dataset using numpy.loadtxt
with open('./input/chameleon_clusters.txt', 'r') as f:
    ds_arr = np.loadtxt(f, skiprows=1, delimiter=',')

kmeans = KMeans(6, max_iter=100)
predicts = kmeans.fit_predict(ds_arr)
kmeans.dump_to_file(filename)

s_samples = silhouette_samples(ds_arr, predicts, plot_chart=True)
print(f'Silhouette samples score: {s_samples}')

score = s_samples.mean()
print(f'Silhouette avg score: {score}')

k_values = list(range(2, 30))
print(f'Best k value: {find_K(ds_arr, k_values, plot_chart=True)}')

