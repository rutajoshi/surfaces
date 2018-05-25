import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import collections

def make_dataset(image):
    """
    We assume the image is a 2d matrix, with the depths at pixels (x,y).
    Returns an array with 3 columns (x, y, depth) and as many rows as pixels.
    """
    dataset = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            row = [x, y, image[x][y]]
            dataset.append(row)
    return np.array(dataset)

def cluster(image, k=4):
    # Make dataset
    # If you have a DepthImage, use image._data instead of making a dataset
    X = make_dataset(image)
    # Cluster data
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # Plot things
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()

    # Retrieve centroids
    return y_kmeans

def biggest_cluster(y_kmeans):
    counter = collections.Counter(y_kmeans)
    best_cluster = counter.most_common(1)[0][0]
    return best_cluster
