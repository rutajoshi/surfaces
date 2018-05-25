%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

def make_dataset(image):
    """
    We assume the image is a 2d matrix, with the depths at pixels (x,y).
    Returns an array with 3 columns (x, y, depth) and as many rows as pixels.
    """
    dataset = []
    num_rows = image.shape[0] * image.shape[1]
    for x in image.shape[0]:
        for y in image.shape[1]:
            row = [x, y, image[x][y]]
            dataset.append(row)
    return np.array(dataset)

def cluster(image, k=4):
    # Make dataset
    X = make_dataset(image)

    # Cluster data
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # Retrieve centroids
    return y_kmeans
