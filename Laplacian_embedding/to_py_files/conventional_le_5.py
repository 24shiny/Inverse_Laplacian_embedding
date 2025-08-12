import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding as se

class Conventional_LE:
    """Laplacian embedding with the use of SpectralEmbedding in sklearn"""
    def __init__(self, X, n_centers):
        """initialize variables"""
        self.X = X # data
        self.n_centers = n_centers
        self.Y = None # label estimated by KMeans
        self.emb_X = None # data embedded on LE
        self.score = -1
    
    def embed_data(self):
        A = np.exp(-pairwise_distances(self.X, metric='euclidean'))
        emb = se(n_components=4, affinity='precomputed', random_state=42)
        self.emb_X = emb.fit_transform(A)
        self.get_score()
        self.plot_embedded_X()
        
    def get_score(self): # ss or ard
        km = KMeans(n_clusters=self.n_centers, random_state=42).fit(self.X)
        self.Y = km.fit_predict(self.X)
        self.score = ss(self.X, self.Y)
    
    def plot_embedded_X(self): 
        plt.scatter(self.emb_X[:, 0], self.emb_X[:, 1], c=self.Y, cmap='viridis', s=30, alpha=0.8)
        plt.title(f'Data on LE with score {self.score}')
        plt.show()
        
        