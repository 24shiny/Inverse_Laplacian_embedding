import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.metrics import silhouette_score as ss
from sklearn.cluster import KMeans
from itertools import combinations as cb

class My_LE:
    """Laplacian embedding with inverse power method and deflation"""
    def __init__(self):
        """Initialize variables w.r.t an example dataset:make_blobs"""
        X, Y = datasets.make_blobs(n_samples=256, centers=4, random_state=42)
        self.X = X # data
        self.n_centers = 4
        self.Y = Y 
        self.eval_metric = 'ss'
        self.score = -1
        self.score_list = []
        self.pair = None
        self.def_val = []
        self.def_vec = []
        # example results
        self.plot_example_data()
        self.embed_data()
        self.Y = None

    def plot_example_data(self):
        """"Plot the example data:make_blobs"""
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='viridis', s=30, alpha=0.8)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Original data')
        plt.show()        
    
    def set_data(self, X, n_centers):
        """Updata data"""
        self.X = X
        self.n_centers = n_centers

    def set_label(self, Y):
        """Update labels"""
        self.Y = Y

    @staticmethod
    def power_method(A):
        """Return a dominant eigenpair"""
        num_iter=200
        tol=1e-6
        n = A.shape[0]
        x = np.ones(n)

        for _ in range(num_iter):
            x = A @ x
            x_norm = np.linalg.norm(x)
            x_next = x / x_norm
    
            # stopping criterion
            if np.linalg.norm(x_next - x) < tol:
                break
            x = x_next
    
        eig_val = (x.T @ (A @ x)) / (x.T @ x)
        eig_vec = x
        return eig_val, eig_vec
        
    @staticmethod
    def deflation(A, eig_val, eig_vec):
        """Shift the eigenspace, while removing the effect of the eigenpair"""
        return A - eig_val * np.outer(eig_vec, eig_vec)
        
    def core_routine(self):
        """Create matricies A, D, L and L inverse and Perform power and deflation method"""
        Adj = np.exp(-pairwise_distances(self.X, metric='euclidean'))
        D = np.diag(np.sum(Adj, axis=1))
        L = D - Adj
        L_inv = np.linalg.inv(L + np.eye(L.shape[0]) * 1e-6)
        L_inv_norm = normalize(L_inv)
    
        # power and deflation methods
        def_mat = L_inv

        def_val_temp = []
        def_vec_temp = []
        
        for i in range(self.n_centers):  
            eig_val, eig_vec = self.power_method(def_mat)
            def_val_temp.append(eig_val)
            def_vec_temp.append(eig_vec)
            def_mat = self.deflation(def_mat, eig_val, eig_vec)

        # update list
        self.def_val = def_val_temp
        self.def_vec = def_vec_temp
    
    # @staticmethod
    def get_score(self, x): # ss or ard
        """Calculate the Silhouette_score w.r.t data embedded on LE"""
        km = KMeans(n_clusters=self.n_centers, random_state=42).fit(x)
        label_predicted = km.fit_predict(x)
        try:
            if self.eval_metric == 'ss':
                return ss(x, label_predicted), label_predicted
            elif self.eval_metric == 'ars':
                return ars(self.Y, label_predicted), label_predicted
        except:
            print('Set eval_metric as either ss or ars')
    
    def get_best_le(self):
        """Find the best combination of eigenvectors of L inverse"""
        elem = range(len(self.def_vec))
        all_cb = list(cb(elem,2))
        score_list_temp = []
        label_list_temp = []
        for pair in all_cb:
            temp_data = np.stack((self.def_vec[pair[0]], self.def_vec[pair[1]]), axis=1)
            temp_score, temp_label = self.get_score(temp_data)
            score_list_temp.append(temp_score)
            label_list_temp.append(temp_label)
        best_idx = np.argmax(score_list_temp)
        self.pair = all_cb[best_idx]
        self.score = np.round(score_list_temp[best_idx],3)
        self.score_list.append(self.score)
        self.Y = label_list_temp[best_idx]
   
    def plot_embedded_X(self):
        """Plot data embedded on LE without labels"""
        i = self.pair[0]
        j = self.pair[1]
        if self.Y is None:
            plt.scatter(self.def_vec[i], self.def_vec[j], s=30, alpha=0.8) # w/o labels colored
        else:
            plt.scatter(self.def_vec[i], self.def_vec[j], c=self.Y, s=30, alpha=0.8) # w/ labels colored
        plt.title(f'Data on LE with Silhouette score {self.score}')
        plt.show()
        
    def embed_data(self):
        """Execute major routines"""
        self.core_routine() # def_vec, def_val
        self.get_best_le() # pair, score, Y
        # self.plot_embedded_X() 