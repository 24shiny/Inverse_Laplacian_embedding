import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh
import pennylane as qml
from itertools import product
from scipy.linalg import expm
from itertools import combinations as cb
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.cluster import KMeans

class My_quantum_LE:
    """Quantum Laplacian embedding with inverse power method and deflation"""
    def __init__(self):
        """Initialize variables w.r.t an example dataset:make_blobs"""
        X, Y = datasets.make_blobs(n_samples=16, centers=4, random_state=42)
        self.X = X # data
        self.n_centers = 4
        self.Y = Y 
        self.score = -1
        self.pair = None
        self.def_val = []
        self.def_vec = []
        
        # variables assoicated with quantum circuits
        self.def_mat = None
        self.coeff = None
        self.op = None
        self.n_control = None
        self.n_target = None
        self.n_tot = None
        self.scaling_factor = None
        self.dev = None
        self.qnode = None
        self.reg = None
        self.op = None
        self.proj_norm_coeffs = []

        # initialize quantum circuits
        self.calculate_L_inverse()
        self.update_circuit_config()

        # example results
        self.plot_example_data()
        self.embed_data()

    def power_circuit(self, iter_num, input_coeff):
        """Quantum circuits to implement the power method"""
        self.init_state(iter_num, input_coeff)
        self.block_encoding()
        return qml.state()

    def run_power_circuit(self, iter_num, input_coeff):
        """Run the power circuit defined above"""
        if self.qnode is None:
            raise RuntimeError("qnot has not yet been initialized.")
        return self.qnode(iter_num, input_coeff)
        
    def plot_example_data(self):
        """"Plot the example data:make_blobs"""
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='viridis', s=30, alpha=0.8)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Original data')
        plt.show()        
    
    def get_data(self, X, n_centers):
        """Updata data"""
        self.X = X
        self.n_centers = n_centers

    def get_label(self, Y):
        """Update labels"""
        self.Y = Y
        self.label_flag = 1

    def calculate_L_inverse(self):
        """Create matricies A, D, L and L inverse"""
        Adj = np.exp(-pairwise_distances(self.X, metric='euclidean'))
        D = np.diag(np.sum(Adj, axis=1))
        L = D - Adj
        L_inv = np.linalg.inv(L + np.eye(L.shape[0]) * 1e-6)
        L_inv_norm = L_inv / np.linalg.norm(L_inv)
        self.def_mat = L_inv_norm
        
    def update_circuit_config(self):
        """Update quantum circuit configuration"""
        self.lcu() # coeff and op updated
        self.init_config() # n_control and n_target
        self.scaling_factor = sum(abs(self.coeff))
        self.n_tot = self.n_control+self.n_target+1
        self.dev = qml.device("default.qubit", self.n_tot)
        self.qnode = qml.QNode(self.power_circuit, self.dev)
        self.reg = qml.registers({"control": self.n_control, "target": self.n_target, "super_work":1})
        self.op = [qml.map_wires(op_elem, {i: i + self.n_control for i in range(self.n_target)}) for op_elem in self.op]
        
    @staticmethod
    def check_divisibility(a, b):
        if a % b == 0:
            return 1
        else:
            return 0   
    
    def lcu(self):
        """Pauli decomposition of a given matrix"""
        LCU = qml.pauli_decompose(self.def_mat)
        coeff, op = LCU.terms()
        N = int(np.power(2,np.floor(math.log2(len(coeff)))))
        if self.check_divisibility(len(coeff), N) == 0:
            coeff, op = self.lcu_reduction(coeff, op, N) # variables replaced
        coeff_sqrt = np.sqrt([abs(i) for i in coeff])
        norm_coeff = coeff_sqrt / np.linalg.norm(coeff_sqrt)
        self.coeff = norm_coeff
        self.op = op   

    @staticmethod
    def lcu_reduction(coeff, op, N): # extract N top terms
        """Simplify the Pauli decomposition"""
        coeff = coeff[1:] # exclude the first terms
        op = op[1:]
        idx = np.argsort(coeff)[-N:][::-1]
        coeff_red = [coeff[i] for i in idx]
        op_red = [op[i] for i in idx]
        return coeff_red, op_red

    def init_config(self):
        """Quantum circuit configuration"""
        self.n_control = int(np.floor(math.log2(len(self.coeff))))
        self.n_target = int(math.log2(self.def_mat.shape[0]))

    def init_state(self, iter_num, input_coeff):
        """Prepare a quantum state on target wires"""
        if iter_num==0:
            for i in self.reg['target']:
                qml.Hadamard(i)
        else:
            base = self.get_basis_state(self.n_target)
            qml.Superposition(input_coeff, base, self.reg['target'], work_wire=self.reg['super_work'])
            
    @staticmethod
    def get_basis_state(n_qubits):
        return [list(state) for state in product([0, 1], repeat=n_qubits)]
        
    def block_encoding(self):
        """Block encoding of a given matrix"""
        qml.StatePrep(self.coeff, wires=self.reg['control'])
        qml.Select(self.op, control=self.reg['control'])
        qml.adjoint(qml.StatePrep(self.coeff, wires=self.reg['control']))

    def power_method(self):
        """Power method to find the dominant eigenpair"""
        num_iter=100
        tol=1e-4
        iter_num = 0
        input_coeff = 0
        proj_norm_coeffs = []
        vals = []
        vecs = []
    
        while iter_num < num_iter:
            # vec = self.power_circuit(iter_num, input_coeff)
            vec = self.run_power_circuit(iter_num, input_coeff)
            proj_norm_coeff, proj, proj_norm = self.get_next_vec(vec)
    
            if iter_num != 1:
                self.proj_norm_coeffs.append(proj_norm_coeff)
    
            vec = np.prod(proj_norm_coeffs) * proj
            vecs.append(vec)
            val = (vec.T @ (self.def_mat @ vec)) / (vec.T @ vec)
            vals.append(val)
    
            # stopping criteria
            if iter_num > 0 and abs(vals[-1] - vals[-2]) < tol:
                print(f"program stops at {iter_num}th iteration")
                break
     
            # updates for the next iteration
            iter_num += 1
            input_coeff = proj_norm
            
        return vals[-1], vecs[-1]
        
    @staticmethod
    def deflation(A, eig_val, eig_vec):
        """Deflate a given matrix w.r.t an eigenpair found"""
        return A - eig_val * np.outer(eig_vec, eig_vec)

    @staticmethod
    def zero_projector(n_qubits, measured_wires, outcomes):
        """Project a resultant quantum state on |0> in the first qubit"""
        return [
            i for i in range(2 ** n_qubits)
            if all(format(i, f'0{n_qubits}b')[wire] == str(outcome) 
                   for wire, outcome in zip(measured_wires, outcomes))]
    
    def get_next_vec(self, vec):
        """Get the next dominant, approximate eigenvector"""
        proj_idx = self.zero_projector(self.n_tot, measured_wires=self.reg['control'].tolist(),
                                  outcomes=np.zeros(len(self.reg['control']), dtype=int).tolist())
        proj = np.array([vec[i] for i in proj_idx]) # project onto |0> in the control qubit
        proj = proj[0::2] # effectively remove the auxiliary qubit
        proj_norm_coeff = np.linalg.norm(proj)
        proj_norm = proj / proj_norm_coeff
        return proj_norm_coeff, np.real(proj), np.real(proj_norm)

    def core_routine(self):
        """Add vectors to def_vec"""
        for i in range(self.n_centers):  
            eig_val, eig_vec = self.power_method()
            # eig_val, eig_vec = self.qnode(iter_num, input_coeff)
            self.def_vec.append(eig_vec)
            self.def_mat = self.deflation(self.def_mat, eig_val, eig_vec)
            self.update_circuit_config()

    def get_score(self, x):
        """Calculate the Silhouette_score w.r.t data embedded on LE"""
        km = KMeans(n_clusters=self.n_centers, random_state=42).fit(x)
        label_predicted = km.fit_predict(x)
        return ss(x, label_predicted), label_predicted
    
    def get_best_le(self):
        """Find the best combination of eigenvectors of L inverse"""
        elem = range(len(self.def_vec))
        all_cb = list(cb(elem,2))
        score_list = []
        label_list = []
        for pair in all_cb:
            temp_data = np.stack((self.def_vec[pair[0]], self.def_vec[pair[1]]), axis=1)
            temp_score, temp_label = self.get_score(temp_data)
            score_list.append(temp_score)
            label_list.append(temp_label)
        best_idx = np.argmax(score_list)
        self.pair = all_cb[best_idx]
        self.score = np.round(score_list[best_idx],3)
        self.Y = label_list[best_idx]
   
    def plot_embedded_X(self):
        """Plot data embedded on LE without labels"""
        i = self.pair[0]
        j = self.pair[1]
        plt.scatter(self.def_vec[i], self.def_vec[j], c=self.Y, s=30, alpha=0.8) # w/ labels colored
        plt.title(f'Data on LE with score {self.score}')
        plt.show()
        
    def embed_data(self):
        """Execute major routines"""
        self.core_routine() # def_vec, def_val
        self.get_best_le() # pair, score, Y
        self.plot_embedded_X()
