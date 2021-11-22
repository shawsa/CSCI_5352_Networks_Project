import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import scipy.sparse as sp
import random
import math

def randomize_edges(N, edge_list, random_edges):
    m = len(edge_list)
    num_edges = len(edge_list)
    random.shuffle(edge_list)
    if random_edges > 0:
        unallocated_weights = [w for a, b, w in edge_list[m-random_edges:]]
        edge_list = edge_list[:-random_edges]
    while len(edge_list) < num_edges:
        a, b = random.randrange(N), random.randrange(N)
        if a != b and (a, b) not in edge_list:
            edge_list.append((a,b,unallocated_weights[-1]))
            unallocated_weights = unallocated_weights[:-1]
    return edge_list

def edges_to_adjacency_matrix(N, edge_list, matrix_type='dense', symmetric=True):
    if symmetric:
        edge_list = edge_list.copy() + [(b, a, w) for a, b, w in edge_list]
    if matrix_type == 'dense':
        A = np.zeros((N, N))
        for a, b, w in edge_list:
            A[b, a] = w
    elif matrix_type == 'csr':
        data = [w for a, b, w in edge_list]
        row_index = [b for a, b, w in edge_list]
        col_index = [a for a, b, w in edge_list]
        A = sp.csr_matrix((data, (row_index, col_index)), shape=(N,N))
        
    return A

def sigmoid(x):
    return 1/(1+np.exp(x))
    
def relu(x):
    return x*np.heaviside(x, 0)

class Wilson_Cowan_Network:
    '''
    A general Wilson-Cowan network.
    '''
    def __init__(self, EE_mat, EI_mat, IE_mat, II_mat):
        NE = EE_mat.shape[0]
        NI = II_mat.shape[0]
        assert EE_mat.shape == (NE, NE)
        assert EI_mat.shape == (NE, NI)
        assert IE_mat.shape == (NI, NE)
        assert II_mat.shape == (NI, NI)
        self.NE = NE
        self.NI = NI
        self.EE_mat = EE_mat.copy()
        self.EI_mat = EI_mat.copy()
        self.IE_mat = IE_mat.copy()
        self.II_mat = II_mat.copy()
        self.excitatory_firing_rate = lambda x: x*np.heaviside(x,0)
        self.inhibitory_firing_rate = lambda x: x*np.heaviside(x,0)
        self.excitatory_stimulus = lambda t: 0
        self.inhibitory_stimulus = lambda t: 0
        self.τE = 1/8
        self.τI = 1/4
        self.variance = 0.1
        self.E0 = np.zeros(NE)
        self.I0 = np.zeros(NI)
        
    def simulate(self, t_final=10, t0=0, Δt=1e-2):
        steps = math.ceil((t_final-t0)/Δt)
        ts = np.arange(steps+1)*Δt + t0
        Es = np.zeros((steps+1, self.NE))
        Is = np.zeros((steps+1, self.NI))
        Es[0] = self.E0.copy()
        Is[0] = self.I0.copy()
        for index, t in enumerate(ts[:-1]):
            Es[index+1] = Es[index] + (-Es[index] + self.excitatory_firing_rate(self.EE_mat@Es[index] 
                                                  - self.EI_mat@Is[index]))*Δt/self.τE + np.sqrt(2*Δt)*self.variance*np.random.randn(self.NE)
            Is[index+1] = Is[index] + (-Is[index] + self.inhibitory_firing_rate(self.EI_mat@Es[index] 
                                                  - self.II_mat@Is[index]))*Δt/self.τI
        return ts, Es, Is
    
def nazemi_jamali_network(N, coupling_weight=.8, neighbor_radius=3, random_edges = 0):
    '''
    Networks from Nazemi and Jamali (2019) - Frontiers in Computational Neuroscience.
    '''
    EE_weight = 8
    II_weight = 4
    EI_weight = 16
    IE_weight = 8

    # standard Wilson-Cowan connections
    EE_edges = [(a, a, EE_weight) for a in range(N)]
    II_edges = [(a, a, II_weight) for a in range(N)]
    EI_edges = [(a, a, EI_weight) for a in range(N)]
    IE_edges = [(a, a, IE_weight) for a in range(N)]

    # ring connections
    coupling_edges = [(a, (a+i)%N, coupling_weight) for a in range(N) for i in range(1, neighbor_radius+1)]

    coupling_edges = randomize_edges(N, coupling_edges, random_edges)

    EE_edges = EE_edges + coupling_edges
    
    matrix_type = 'dense'
    if N > 100:
        matrix_type = 'csr'

    EE_mat = edges_to_adjacency_matrix(N, EE_edges, symmetric=True, matrix_type=matrix_type)
    II_mat = edges_to_adjacency_matrix(N, II_edges, symmetric=True, matrix_type=matrix_type)
    EI_mat = edges_to_adjacency_matrix(N, EI_edges, symmetric=True, matrix_type=matrix_type)
    IE_mat = edges_to_adjacency_matrix(N, IE_edges, symmetric=True, matrix_type=matrix_type)

    wcn = Wilson_Cowan_Network(EE_mat, EI_mat, IE_mat, II_mat)

    aE = 0.8
    θE = 2
    wcn.excitatory_firing_rate = lambda x: sigmoid(aE*(x - θE))
    aI = 0.8
    θI = 8
    wcn.inhibitory_firing_rate = lambda x: sigmoid(aI*(x - θI))

    wcn.E0 = np.random.random(N)
    wcn.I0 = np.random.random(N)
    
    return wcn
