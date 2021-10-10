import matplotlib.pyplot as plt
import numpy as np

from itertools import product

import scipy.sparse as sp

import random

from math import ceil

class Wilson_Cowan_system:
    def __init__(self, N=50, A=None):
        if A is not None:
            assert A.shape[0] == A.shape[1]
            N = A.shape[0]
            self.A = A
        self.N = N
        self.E0 = np.zeros(N)
        self.I0 = np.zeros(N)
        self.params = {
            'τE': 1/8,
            'τI': 1/4,
            'aE': .8,
            'aI': .8,
            'θE': 2,
            'θI': 5,
            'cEE': 8,
            'cEI': 16,
            'cIE': 8,
            'cII': 4,
            'ν': .5
        }
        self.firing_rate = lambda x: 1/(1+np.exp(-x))
        self.external_stimulus = lambda t: np.random.random(self.N)
        
    def set_initial_excitatory(self, E):
        assert E.shape == self.E0.shape
        self.E0 = E.copy()
        
    def set_initial_inhibitory(self, I):
        assert I.shape == self.I0.shape
        self.I0 = I.copy()
        
    def set_initial(self, E, I):
        self.set_initial_excitatory(E)
        self.set_initial_inhibitory(I)
        
    def set_adjacency_matrix(self, A):
        self.A = A.copy()
        
    def add_edges_from(self, edge_list, symmetric=True):
        for a, b in edge_list:
            self.A[b, a] = 1
            if symmetric:
                self.A[a, b] = 1
                
    def update_params(self, params):
        self.params = {**self.params, **params}
                
    def set_firing_rate(self, f):
        self.firing_rate = f
        
    def set_external_stimulus(self, p):
        self.external_stimulus = p
                
    def show_network(self):
        plt.figure(figsize=(16, 12))
        plt.subplot(1, 2, 1)
        angles = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        for edge in [ (a, b) for a, b in product(range(self.N), range(self.N)) if self.A[a, b]==1]:
            plt.plot(np.cos([angles[node] for node in edge]), np.sin([angles[node] for node in edge]), 'k-')
        plt.plot(np.cos(angles), np.sin(angles), 'bo', markersize=(10))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplot(1, 2, 2)
        plt.spy(self.A)
        plt.show()
                
    def simulate(self, t_final, t0=0, dt=1e-3):
        return Wilson_Cowan_system._simulate(t_final, t0, dt, self.N, self.E0, self.I0, 
                                             self.A,
                                             self.firing_rate,
                                             self.external_stimulus,
                                             **self.params)
            
    def _simulate(t_final, t0, Δt, N, E0, I0, A, firing_rate, external_stimulus,
                  τE, τI, aE, aI, θE, θI, cEE, cEI, cIE, cII, ν):
        steps = ceil((t_final-t0)/Δt)
        ts = np.arange(steps+1)*Δt + t0
        Es = np.zeros((steps+1, N))
        Is = np.zeros((steps+1, N))
        t = t0
        E = E0.copy()
        I = I0.copy()
        for step, t in enumerate(ts):
            Es[step], Is[step] = E, I
            E += (-Es[step] + firing_rate(aE*( cEE*Es[step] - cEI*Is[step] - θE + external_stimulus(t) + ν*A@Es[step] )))/τE * Δt
            I += (-Is[step] + firing_rate(aI*( cIE*Es[step] - cII*Is[step] - θI)))/τI * Δt
            
        return ts, Es, Is
    
def ring_edge_list(N=50, link_radius=3):
    return [(a,(a+i)%N) for a in range(N) for i in range(-link_radius, link_radius+1) if i!=0]

def randomize_edges(N, edge_list, random_edges):
    num_edges = len(edge_list)
    random.shuffle(edge_list)
    if random_edges > 0:
        edge_list = edge_list[:-random_edges]
    while len(edge_list) < num_edges:
        a, b = random.randrange(N), random.randrange(N)
        if a != b and (a, b) not in edge_list:
            edge_list.append((a,b))
    return edge_list
            
def edge_list_to_matrix(N, edge_list, matrix_type='dense'):
    if matrix_type == 'dense':
        A = np.zeros((N, N))
        for a, b in edge_list:
            A[b, a] = 1 
    elif matrix_type == 'csr':
        data = [1 for _ in edge_list]
        row_index = [b for a, b in edge_list]
        col_index = [a for a, b in edge_list]
        A = sp.csr_matrix((data, (row_index, col_index)), shape=(N,N))
        
    return A
