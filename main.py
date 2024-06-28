from typing import List
import numpy as np
from numpy.linalg import svd

class MPSBuilder():
    def __init__(self, qubits_num, dim):
        self.qubits_num = qubits_num
        self.dim = dim
        self.mps_matrices = []

    def state_to_vector(self, dirac_notation):
        pass
    
    def createMPS(self, state_vector: np.ndarray):
        new_state = state_vector
        alpha = 1
        for i in range(1, self.qubits_num):
            new_state = new_state.reshape((self.dim * alpha, (self.dim**(self.qubits_num-i))), order="F")
            U, S, V = svd(new_state, full_matrices=False)
            alpha = U.shape[1]
            if i == 1:
                self.mps_matrices.append(U)
            else:
                self.mps_matrices.append(U.reshape((U.shape[0]-self.dim, self.dim, U.shape[1]), order="F"))
                if i == self.qubits_num - 1:
                    self.mps_matrices.append(np.diag(S) @ V)
            new_state = np.diag(S) @ V

    def get_MPS(self) -> List:
        return self.mps_matrices

dim = 2
qubits_num = 3

mps_matrices = []

state = np.array([1, 0, 0, 0, 0, 0, 0, 1]) * 1/np.sqrt(2)



    
m = MPSBuilder(3, 2)
m.createMPS(state)
print(m.get_MPS())
        
