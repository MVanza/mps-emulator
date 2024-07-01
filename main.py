from typing import List
import numpy as np
from numpy.linalg import svd
import re

class MPSBuilder():
    def __init__(self, qubits_num, dim):
        self.qubits_num = qubits_num
        self.dim = dim
        self.mps_matrices = []

    def _bin_to_nparray(self, bin):
        state = np.zeros((dim**qubits_num))
        ind = 0
        for e, n in enumerate(bin.group(1)):
            ind += self.dim**(self.qubits_num-1-e) * int(n)
        state[ind] = 1
        return f'np.array({state.tolist()})'
    
    @staticmethod
    def _elem_func_to_np(elem_func):
        return f"np.{elem_func.group(0)}"    

    def state_to_vector(self, dirac_notation):
        temp_state = re.sub(r'[a-zA-Z]+', self._elem_func_to_np, dirac_notation)
        result_state = re.sub(r'\|(\d+)>', self._bin_to_nparray, temp_state)
        print(result_state)
        return eval(result_state)
        
        
    
    def createMPS(self, state_vector: np.ndarray):
        new_state = state_vector
        alpha = 1
        for i in range(1, self.qubits_num):
            # print(f"new_state before reshape {new_state}")
            new_state = new_state.reshape((self.dim * alpha, (self.dim**(self.qubits_num-i))), order="F")
            # print(f"new_state after reshape {new_state}")
            U, S, V = svd(new_state, full_matrices=False)
            # print(f"U is {U}, S is {S}, V is {V}")
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
    
    def check_MPS(self, state_vector):
        mps_state_vector = self.mps_matrices[0]
        for m in self.mps_matrices[1:]:
            mps_state_vector = mps_state_vector @ m
        mps_state_vector = np.where(mps_state_vector < 1e-15, 0, mps_state_vector)
        return (state_vector == mps_state_vector.reshape((self.dim**(self.qubits_num)))).all()

dim = 2
qubits_num = 3

mps_matrices = []

state = np.array([0, 1, 1, 0, 1, 0, 0, 0]) * 1/np.sqrt(3)

m = MPSBuilder(3, 2)
new_state = m.state_to_vector('1/sqrt(2) * (|000> + |111>)')
m.createMPS(new_state)
print(m.get_MPS())
print(m.check_MPS(new_state))

        
