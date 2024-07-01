from MPSEmulator import MPSBuilder
import numpy as np

dim = 2
qubits_num = 3

state = np.array([0, 1, 0, 0, 0, 0, 0, 1])

m = MPSBuilder(qubits_num, dim)
new_state = m.state_to_vector('1/sqrt(3) * (|001> + |010> + |100>)')
m.create_right_MPS(new_state)
print(m.get_MPS())
print(m.check_MPS(new_state))
