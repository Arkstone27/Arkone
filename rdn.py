from numba.experimental import jitclass
from numba import njit, prange, types

import numpy as np

type_fn_activation = types.FunctionType(types.float32(types.float32))

spec = [
    ("repartition", types.uint8[:]),
    ("nb_inputs", types.uint8),
    ("nb_outputs", types.uint8),

    ("nb_neurones", types.uint8),
    ("nb_hidden", types.uint8),

    ("fn_activation", type_fn_activation)
]

@jitclass(spec)
class RdN(object):
    def __init__(self, repartition, fn_activation):
        self.repartition = repartition
        self.nb_inputs = repartition[0]
        self.nb_outputs = repartition[1]

        self.nb_neurones = repartition[2] # Nb de neurones sur une couche cachée
        self.nb_hidden = repartition[3] # Nb de couches cachées

        self.fn_activation = fn_activation

@njit
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    repartition = np.array([3,2,3,5], dtype=np.uint8)
    rdn = RdN(repartition, sigmoide)