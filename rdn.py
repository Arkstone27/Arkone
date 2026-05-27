from numba.experimental import jitclass
from numba import njit, types

import numpy as np

type_fn_activation = types.FunctionType(types.float32(types.float32))
type_fn_output = types.FunctionType(types.float32[:](types.float32[:]))

spec = [
    ("poids", types.float32[:, :, :]),
    ("biais", types.float32[:, :]),

    ("repartition", types.uint8[:]),
    ("nb_inputs", types.uint8),
    ("nb_outputs", types.uint8),

    ("nb_neurones", types.uint8),
    ("nb_hidden", types.uint8),

    ("fn_activation", type_fn_activation),
    ("fn_output", type_fn_output),
    ("derniere_action", types.bool),

    ("outputs", types.float32[:])
]
@njit(types.float32(types.float32))
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

@njit(types.float32[:](types.float32[:]))
def sigmoide_output(output):
    return 1 / (1 + np.exp(-output))

@jitclass(spec)
class RdN(object):
    def __init__(self, data_poids, data_biais, repartition, fn_activation, fn_output, derniere_action):
        self.poids = data_poids
        self.biais = data_biais

        self.repartition = repartition
        self.nb_inputs = repartition[0]
        self.nb_outputs = repartition[1]

        self.nb_neurones = repartition[2] # Nb de neurones sur une couche cachée
        self.nb_hidden = repartition[3] # Nb de couches cachées

        self.fn_activation = fn_activation
        self.fn_output = fn_output
        self.derniere_action = derniere_action

        self.outputs = np.empty(self.nb_outputs, dtype=np.float32)

    def calcul_output(self, valeurs_inputs):
        self.outputs = output(self, valeurs_inputs)

@njit
def rdn_aleatoire(repartition, fn_activation, fn_output, derniere_action):
    nb_couches = int(repartition[3]) + 1
    nb_neurones = int(repartition[2])

    data_poids = np.random.rand(nb_couches, nb_neurones, nb_neurones).astype(np.float32) * 2 - 1
    data_biais = np.random.rand(nb_couches, nb_neurones).astype(np.float32) * 2 - 1

    return RdN(data_poids, data_biais, np.array(repartition, dtype=np.uint8), fn_activation, fn_output, derniere_action)

@njit
def output(rdn, valeurs_inputs):
    return np.empty(rdn.nb_outputs, dtype=np.float32)


if __name__ == '__main__':
    repartition = (3,2,3,5)
    rdn = rdn_aleatoire(repartition, sigmoide, sigmoide_output, False)

    rdn.calcul_output(np.array([1,2,3], dtype=np.float32))
