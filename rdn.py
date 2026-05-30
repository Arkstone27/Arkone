from numba.experimental import jitclass
from numba import njit, types

import numpy as np

type_fn_activation = types.FunctionType(types.float32(types.float32))

spec = [
    ("poids", types.float32[:, :, :]),
    ("biais", types.float32[:, :]),

    ("repartition", types.UniTuple(types.uint8, 4)),
    ("nb_inputs", types.uint8),
    ("nb_outputs", types.uint8),

    ("nb_neurones", types.uint8),
    ("nb_hidden", types.uint8),

    ("fn_activation", type_fn_activation),
    ("derniere_action", types.boolean),

    ("outputs", types.float32[:])
]
@njit(types.float32(types.float32))
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

@jitclass(spec)
class RdN(object):
    def __init__(self, data_poids, data_biais, repartition, fn_activation, derniere_action):
        self.poids = data_poids
        self.biais = data_biais

        self.repartition = repartition
        self.nb_inputs = repartition[0]
        self.nb_outputs = repartition[1]

        self.nb_neurones = repartition[2] # Nb de neurones sur une couche cachée
        self.nb_hidden = repartition[3] # Nb de couches cachées

        self.fn_activation = fn_activation
        self.derniere_action = derniere_action

        self.outputs = np.empty(self.nb_outputs, dtype=np.float32)

    def calcul_output(self, valeurs_inputs):
        self.outputs = calcul_output(self, valeurs_inputs)

        return self.outputs

    def update(self, new_poids, new_biais):
        self.poids = new_poids
        self.biais = new_biais

    def copy(self):
        return RdN(self.poids.copy(), self.biais.copy(), self.repartition, self.fn_activation, self.derniere_action)

@njit
def rdn_aleatoire(repartition, fn_activation, derniere_action):
    nb_inputs, nb_outputs = int(repartition[0]), int(repartition[1])
    nb_neurones, nb_couches = int(repartition[2]), int(repartition[3]) + 1

    max_neurones = max(nb_inputs, nb_outputs, nb_couches)

    data_poids = np.random.randn(nb_couches, max_neurones, max_neurones).astype(np.float32) * np.float32(0.1)
    data_biais = np.random.randn(nb_couches, max_neurones).astype(np.float32) * np.float32(0.1)

    return RdN(data_poids, data_biais, repartition, fn_activation, derniere_action)

@njit
def calcul_output(rdn, valeurs_inputs):
    output = np.zeros(rdn.nb_neurones, dtype=np.float32)
    poids, biais = rdn.poids, rdn.biais

    nb_neurones, fn_activation = rdn.nb_neurones, rdn.fn_activation

    for indice_suivant in range(nb_neurones):
        for indice_neurone in range(rdn.nb_inputs):
            output[indice_suivant] += poids[0, indice_neurone, indice_suivant] * valeurs_inputs[indice_neurone]
        output[indice_suivant] += biais[0, indice_suivant]
        output[indice_suivant] = fn_activation(output[indice_suivant])

    for indice_couche in range(1, rdn.nb_hidden):
        inputs, output = output, np.zeros(rdn.nb_neurones, dtype=np.float32)

        for indice_suivant in range(nb_neurones):
            for indice_neurone in range(nb_neurones):
                output[indice_suivant] += poids[indice_couche, indice_neurone, indice_suivant] * inputs[indice_neurone]
            output[indice_suivant] += biais[indice_couche, indice_suivant]
            output[indice_suivant] = fn_activation(output[indice_suivant])

    indice_output = rdn.nb_hidden
    inputs, output = output, np.zeros(rdn.nb_outputs, dtype=np.float32)

    for indice_suivant in range(rdn.nb_outputs):
        for indice_neurone in range(nb_neurones):
            output[indice_suivant] += poids[indice_output, indice_neurone, indice_suivant] * inputs[indice_neurone]
        output[indice_suivant] += biais[indice_output, indice_suivant]

    return output

if __name__ == '__main__':
    repartition = (3,3,1,5)
    rdn = rdn_aleatoire(repartition, sigmoide, False)

    print(rdn.calcul_output(np.array([1,2,3], dtype=np.float32)))
    print(rdn.calcul_output(np.array([3,2,3], dtype=np.float32)))
