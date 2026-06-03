from numba import njit, prange
from numba.typed import List

import numpy as np

from rdn import rdn_aleatoire, sigmoide, RdN

@njit
def evaluer_simple(rdn):
    resultat_1 = rdn.calcul_output(np.array([0.2,0.3]))
    resultat_1 = np.abs(sigmoide(resultat_1[0]) - 0.432)

    resultat_2 = rdn.calcul_output(np.array([-0.1,0.7]))
    resultat_2 = np.abs(sigmoide(resultat_2[0]) - 0.67)

    return np.float32((resultat_1 + resultat_2) / 2)

def creer_data_set():
    inputs_liste = [[0.2,0.3], [-0.1, 0.7]]
    sorties_attendues = [[0.432], [0.67]]

    inputs = np.empty((2,2), dtype=np.float32)
    for i in range(len(inputs_liste)):
        inputs[i] = np.array(inputs_liste[i])

    sorties = np.empty((2,1), dtype=np.float32)
    for i in range(len(sorties_attendues)):
        sorties[i] = np.array(sorties_attendues[i])

    return inputs, sorties
if __name__ == '__main__':
    print(creer_data_set())
    repartition = (2,1,4,3)

    rdn = rdn_aleatoire(repartition, False)
    note = evaluer_simple(rdn)
    print(note)
