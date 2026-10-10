from numba import types, objmode, njit
from numba.typed import List, Dict

import numpy as np

from rdn import rdn_aleatoire, sigmoide,softmax, choix_max

import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning
warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

type_grille = types.Tuple([types.int8] * 10)
type_coup = types.Tuple((types.int8, types.int8))

@njit
def evaluer_simple(rdn):
    resultat_1 = rdn.calcul_output(np.array([0.2,0.3]))
    resultat_1 = np.abs(sigmoide(resultat_1[0]) - 0.432)

    resultat_2 = rdn.calcul_output(np.array([-0.1,0.7]))
    resultat_2 = np.abs(sigmoide(resultat_2[0]) - 0.67)

    return np.float32((resultat_1 + resultat_2) / 2)

@njit
def evaluer_morpion(rdn):

    cache = Dict.empty(key_type=type_grille, value_type=type_coup)
    nb_parties, note = 100, 0
    for _ in range(nb_parties):
        if partie_morpion(coup_rdn, coup_aleatoire, rdn, cache) == 0:
            note += 1

        # if partie_morpion(coup_aleatoire, coup_aleatoire, rdn) == 1:
        #     note += 1

    return note / nb_parties

@njit
def coup_rdn(grille, joueur, rdn, cache):
    grille_plat = grille.ravel()
    tuple_inputs = (joueur, grille_plat[0], grille_plat[1], grille_plat[2],
                    grille_plat[3], grille_plat[4], grille_plat[5],
                    grille_plat[6], grille_plat[7], grille_plat[8])
    if tuple_inputs in cache:
        return cache[tuple_inputs]

    inputs = np.empty(10, dtype=np.float32)
    inputs[0], inputs[1:] = joueur, grille_plat

    output = softmax(rdn.calcul_output(inputs))
    for i in range(9):
        if grille_plat[i] != -1:
            output[i] = 0

    indice_case = choix_max(output)
    x = indice_case // 3
    y = indice_case % 3

    cache[tuple_inputs] = (x,y)
    return x, y

@njit
def coup_aleatoire(grille, joueur, rdn, cache):
    libres_x = np.empty(9, dtype=np.int8)
    libres_y = np.empty(9, dtype=np.int8)
    compteur = 0

    for x in range(3):
        for y in range(3):
            if grille[x, y] == -1:
                libres_x[compteur], libres_y[compteur] = x, y
                compteur += 1
    choix_case = np.random.randint(0, compteur)
    return libres_x[choix_case], libres_y[choix_case]

@njit
def coup_utilisateur(grille, joueur, rdn, cache):
    grille_plat = grille.ravel()

    with objmode(indice_case=types.int8):

        print(grille)
        indice_case = int(input("Quel case ?"))
        while grille_plat[indice_case] != -1:
            indice_case = int(input("Invalide !"))

    cases = ((0,0),(0,1),(0,2), (1,0),(1,1),(1,2), (2,0),(2,1),(2,2))
    return cases[indice_case]

@njit
def grille_gagnante(grille, joueur):
    # Colonnes et lignes
    for i in range(3):
        ligne_gagnante = True
        colonne_gagnante = True

        for j in range(3):
            if grille[i, j] != joueur: ligne_gagnante = False
            if grille[j, i] != joueur: colonne_gagnante = False

        if ligne_gagnante or colonne_gagnante:
            return True

    # Diagonales
    if grille[1,1] != joueur:
        return False

    if (grille[0,0] == joueur and grille[2,2] == joueur) or (grille[0,2] == joueur and grille[2,0] == joueur):
        return True

    return False

@njit
def partie_morpion(joueur_0, joueur_1, rdn, cache):
    grille = -np.ones((3,3), np.int8)

    nb_tours, joueur = 0, 0
    resultat_partie = -1

    while nb_tours < 9:
        if joueur == 0:
            x_coup, y_coup = joueur_0(grille, joueur, rdn, cache)
        else:
            x_coup, y_coup = joueur_1(grille, joueur, rdn, cache)

        grille[x_coup, y_coup] = joueur
        if 4 < nb_tours and grille_gagnante(grille, joueur):
            resultat_partie = joueur
            break
        joueur = 0 if joueur == 1 else 1
        nb_tours += 1

    return resultat_partie

def creer_data_set():
    inputs_liste = [[0.2,0.3], [-0.1, 0.7]]
    sorties_attendues = [[0.432], [0.67]]

    inputs = np.array(inputs_liste, dtype=np.float32)
    sorties = np.array(sorties_attendues, dtype=np.float32)

    return inputs, sorties
if __name__ == '__main__':
    repartition = (10,9,10,3)
    rdn = rdn_aleatoire(repartition, False)

    print(evaluer_morpion(rdn))
    print(partie_morpion(coup_rdn, coup_utilisateur, rdn, Dict.empty(key_type=type_grille, value_type=type_coup)))

