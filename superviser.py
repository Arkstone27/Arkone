from numba import njit, prange

import numpy as np

from exemple_evaluer import creer_data_set
from rdn import rdn_aleatoire, softmax, RdN

@njit(parallel=True)
def evaluer(rdn, fonction_fin, data_set):
    inputs, sorties_attendues = data_set
    nb_lignes, taille_sortie = inputs.shape[0], sorties_attendues[0].shape[0]

    somme_erreurs = np.empty(nb_lignes, dtype=np.float32)
    for idx in prange(nb_lignes):
        sortie = fonction_fin(rdn.calcul_output(inputs[idx]))
        sortie_attendu = sorties_attendues[idx]

        somme_erreur = np.float32(0)
        for idx_sortie in range(taille_sortie):
            somme_erreur += (sortie[idx_sortie] - sortie_attendu[idx_sortie])**2
        somme_erreurs[idx] = somme_erreur / taille_sortie

    erreur_totale = np.float32(0)
    for idx in range(nb_lignes):
        erreur_totale += somme_erreurs[idx]
    return erreur_totale / nb_lignes

@njit
def entrainement_data(data_set, modele_rdn, fonction_fin, note_objectif):
    taux_entrainement = np.float32(0.1)
    rdn = rdn_aleatoire(*modele_rdn)

    print(evaluer(rdn, fonction_fin, data_set))
    n = 0
    while n < 100:
        n += 1


if __name__ == "__main__":
    modele_rdn = ((2,1,3,4), False)

    data_set = creer_data_set()
    entrainement_data(data_set, modele_rdn, softmax, 0.2)

