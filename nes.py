from numba import njit, prange, types
from numba.typed import List, Dict

import numpy as np

from rdn import rdn_aleatoire
from exemple_evaluer import evaluer_morpion, partie_morpion, coup_rdn, coup_utilisateur


# Initialisation
@njit
def creation_population(nb_agents, modele_rdn):
    rdns = List()

    for _ in range(nb_agents):
        rdns.append(rdn_aleatoire(*modele_rdn))

    return rdns

# Etape 2 - Perturbations
@njit
def croisement(rdn_enfant, rdn_1, rdn_2):
    masque_poids = np.random.randint(0, 2, size=rdn_1.poids.shape).astype(np.bool_)
    masque_biais = np.random.randint(0, 2, size=rdn_1.biais.shape).astype(np.bool_)

    new_poids = np.where(masque_poids, rdn_1.poids, rdn_2.poids)
    new_biais = np.where(masque_biais, rdn_1.biais, rdn_2.biais)

    rdn_enfant.update(new_poids, new_biais)

@njit(parallel=True)
def croisements(rdns, nb_survivants, classement):
    indices_meilleurs = classement[:nb_survivants]
    for idx in prange(nb_survivants, len(rdns)):
        rdn_enfant = rdns[classement[idx]]

        idx_parent_1 = np.random.randint(0, nb_survivants)
        rdn_1 = rdns[indices_meilleurs[idx_parent_1]]

        idx_parent_2 = np.random.randint(0, nb_survivants)
        rdn_2 = rdns[indices_meilleurs[idx_parent_2]]

        croisement(rdn_enfant, rdn_1, rdn_2)

@njit
def mutations(rdns, nb_survivants, classement, sigma_poids, sigma_biais):
    shape_poids, shape_biais = rdns[0].poids.shape, rdns[0].biais.shape

    for idx in range(nb_survivants, len(rdns)):
        rdn_enfant = rdns[classement[idx]]

        perturbation_poids = np.random.randn(*shape_poids).astype(np.float32) * sigma_poids
        perturbation_biais = np.random.randn(*shape_biais).astype(np.float32) * sigma_biais

        rdn_enfant.update(rdn_enfant.poids + perturbation_poids, rdn_enfant.biais + perturbation_biais)

# Etape 3 - Evaluation
@njit(parallel=True)
def evaluation(rdns, fn_evaluer, indice_debut, classement, notes):
    nb_rdns = len(rdns)

    for idx in prange(indice_debut, nb_rdns):
        idx_rdn = classement[idx]
        notes[idx_rdn] = fn_evaluer(rdns[idx_rdn])

@njit
def entrainement_NES(modele_rdn: tuple, fn_evaluer, note_objectif: float):
    # Initialisation
    sigma_poids, sigma_biais = np.float32(0.05), np.float32(0.005)
    rdns, nb_survivants = creation_population(100, modele_rdn), 30

    notes, classement = np.empty(len(rdns), dtype=np.float32), np.arange(len(rdns))
    evaluation(rdns, fn_evaluer, 0, classement,notes)
    note_min, note_avant = notes[classement[0]], notes[classement[0]]

    # Entrainement
    n = 0
    while note_min >= note_objectif and n < 1_000_000:
        n += 1

        classement = np.argsort(notes)
        note_min = notes[classement[0]]
        if note_avant > note_min:
            print(n, note_min)
        note_avant = note_min

        croisements(rdns, nb_survivants, classement)
        mutations(rdns, nb_survivants, classement, sigma_poids, sigma_biais)
        evaluation(rdns, fn_evaluer, nb_survivants, classement, notes)

    return rdns[classement[0]]

if __name__ == "__main__":
    modele_rdn = ((10,9,25,6), False)
    rdn = entrainement_NES(modele_rdn, evaluer_morpion, 0.15)

    type_grille = types.Tuple([types.int8] * 10)
    type_coup = types.Tuple((types.int8, types.int8))

    print(partie_morpion(coup_rdn, coup_utilisateur, rdn, Dict.empty(key_type=type_grille, value_type=type_coup)))
