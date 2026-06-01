from numba import njit, prange
from numba.typed import List

import numpy as np

from rdn import rdn_aleatoire, sigmoide, RdN
from exemple_evaluer import evaluer_snake

# Initialisation
@njit
def creation_population(nb_agents, modele_rdn):
    rdns = List()

    for _ in range(nb_agents):
        rdns.append(rdn_aleatoire(*modele_rdn))

    return rdns

# Etape 2 - Perturbations
@njit
def mutations(rdns, nb_survivants, classement,sigma_poids, sigma_biais):
    shape_poids, shape_biais = rdns[0].poids.shape, rdns[0].biais.shape
    choix_parents = np.random.randint(0, nb_survivants, nb_survivants)

    indices_meilleurs = classement[:nb_survivants]

    for idx in range(nb_survivants, len(rdns)):
        rdn_enfant = rdns[classement[idx]]
        rdn_parent = rdns[indices_meilleurs[np.random.randint(0, nb_survivants)]]

        perturbation_poids = np.random.randn(*shape_poids).astype(np.float32) * sigma_poids
        perturbation_biais = np.random.randn(*shape_biais).astype(np.float32) * sigma_biais

        rdn_enfant.update(rdn_parent.poids + perturbation_poids, rdn_parent.biais + perturbation_biais)

# Etape 3 - Evaluation
@njit(parallel=True)
def evaluation(rdns, fn_evaluer, indice_debut, classement, notes):
    nb_rdns = len(rdns)

    for idx in prange(indice_debut, nb_rdns):
        idx_rdn = classement[idx]
        notes[idx] = fn_evaluer(rdns[idx_rdn])

def entrainement_NES(modele_rdn: tuple, fn_evaluer, note_objectif: float):
    """
    modele_rdn: [repartition, fn_activation, fn_output, derniere_action]
    fn_evaluer: Une fonction njit qui prend un rdn de type modele_rdn
    et qui evaluer le rdn (meilleure note = 0)
    """

    # Initialisation
    sigma_poids, sigma_biais = np.float32(0.05), np.float32(0.05)
    rdns, nb_survivants = creation_population(200, modele_rdn), 30

    notes, classement = np.empty(len(rdns), dtype=np.float32), np.arange(len(rdns))
    evaluation(rdns, fn_evaluer, 0, classement,notes)
    note_min = notes[classement[0]]

    # Entrainement
    n = 0
    while note_min >= note_objectif and n < 1000:
        n += 1

        classement = np.argsort(notes)
        mutations(rdns, nb_survivants, classement, sigma_poids, sigma_biais)
        evaluation(rdns, fn_evaluer, nb_survivants, classement,notes)

        note_min = notes[classement[0]]
        print(note_min)

    return rdns[classement[0]]

if __name__ == "__main__":
    modele_rdn = ((256,4,16,1), sigmoide, False)
    entrainement_NES(modele_rdn, evaluer_snake, 0.0)
