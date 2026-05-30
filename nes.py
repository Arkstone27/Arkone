from numba import njit, prange
from numba.typed import List

import numpy as np

from rdn import rdn_aleatoire, sigmoide, RdN
from exemple_evaluer import evaluer_snake

# Initialisation
@njit
def creation_enfants(rdn_solution, nb_enfants=20):
    rdns_enfant = List()

    for _ in range(nb_enfants):
        rdns_enfant.append((rdn_solution.copy(), rdn_solution.copy()))
    return rdns_enfant

@njit
def creation_perturbations(rdn_solution: RdN, nb_enfants=20):
    poids_solution, biais_solution = rdn_solution.poids, rdn_solution.biais
    shape_poids, shape_biais = poids_solution.shape, biais_solution.shape

    return np.empty((nb_enfants, *shape_poids), dtype=np.float32), np.empty((nb_enfants, *shape_biais), dtype=np.float32)

# Etape 1 - Perturbations
@njit
def perturbations(rdns_enfant, rdn_solution: RdN, perturbations_poids, perturbations_biais, sigma_poids, sigma_biais):
    poids_solution, biais_solution = rdn_solution.poids, rdn_solution.biais
    shape_poids, shape_biais = poids_solution.shape, biais_solution.shape

    idx = 0
    for rdn_enfant, rdn_enfant_miroir in rdns_enfant:
        perturbation_poids = np.random.randn(*shape_poids).astype(np.float32)
        perturbation_biais = np.random.randn(*shape_biais).astype(np.float32)

        rdn_enfant.update(poids_solution + perturbation_poids * sigma_poids, biais_solution + perturbation_biais * sigma_biais)
        rdn_enfant_miroir.update(poids_solution - perturbation_poids * sigma_poids, biais_solution - perturbation_biais * sigma_biais)

        perturbations_poids[idx] = perturbation_poids
        perturbations_biais[idx] = perturbation_biais
        idx += 1

# Etape 2 - Evaluation
@njit(parallel=True)
def evaluation(rdns_enfant, fn_evaluer):
    nb_enfants = len(rdns_enfant)
    notes = np.empty((nb_enfants,2), dtype=np.float32)

    for idx in prange(nb_enfants):
        notes[idx, 0] = 1 - fn_evaluer(rdns_enfant[idx][0])
        notes[idx, 1] = 1 - fn_evaluer(rdns_enfant[idx][1])

    return notes

@njit
def standardisation(notes):
    moyenne, ecart_type = notes.mean(), notes.std()

    if ecart_type < 1e-5:
        return None

    for idx_note in range(notes.shape[0]):
        notes[idx_note, 0] = (notes[idx_note, 0] - moyenne) / ecart_type
        notes[idx_note, 1] = (notes[idx_note, 1] - moyenne) / ecart_type
    return None
# Etape 3
@njit
def approximation_gradient(perturbations_poids, perturbations_biais, notes, sigma_poids, sigma_biais):
    gradient_poids = np.zeros(perturbations_poids.shape[1:], dtype=np.float32)
    gradient_biais = np.zeros(perturbations_biais.shape[1:], dtype=np.float32)

    nb_enfants, n = notes.shape
    for idx in range(nb_enfants):
        difference = notes[idx, 0] - notes[idx, 1]

        gradient_poids += perturbations_poids[idx] * difference
        gradient_biais += perturbations_biais[idx] * difference

    gradient_poids /= (nb_enfants * 2.0 * sigma_poids)
    gradient_biais /= (nb_enfants * 2.0 * sigma_biais)
    return gradient_poids, gradient_biais

@njit
def application_gradient(rdn_solution: RdN, gradient_poids, gradient_biais, taux_entrainement):
    poids_solution, biais_solution = rdn_solution.poids, rdn_solution.biais

    poids_solution += taux_entrainement * gradient_poids
    biais_solution += taux_entrainement * gradient_biais

def entrainement_NES(modele_rdn: tuple, fn_evaluer, note_objectif: float):
    """
    modele_rdn: [repartition, fn_activation, fn_output, derniere_action]
    fn_evaluer: Une fonction njit qui prend un rdn de type modele_rdn
    et qui evaluer le rdn (meilleure note = 0)
    """

    # Initialisation
    rdn_solution = rdn_aleatoire(*modele_rdn)

    taux_entrainement, note_rdn = 0.01, fn_evaluer(rdn_solution)
    sigma_poids, sigma_biais = np.float32(0.02), np.float32(0.02)

    rdns_enfant = creation_enfants(rdn_solution, 200)
    perturbations_poids, perturbations_biais = creation_perturbations(rdn_solution, len(rdns_enfant))

    # Entrainement
    n = 0
    while note_rdn >= note_objectif and n < 10000:
        n += 1
        # 1 - Mise a jour des perturbations
        perturbations(rdns_enfant, rdn_solution, perturbations_poids, perturbations_biais, sigma_poids, sigma_biais)

        # 2 - Evaluation
        notes = evaluation(rdns_enfant, fn_evaluer)
        standardisation(notes)

        # 3 - Nouvelle solution
        gradient_poids, gradient_biais = approximation_gradient(perturbations_poids, perturbations_biais, notes, sigma_poids, sigma_biais)
        application_gradient(rdn_solution, gradient_poids, gradient_biais, taux_entrainement)

        note_rdn = fn_evaluer(rdn_solution)
        print(note_rdn)

if __name__ == "__main__":
    modele_rdn = ((256,4,16,1), sigmoide, False)
    entrainement_NES(modele_rdn, evaluer_snake, 0.0)
