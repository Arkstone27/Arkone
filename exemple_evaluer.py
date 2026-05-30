from numba import njit, prange
from numba.typed import List

import numpy as np

from rdn import rdn_aleatoire, sigmoide, RdN

# Snake

@njit
def action_joueur(rdn: RdN, plateau, tete):
    directions = ((1,0),(0,1),(-1,0),(0,-1))
    plateau[tete[0], tete[1]] = - 1

    inputs = plateau.ravel().astype(np.float32)
    output = rdn.calcul_output(inputs)

    idx_max, valeur_max = 0, output[0]
    for idx in range(1,4):
        if valeur_max < output[idx]:
            idx_max, valeur_max = idx, output[idx]

    return directions[idx_max]

@njit
def update_plateau(serpent, besoin_pomme: bool, x_pomme, y_pomme):
    plateau = np.zeros((16, 16), np.int16)

    for x, y in serpent:
        plateau[x, y] = 1

    if not besoin_pomme:
        plateau[x_pomme, y_pomme] = 2
        return plateau, x_pomme, y_pomme

    x_pomme, y_pomme = np.random.randint(0,16), np.random.randint(0,16)
    while plateau[x_pomme, y_pomme] == 1:
        x_pomme, y_pomme = np.random.randint(0, 16), np.random.randint(0, 16)

    plateau[x_pomme, y_pomme] = 2
    return plateau, x_pomme, y_pomme

@njit
def new_partie(rdn: RdN):
    plateau = np.zeros((16,16), np.int16)

    x_pomme, y_pomme = 9, 7
    plateau[7, 7], plateau[x_pomme, y_pomme] = 1, 2 # Tête du serpent et pomme

    serpent = List()
    serpent.append(np.array([7,7]))
    tete = serpent[0]

    temps_survie = 0
    while len(serpent) < 256:
        # Action
        dx, dy = action_joueur(rdn, plateau.copy(), tete)
        dernier_anneau, besoin_pomme = serpent[-1].copy(), False

        # Mouvements des anneaux
        for idx_anneau in range(len(serpent)-1,0,-1):
            serpent[idx_anneau] = serpent[idx_anneau-1].copy()

        # Mouvement de la tête
        tete = serpent[0]
        tete[0] += dx
        tete[1] += dy

        if not (0 <= tete[0] < 16 and 0 <= tete[1] < 16) or plateau[tete[0], tete[1]] == 1:
            break

        if plateau[tete[0], tete[1]] == 2:
            serpent.append(dernier_anneau)
            besoin_pomme = True if len(serpent) < 256 else False

        plateau, x_pomme, y_pomme = update_plateau(serpent, besoin_pomme, x_pomme, y_pomme)
        temps_survie += 1

    score = 1.0 - (temps_survie / 10000.0) - (len(serpent) / 256.0)
    return np.float32(score)

@njit
def evaluer_snake(rdn):
    note = np.float32(0)

    for _ in range(2):
        note += new_partie(rdn)
    return note / 2

if __name__ == '__main__':
    repartition = (256,4,256,1)

    rdn = rdn_aleatoire(repartition, sigmoide, False)
    note = evaluer_snake(rdn)
    print(note)