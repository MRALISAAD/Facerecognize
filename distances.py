import numpy as np
from scipy.spatial import distance
import os

def manhattan_distance(v1, v2):
    """_summary_
    Compute manhattan (cityblock) distance
    dist = sum(|x_i - y_i|)
    dist = distance.cityblock(v1, v2)
    Args:
        v1 (_type_): query image signature
        v2 (_type_): offline signatures
    """
    return np.sum(np.abs(np.array(v1) - np.array(v2)))



def chebyshev_distance(v1, v2):
    """_summary_
    Compute manhattan distance
    dist = max(|x_i - y_i|)
    dist = distance.chebyshev(v1, v2)
    Args:
        v1 (_type_): query image signature
        v2 (_type_): offline signatures
    """
    return np.max(np.abs(np.array(v1) - np.array(v2)))

def canberra_distance(v1, v2):
    """_summary_
    Compute manhattan distance
    
    Args:
        v1 (_type_): query image signature
        v2 (_type_): offline signatures
    """
    return distance.canberra(v1, v2)

import numpy as np

def euclidean_distance(v1, v2):
    """Calcule la distance euclidienne entre deux vecteurs."""
    # Convertir les tableaux en numpy arrays de taille identique
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')

    # Calculer la distance euclidienne
    return np.sqrt(np.sum((v1 - v2)**2))

def retrieve_similar_images(signatures_db, query_features, distance='euclidean', num_results=5):
    """Retourne les images similaires en fonction des caractéristiques de la requête."""
    distances = []
    for features in signatures_db[:, :-1]:
        # Vérifier la compatibilité des dimensions
        if len(query_features) == len(features):
            if distance == 'euclidean':
                dist = euclidean_distance(query_features, features)
                distances.append(dist)
            else:
                print("Distance non prise en charge.")

    # Trier les distances et récupérer les indices des images similaires
    indices = np.argsort(distances)[:num_results]

    # Renvoyer les images similaires avec leurs distances
    similar_images = []
    for idx in indices:
        img_path = signatures_db[idx][-1]
        dist = distances[idx]
        label = os.path.splitext(img_path)[0]  # Récupérer le nom de l'image sans l'extension
        similar_images.append((img_path, dist, label))

    return similar_images


