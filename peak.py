"""
Script de détection de marches à partir du profil de profondeur.

Ce script lit un fichier 'profil.csv' contenant un profil de profondeur extrait 
par le programme principal en C++, lisse le signal, puis détecte les transitions
(marches) en utilisant un détecteur de pics avec une prominence adaptative.

Le résultat (nombre de marches détectées) est écrit dans 'result.txt'.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d, gaussian_filter1d

depth_values = np.loadtxt("profil.csv")

sobel_filter = np.array([-1, 0, 1])
filtered_values = convolve1d(depth_values, sobel_filter, mode='reflect')

# lissage du signal par un filtre gaussien
smoothed_values = gaussian_filter1d(depth_values, sigma=2)

def adaptive_prominence(total_length, base_prominence=1):
    """
    Calcule une prominence adaptative en fonction de la longueur du signal.

    La prominence diminue progressivement vers la fin du profil pour éviter les faux positifs.

    Args:
        total_length (int): Longueur totale du profil.
        base_prominence (float): Valeur de prominence de base.

    Returns:
        float: Valeur de prominence adaptative.
    """
    return base_prominence * np.interp(total_length * 0.85, [0, total_length], [1, 0.5])

prominence_value = adaptive_prominence(len(smoothed_values))
peaks, _ = find_peaks(-smoothed_values, prominence=prominence_value, distance=20)

if len(peaks) == 0:
    peaks, _ = find_peaks(smoothed_values, prominence=prominence_value, distance=20)

# affichage optionnel 
"""
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(depth_values, label="Profil de profondeur", alpha=0.5)
plt.plot(smoothed_values, label="Profil lissé", linewidth=2, color="orange")
plt.plot(peaks, smoothed_values[peaks], "ro", label="Transitions détectées")
plt.xlabel("Position le long de la ligne PCA")
plt.ylabel("Profondeur")
plt.legend()
plt.title("Détection des transitions (marches)")
plt.grid()
plt.show()
""" 

print(f"Nombre de marches détectées : {len(peaks)}")

with open("result.txt", "w") as f:
    f.write(str(len(peaks)) + "\n")
