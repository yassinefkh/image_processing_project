import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d, gaussian_filter1d

depth_values = np.loadtxt("profil.csv")

sobel_filter = np.array([-1, 0, 1])  
filtered_values = convolve1d(depth_values, sobel_filter, mode='reflect')

smoothed_values = gaussian_filter1d(depth_values, sigma=2)

def adaptive_prominence(total_length, base_prominence=1):
    """Réduit `prominence` progressivement vers la fin du profil."""
    return base_prominence * np.interp(total_length * 0.85, [0, total_length], [1, 0.5]) 

prominence_value = adaptive_prominence(len(smoothed_values))

peaks, _ = find_peaks(-smoothed_values, prominence=prominence_value, distance=20)

if len(peaks) == 0:
    peaks, _ = find_peaks(smoothed_values, prominence=prominence_value, distance=20)

plt.figure(figsize=(10, 5))
plt.plot(depth_values, label="Profil de profondeur (original)", alpha=0.5)
plt.plot(smoothed_values, label=f"Profil lissé (Gaussien, sigma=2)", linewidth=2, color="orange")
plt.plot(peaks, smoothed_values[peaks], "ro", label="Transitions détectées")  
plt.xlabel("Position le long de la ligne PCA")
plt.ylabel("Profondeur")
plt.legend()
plt.title("Détection des transitions (marches) avec Prominence Adaptative")
plt.grid()
plt.show()

print(f"Prominence utilisée : {prominence_value}")
print(f"Nombre de marches détectées : {len(peaks)}")
print("Indices des transitions détectées :", peaks)
