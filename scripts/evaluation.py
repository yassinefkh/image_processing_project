import pandas as pd
import numpy as np

# === Paramètres ===
csv_path = "data/test_set_evaluated.csv"

# === Chargement ===
df = pd.read_csv(csv_path)

# === Fonctions utilitaires ===
def compute_metrics(subset):
    mae = np.mean(subset['mae'])
    mse = np.mean(subset['mse'])
    return mae, mse

# === Évaluation globale ===
mae_global, mse_global = compute_metrics(df)
print("\n=== Évaluation globale ===")
print(f"MAE : {mae_global:.2f}")
print(f"MSE : {mse_global:.2f}")

# === Évaluation par difficulté visuelle ===
for diff in ['easy', 'medium', 'hard']:
    subset = df[df['difficulty'] == diff]
    if len(subset) == 0:
        print(f"\nAucune image pour la difficulté : {diff}")
        continue
    mae, mse = compute_metrics(subset)
    print(f"\n--- Difficulté : {diff} ---")
    print(f"Nombre d'images : {len(subset)}")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
