import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Chargement des données
df = pd.read_csv("data/test_set_evaluated.csv")

def compute_correlation(subset, label):
    pearson_corr, pearson_p = pearsonr(subset['steps'], subset['mae'])
    spearman_corr, spearman_p = spearmanr(subset['steps'], subset['mae'])
    print(f"\n--- Difficulté : {label} ---")
    print(f"Nombre d'images : {len(subset)}")
    print(f"Corrélation de Pearson : {pearson_corr:.2f} (p-value={pearson_p:.4f})")
    print(f"Corrélation de Spearman : {spearman_corr:.2f} (p-value={spearman_p:.4f})")

# Corrélation globale
print("\n=== Corrélation globale ===")
compute_correlation(df, "Toutes")

# Corrélation par difficulté
for difficulty in df['difficulty'].unique():
    subset = df[df['difficulty'] == difficulty]
    compute_correlation(subset, difficulty)
