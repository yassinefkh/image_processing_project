import pandas as pd
from sklearn.model_selection import train_test_split

# === Paramètres ===
csv_path = "/Volumes/SSD/M1VMI/S2/image_processing_project/data/annotations.csv"
val_ratio = 0.3
test_ratio = 0.7
random_state = 42

# === Chargement ===
df = pd.read_csv(csv_path)

# === Catégorisation ===
def categorize(steps):
    if 2 <= steps <= 5:
        return "Peu"
    elif 6 <= steps <= 10:
        return "Moyen"
    elif 11 <= steps <= 22:
        return "Beaucoup"
    else:
        return "Extrême"

df["category"] = df["nombre_marches"].apply(categorize)

# === Affichage répartition ===
print("\n--- Répartition par catégorie ---\n")
category_counts = df["category"].value_counts()
for cat, count in category_counts.items():
    print(f"{cat} : {count} images ({(count / len(df)) * 100:.2f}%)")

# === Split stratifié ===
val_set, test_set = train_test_split(
    df,
    test_size=test_ratio,
    stratify=df["category"],
    random_state=random_state
)

# === Sauvegarde ===
val_set.to_csv("data/val_set.csv", index=False)
test_set.to_csv("data/test_set.csv", index=False)

print(f"\nSplit terminé : {len(val_set)} images (val), {len(test_set)} images (test)")
