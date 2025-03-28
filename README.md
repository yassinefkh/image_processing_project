# Comptage de marches d'escalier

## Description du projet

Ce projet a pour objectif d’estimer automatiquement le **nombre de marches** présentes dans une image d'escalier, en exploitant à la fois :
- l'**image en niveaux de gris** de l'escalier,
- sa **carte de profondeur associée**.

L’approche mise en œuvre repose exclusivement sur des **techniques classiques de traitement d'image** (sans Deep Learning ni Machine Learning), combinant :
- **Analyse en Composantes Principales (PCA)** pour estimer l'axe principal de l'escalier,
- **Extraction du profil de profondeur** le long de cet axe,
- **Lissage du signal** et **détection des transitions** (marches) via un script.

À la fin du traitement, une **évaluation quantitative** est réalisée par comparaison aux annotations fournies (ground truth), en calculant les métriques **MSE** et **MAE**.

---

## Structure du projet

```
projet/
├── data/
│   ├── img/               # Images d'entrée (.jpg ou .png)
│   ├── depth/             # Cartes de profondeur correspondantes (_depth.png)
│   └── annotations.csv    # Fichier CSV contenant le nombre réel de marches par image
├── include/
│   └── ImageUtils.hpp     # Déclarations des fonctions utilitaires
├── src/
│   ├── ImageUtils.cpp     # Implémentation des fonctions de traitement d'image
│   └── main.cpp           # Programme principal
├── peak.py                # Script Python de détection des transitions dans le profil
├── Makefile               # Fichier de compilation
└── README.md              
```

---

## Dépendances

**C++ :**
- OpenCV ≥ 4.x (`core`, `imgproc`, `imgcodecs`, `highgui`, `ximgproc`, `photo`, `plot`)

**Python :**
- numpy
- scipy
- matplotlib (optionnel : uniquement pour l'affichage)

---

## Instructions d'utilisation

### Compilation

Dans le dossier du projet, exécuter :

```bash
make
```

Cela génère un exécutable nommé `stair_detector`.

### Exécution

Pour lancer l'analyse sur l'ensemble du dataset :

```bash
./stair_detector
```

Le programme :
1. Parcourt tous les fichiers d’images dans `data/img/`.
2. Pour chaque image :
    - Charge l’image et sa depth map.
    - Applique un prétraitement (flou gaussien + CLAHE).
    - Détecte les contours.
    - Calcule l’axe principal par PCA.
    - Extrait le profil de profondeur le long de cet axe.
    - Sauvegarde le profil dans un fichier `profil.csv`.
    - Exécute un **script Python** (`peak.py`) qui détecte les transitions et compte les marches.
3. Compare le nombre de marches détectées à la vérité terrain.
4. Calcule les métriques globales : **MSE** et **MAE**.

---

## Détails techniques

### 1. **Prétraitement**

L'image en niveaux de gris subit :
- Un **flou gaussien** (pour réduire le bruit).
- Une **égalisation adaptative (CLAHE)** pour améliorer le contraste local.

### 2. **Détection des contours**

Les contours sont extraits via des filtres de **Sobel** appliqués sur l'image prétraitée.

### 3. **Analyse en Composantes Principales (PCA)**

Les points de contour sont utilisés pour estimer l'axe principal de l'escalier :
- Le **vecteur directionnel principal** est calculé.

### 4. **Extraction du profil de profondeur**

Un profil est extrait en échantillonnant les valeurs de la depth map le long de la ligne PCA.

### 5. **Détection des transitions (marches)**

Le fichier Python `peak.py` :
- Lisse le profil avec un **filtre gaussien**.
- Utilise la fonction `find_peaks` de **scipy** avec une **prominence adaptative**.
- Identifie les positions de transitions (marches) et enregistre le nombre détecté dans `result.txt`.

### 6. **Évaluation**

À la fin, le programme calcule :
- **MSE (Mean Squared Error)** : moyenne des carrés des écarts entre le nombre détecté et la vérité terrain.
- **MAE (Mean Absolute Error)** : moyenne des valeurs absolues des écarts.

---

## Résultats et limitations

Le système est capable de détecter un nombre raisonnable de marches, sous certaines conditions :
- Les escaliers doivent être **visibles de face** avec un axe principal bien défini.
- Les depth maps doivent être de qualité suffisante (peu de bruit).

Les erreurs proviennent principalement :
- De profils bruités,
- D’escaliers partiellement visibles,
- D'erreurs de détection de contours lorsque les contrastes sont faibles.
- Des photos d'escaliers non représentatives d'une situation d'application réelle, telles que des escaliers photographiés de très loin ou partiellement obstrués par des objets volumineux.

**Aucun apprentissage automatique n'a été utilisé.**  
L'ensemble du traitement repose sur des méthodes classiques et déterministes de traitement d’image.

---

## Exemple de sortie

```
Image : Groupe2_Image11.jpg | Détecté : 7 | Vérité terrain : 7
Image : Groupe3_Image02.jpg | Détecté : 6 | Vérité terrain : 7
Image : Groupe1_Image05.jpg | Détecté : 8 | Vérité terrain : 8

=== Résultats ===
MSE : 0.33
MAE : 0.33
```


---

## Tentatives avec des méthodes classiques 

Au cours du développement, nous avons également expérimenté l'utilisation de la **Transformée de Hough** pour détecter les marches, en appliquant la détection de droites sur les images de contours.

Pour tenter d'améliorer les performances de cette approche, nous avons mis en place un **prétraitement rigoureux** :
- **Réduction du bruit** (flou gaussien, filtre médian).
- **Filtrage morphologique** (ouverture, fermeture).
- **Filtrage a posteriori** des lignes détectées : par **angle** (seulement les lignes proches de l’horizontal), par **excentricité** (droites trop courtes exclues), et par **fusion des lignes parallèles proches**.

Malgré ces efforts, cette approche par Hough Transform s'est révélée **inefficace et peu robuste** :
- Elle est **très sensible aux paramètres** (seuils, longueur minimale, distance maximale entre lignes, etc.).
- Les **paramètres varient fortement** d'une image à l'autre, ce qui empêche une configuration stable et généralisable.
- Même avec un filtrage post-traitement, la détection comporte souvent des **faux positifs ou des oublis**, notamment lorsque les marches sont partiellement visibles, obstruées ou mal contrastées.

En pratique, pour obtenir des résultats cohérents, il faudrait :
- soit accepter une **large marge d’erreur**,
- soit adapter manuellement les paramètres pour chaque image, ce qui **annule le caractère déterministe** et automatique du traitement.

Par comparaison, la méthode que nous avons retenue, basée sur les **profils de profondeur extraits le long de l'axe principal par PCA**, fournit des résultats **plus fiables et stables**. Toutefois, elle repose sur certaines **hypothèses fortes** :
- Les escaliers doivent être **vus de face** (l'axe principal est bien horizontal).
- Les escaliers doivent être **rectilignes** (pas d’escalier en colimaçon).
- La qualité et l'alignement de la **carte de profondeur** doivent être suffisants.

Cette méthode exploite donc l'information supplémentaire contenue dans la **depth map**, ce qui la rend plus performante, mais aussi **moins généralisable** à des cas où les informations de profondeur seraient indisponibles ou imprécises.


---

## Auteurs

Projet réalisé dans le cadre du Master 1 Informatique -  Vision et Machine Intelligente (VMI)  
**Matière : Introduction à l'Analyse d'images - KURTZ Camille**  
Année universitaire 2024-2025


Voici une **section complète** propre et bien rédigée que tu peux ajouter à ton README sous la section **Limites et perspectives** par exemple :
