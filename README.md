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

### **Description de notre méthodologie**

Notre approche repose sur un pipeline classique de traitement d'image, suivi d'une analyse de profil de profondeur pour estimer le nombre de marches présentes sur chaque image. Voici les différentes étapes :

### 1. **Prétraitement**

Chaque image est d'abord convertie en **niveaux de gris**, puis soumise à deux traitements :
- **Flou gaussien** : permet de **réduire le bruit local** et les petites irrégularités susceptibles de perturber la détection des contours.
- **Égalisation adaptative de l'histogramme (CLAHE)** : améliore localement le **contraste** pour rendre les contours plus visibles, même en cas d'éclairage hétérogène.

---

### 2. **Détection des contours**

Nous appliquons un filtre de **Sobel** sur l'image prétraitée pour détecter les gradients d'intensité, révélateurs des **contours des marches**.  
Le résultat est ensuite seuillé afin d'obtenir une image binaire où seuls les contours marquants sont conservés.

---

### 3. **Analyse en Composantes Principales (PCA)**

L'étape clé suivante consiste à **analyser la structure géométrique des contours** :
- Nous extrayons les **coordonnées des points de contour** détectés.
- Nous appliquons une **Analyse en Composantes Principales (PCA)** sur ces points pour :
  - Calculer le **centre de gravité** des contours.
  - Identifier la **direction principale** des contours, c'est-à-dire l'axe le long duquel la variance des points est maximale.  
  Cet axe correspond à l'**orientation dominante de l'escalier** sur l'image.

---

### 4. **Extraction du profil de profondeur**

À partir de la direction principale obtenue par PCA :
- Nous **échantillonnons les valeurs de la carte de profondeur** (depth map) le long de cette ligne.
- Ce **profil de profondeur** reflète les variations de hauteur associées aux différentes marches de l'escalier.

Le profil est sauvegardé sous forme d'un fichier CSV.

---

### 5. **Détection des transitions (marches) sur les Depth Maps**

L'analyse du profil est réalisée par un **script Python** (`peak.py`) qui procède comme suit :
- **Lissage du profil** à l'aide d'un **filtre gaussien** pour atténuer les variations parasites.
- Application de la fonction `find_peaks` de la bibliothèque **scipy.signal** pour détecter les **creux (ou pics)** correspondant aux transitions entre les marches.
- Un seuil de **prominence adaptative** est utilisé, afin de tenir compte de la variabilité du profil.
- Le **nombre de marches détecté** est sauvegardé dans un fichier texte (`result.txt`).

---

### 6. **Évaluation**

Pour évaluer les performances de notre méthode, nous comparons le nombre de marches détecté avec la **vérité terrain** issue des annotations manuelles.

Nous calculons deux métriques classiques d'erreur :
- **MAE (Mean Absolute Error)** : moyenne des écarts absolus entre le nombre détecté et le nombre réel de marches.
  
  > Le MAE est privilégié ici car il s'agit d'une **tâche de régression** : nous estimons un **nombre** (de marches) et non une classe.  
  > L'accuracy ou les scores classiques de classification ne sont pas adaptés.  
  > Le MAE fournit une information claire : « en moyenne, combien d’erreurs de comptage faisons-nous par image ».

- **MSE (Mean Squared Error)** : moyenne des carrés des écarts.
  
  > Le MSE est également calculé pour information, mais il est moins pertinent dans notre cas : il est très sensible aux **erreurs extrêmes** (il punit fortement les grosses erreurs) et masque les tendances globales.

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

## Découpage et analyse des données

### Analyse des classes et catégorisation

Avant d'évaluer notre méthode, nous avons réalisé une analyse statistique approfondie de notre jeu de données, constitué de 93 images annotées avec le nombre de marches visibles sur chaque image. Cette analyse a révélé une forte hétérogénéité dans la distribution des classes. En effet, certaines valeurs du nombre de marches étaient très peu représentées : plusieurs classes ne comptaient qu'une ou deux occurrences dans l'ensemble du dataset. Ce déséquilibre important rendait difficile l'utilisation d'une stratification directe basée sur la valeur exacte du nombre de marches, notamment pour garantir une évaluation représentative.

Pour pallier ce problème, nous avons choisi de regrouper les classes selon des plages définies, afin de lisser cette distribution. Nous avons ainsi introduit une nouvelle variable **category**, définie comme suit :

| Catégorie | Plage du nombre de marches |
|:---------:|:-------------------------:|
| Peu      | 2 à 5 marches             |
| Moyen    | 6 à 10 marches            |
| Beaucoup | 11 à 22 marches           |
| Extrême  | 34 marches et plus        |

Cette catégorisation permet de regrouper les images selon des niveaux de complexité.

### Découpage du dataset

Une fois la catégorisation réalisée, nous avons procédé à un **découpage du dataset en deux sous-ensembles** :
- **Validation set** : 30% des données
- **Test set** : 70% des données

Le choix de ces proportions s'explique par la nature de notre méthode : notre approche ne repose pas sur un modèle nécessitant un apprentissage supervisé (il ne s'agit pas d'un modèle prédictif entraîné), mais nous avions besoin d'un ensemble de validation pour évaluer, pendant le développement, les performances intermédiaires de notre pipeline. L'ensemble de test est quant à lui utilisé pour l'évaluation finale.

Le découpage a été effectué à l’aide d’un **stratified split**, c’est-à-dire en préservant les proportions des catégories (`Peu`, `Moyen`, `Beaucoup`, `Extrême`) dans chacun des sous-ensembles. Cette stratification garantit que les deux ensembles restent représentatifs de la diversité des situations présentes dans notre base de données, tout en évitant que certaines catégories soient sous-représentées dans l’un ou l’autre.

**Répartition obtenue :**
| Catégorie | Total | Validation set | Test set |
|:---------:|:----:|:--------------:|:-------:|
| Peu      | 17   | 5              | 12      |
| Moyen    | 45   | 13             | 32      |
| Beaucoup | 28   | 8              | 20      |
| Extrême  | 3    | 1              | 2       |
| **Total** | 93   | 27             | 66      |

Cette démarche nous permet de garantir une évaluation objective, en évitant que les résultats soient biaisés par un déséquilibre ou une homogénéité excessive des images d’évaluation.


---

## Annotation de la difficulté et évaluation différenciée

Au-delà du nombre de marches, nous avons également choisi d’introduire une **catégorie qualitative de difficulté** pour chaque image, afin d’affiner l’analyse de nos résultats et de mieux interpréter les performances de notre méthode.

Cette difficulté a été annotée **manuellement** en fonction de plusieurs critères visuels observés dans les images :
- **Présence d'obstacles** ou objets partiellement obstruant les marches.
- **Perspective compliquée** (vue non frontale, fort angle). 
- **Éloignement important de l'escalier** ou marches très petites visuellement.

À partir de ces observations, chaque image a été classée dans l'une des trois catégories suivantes :
| Difficulté | Description |
|:---------:|:-----------------------------------------------------------:|
| Facile   | Escalier bien visible, peu d'obstacles, perspective frontale. |
| Moyenne  | Quelques obstacles, marches partiellement visibles, bruit modéré. |
| Difficile| Obstruction importante, perspective difficile, qualité dégradée.  |

Cette information a été ajoutée dans le fichier **annotations.csv** sous la colonne `difficulty`.

### Objectif de cette catégorisation

L’objectif de cette annotation est la suivante :
- **Évaluation fine** : mesurer la performance de notre méthode non seulement sur l’ensemble du test set, mais aussi séparément pour les images "Facile", "Moyenne" et "Difficile".


---

## Évaluation et analyse des performances

### Métriques utilisées

Dans ce projet, l'objectif est d'estimer **le nombre de marches d'un escalier** à partir d'une image et de sa carte de profondeur.  
Il s'agit donc d'une **tâche de régression** (prédire une valeur numérique), contrairement à une tâche de classification où l'on cherche à prédire une catégorie discrète.  
Par conséquent, des métriques comme l'**accuracy** ou le **score-F1**, couramment utilisées en classification, ne sont **pas adaptées** ici.

Nous avons choisi d'utiliser principalement le **MAE (Mean Absolute Error)**, qui correspond à l'erreur absolue moyenne entre le nombre de marches prédit et la vérité terrain.  
Cette métrique a l'avantage d'être **interprétable directement** : un MAE de 3 signifie qu'en moyenne, le système se trompe de 3 marches.

Nous avons également calculé le **MSE (Mean Squared Error)** à titre indicatif. Cependant, cette métrique est moins pertinente dans notre cas, car :
- Elle **pénalise fortement les erreurs importantes** (les grandes erreurs ont un poids quadratique).
- Elle est **plus difficile à interpréter directement** (un MSE de 45 n'a pas de signification concrète sur la tâche).
- Elle est **fortement influencée par les cas extrêmes** ou les anomalies.

C'est pourquoi, pour analyser les performances, nous nous basons principalement sur le **MAE**.

---

### Résultats globaux

L'évaluation globale sur l'ensemble du test set donne les résultats suivants :

| Difficulté | Nombre d'images | MAE  | MSE   |
|----------:|:--------------:|:----:|:----:|
| Easy     | 36            | 3.22 | 28.50 |
| Medium   | 12            | 4.50 | 66.67 |
| Hard     | 5             | 6.20 | 121.00 |
| **Global** | **53**        | **3.79** | **45.87** |

---

### Analyse

Nous avons également réalisé une évaluation **stratifiée** selon la difficulté visuelle des images (catégorie fournie manuellement) :
- **Easy :** Bonnes performances avec un MAE faible. Les images sont simples, bien cadrées, et l'estimation est généralement correcte.
- **Medium :** L'erreur augmente légèrement, souvent en raison de marches peu visibles ou de depth maps bruitées.
- **Hard :** Les erreurs sont élevées. Cela s'explique par la présence d'images avec obstacles, angles extrêmes ou artefacts visuels ne respectant pas nos hypothèses (vue de face, escalier droit).

Cette analyse démontre que la difficulté visuelle impacte directement les performances.  
De plus, les cas extrêmes ont une **influence disproportionnée sur le MSE** par rapport au MAE. C’est pourquoi nous privilégions le MAE, qui reflète mieux l’erreur moyenne réelle.

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

