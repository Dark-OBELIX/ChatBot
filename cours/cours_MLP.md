# 🤖 Cours : L’algorithme du Perceptron Multicouche (MLP)

## 1. Principe général

Le **Perceptron Multicouche (MLP)** est un **algorithme d’apprentissage supervisé** appartenant à la famille des **réseaux de neurones artificiels**.  
Il apprend à prédire une **sortie (y)** à partir d’une **entrée (x)** grâce à un **ensemble de poids ajustés automatiquement** durant l’entraînement.

➡️ Le but du MLP est de **trouver les poids** qui minimisent l’erreur entre la **prédiction du réseau** et la **valeur réelle attendue**.

---

## 2. Fonctionnement global

L’apprentissage d’un MLP se déroule en **deux phases principales** :

1. **Propagation avant (forward pass)**  
   - On fait circuler les données d’entrée à travers les couches successives du réseau.  
   - Chaque neurone applique une **transformation linéaire** suivie d’une **fonction d’activation non linéaire**.  
   - On obtient une **sortie prédite** à la fin du réseau.

2. **Rétropropagation (backward pass)**  
   - On compare la sortie prédite à la sortie réelle à l’aide d’une **fonction de coût** (ex. : erreur quadratique moyenne ou entropie croisée).  
   - L’erreur obtenue est **rétropropagée** dans le réseau.  
   - Les **poids sont mis à jour** selon la **descente de gradient** pour réduire cette erreur à chaque itération.

---

## 3. Étapes mathématiques détaillées

### 🧮 Étape 1 : Calcul de la propagation avant

Pour une couche \( l \) :
\[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
\]
\[
a^{(l)} = f(z^{(l)})
\]

où :
- \( W^{(l)} \) : matrice des poids de la couche \( l \)
- \( b^{(l)} \) : biais
- \( f \) : fonction d’activation (ex. : ReLU, Sigmoïde)
- \( a^{(l-1)} \) : sortie de la couche précédente

---

### 🧠 Étape 2 : Calcul de la fonction de coût

Pour mesurer l’erreur :
\[
J = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(y_i, \hat{y}_i)
\]
avec :
- \( y_i \) : vraie étiquette  
- \( \hat{y}_i \) : prédiction du modèle  
- \( \mathcal{L} \) : fonction de perte (souvent l’entropie croisée pour la classification)

---

### 🔁 Étape 3 : Rétropropagation du gradient

On calcule le gradient de l’erreur par rapport à chaque poids grâce à la **règle de la chaîne** :

\[
\frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
\]

\[
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)})
\]

Puis on met à jour les poids :

\[
W^{(l)} := W^{(l)} - \eta \frac{\partial J}{\partial W^{(l)}}
\]
\[
b^{(l)} := b^{(l)} - \eta \frac{\partial J}{\partial b^{(l)}}
\]

où :
- \( \eta \) est le **taux d’apprentissage** (learning rate)
- \( f'(z) \) est la dérivée de la fonction d’activation

---

## 4. Fonctions d’activation courantes

| Fonction | Expression | Rôle principal |
|-----------|-------------|----------------|
| **ReLU**  | \( f(x) = \max(0, x) \) | Brise la linéarité, accélère l’apprentissage |
| **Sigmoïde** | \( f(x) = \frac{1}{1+e^{-x}} \) | Donne des valeurs entre 0 et 1 (utile pour les probabilités) |
| **Tanh** | \( f(x) = \tanh(x) \) | Centrée sur 0, utile pour couches intermédiaires |
| **Softmax** | \( f_i(x) = \frac{e^{x_i}}{\sum_j e^{x_j}} \) | Produit une distribution de probabilité sur les classes |

---

## 5. Exemple simplifié d’apprentissage

Prenons un réseau avec :
- Entrée : 3 neurones (valeurs normalisées)
- Couche cachée : 4 neurones ReLU
- Sortie : 2 neurones Softmax (classification binaire)

**Processus d’apprentissage :**
1. Les données passent dans le réseau → sortie prédite.
2. L’erreur est calculée avec la vraie étiquette.
3. L’erreur est rétropropagée → mise à jour des poids.
4. Le processus est répété sur plusieurs **époques** jusqu’à convergence.

---

## 6. Avantages et limites du MLP

### ✅ Avantages
- Capable d’apprendre **des relations non linéaires complexes**.  
- Fonctionne bien sur des données vectorisées (ex. : texte, signaux, données tabulaires).  
- Architecture flexible et facilement personnalisable.

### ⚠️ Limites
- Exige un **temps d’entraînement** parfois long.  
- Nécessite **beaucoup de données** pour bien généraliser.  
- Moins performant que des architectures séquentielles (RNN, LSTM, Transformers) pour le texte ou les séries temporelles.

---

## 7. Résumé

> 🔹 Le MLP est une **architecture de réseau de neurones à propagation avant**.  
> 🔹 Il apprend par **descente de gradient** et **rétropropagation de l’erreur**.  
> 🔹 Il repose sur un **enchaînement de transformations linéaires + activations non linéaires**.  
> 🔹 Il est simple, puissant et constitue souvent la **base de compréhension** avant d’aborder les architectures plus avancées.
