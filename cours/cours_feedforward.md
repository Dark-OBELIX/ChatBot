# 🧠 Cours : Les réseaux de neurones à propagation avant (Feedforward Neural Networks)

## 1. Définition générale

Un **réseau de neurones à propagation avant** (*Feedforward Neural
Network*, ou FFNN) est un modèle d'apprentissage supervisé dans lequel
les informations circulent **uniquement dans un sens** :\
de la **couche d'entrée** vers la **couche de sortie**, en passant par
une ou plusieurs **couches cachées**.

Ce type de réseau ne comporte **aucune rétroaction** ni **mémoire
temporelle**.\
Chaque entrée est traitée indépendamment des précédentes.

------------------------------------------------------------------------

## 2. Structure d'un réseau à propagation avant

Un réseau de ce type est généralement composé de :

1.  **Couche d'entrée**
    -   Reçoit les données sous forme numérique (dans ton cas, les
        vecteurs TF-IDF représentant les phrases).
2.  **Couches cachées**
    -   Réalisent des transformations internes à l'aide de neurones
        connectés et de fonctions d'activation non linéaires, comme
        **ReLU**.\
    -   Peuvent inclure des mécanismes de régularisation comme le
        **Dropout** pour éviter le surapprentissage.
3.  **Couche de sortie**
    -   Produit les probabilités associées à chaque classe de sortie
        (dans ton projet, chaque type de réponse du chatbot).\
    -   L'activation finale est généralement une **fonction Softmax**,
        adaptée à la classification multi-classes.

------------------------------------------------------------------------

## 3. Le perceptron multicouche (MLP)

Le **Perceptron Multicouche (MLP)** est une implémentation classique
d'un réseau à propagation avant.\
Il se compose d'au moins : - une couche d'entrée,\
- une couche cachée,\
- et une couche de sortie.

Dans ton projet : - Entrée → vecteurs TF-IDF\
- Couche cachée → 8 neurones + ReLU + Dropout(0.3)\
- Sortie → probabilité de chaque catégorie de réponse

Le MLP apprend à associer chaque phrase vectorisée à une **intention**
du chatbot.

------------------------------------------------------------------------

## 4. Terminologie française

  -----------------------------------------------------------------------
  Terme anglais  Traduction française         Autre traduction possible
                 courante                     
  -------------- ---------------------------- ---------------------------
  Feedforward    Réseau de neurones à         Réseau de neurones à action
  Neural Network propagation avant            directe

  Multilayer     Perceptron multicouche       ---
  Perceptron                                  
  (MLP)                                       
  -----------------------------------------------------------------------

Ainsi, on peut dire que ton modèle est : \> 🔹 un **Perceptron
Multicouche**,\
\> 🔹 appartenant à la famille des **Réseaux de neurones à propagation
avant**.

------------------------------------------------------------------------

## 5. Avantages et limites

### ✅ Avantages

-   Simplicité conceptuelle et rapidité d'entraînement.\
-   Efficace pour des tâches de classification simples sur des données
    tabulaires ou textuelles vectorisées.

### ⚠️ Limites

-   Ne prend pas en compte la **structure ou le sens contextuel** du
    langage.\
-   Moins performant que des architectures modernes (RNN, LSTM,
    Transformers) pour le traitement du langage naturel.
