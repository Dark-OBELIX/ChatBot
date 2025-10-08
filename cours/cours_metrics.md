# 📊 Cours : Notions clés dans l'entraînement d’un réseau de neurones

## 1. Époque (Epoch)

Une **époque** correspond à **un passage complet de l’ensemble des données d’entraînement** à travers le modèle.  
Autrement dit, lorsque toutes les données ont été utilisées une fois pour ajuster les poids, on dit qu’une époque est terminée.

### 🌀 Exemple :
Si ton jeu de données contient 1 000 échantillons et que tu les envoies au modèle par lots de 100 (batch size = 100),  
alors **une époque = 10 itérations** (1000 / 100 = 10).

👉 En général, on entraîne un modèle sur plusieurs époques (ex. : 50 ou 100) pour lui permettre de mieux apprendre.

---

## 2. Fonction de perte (Loss Function)

La **fonction de perte** (ou *loss*) mesure **l’écart entre la sortie prédite et la sortie réelle**.  
C’est **la quantité que le modèle cherche à minimiser** pendant l’apprentissage.

### 🔧 Exemples de fonctions de perte :
| Type de tâche | Fonction de perte courante | Description |
|----------------|-----------------------------|--------------|
| Régression | Erreur quadratique moyenne (MSE) | Mesure la distance entre les valeurs réelles et prédites |
| Classification binaire | Binary Cross-Entropy | Compare les probabilités prédites à la vérité terrain |
| Classification multi-classes | Categorical Cross-Entropy | Généralisation pour plusieurs classes |

### 🧮 Exemple (Binary Cross-Entropy) :
\[
L = - \frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
\]

---

## 3. Précision (Accuracy)

La **précision (accuracy)** représente le **pourcentage de prédictions correctes** sur l’ensemble des données.  

### 🧮 Formule :
\[
\text{Accuracy} = \frac{\text{Nombre de bonnes prédictions}}{\text{Nombre total d'échantillons}}
\]

### 🧠 Exemple :
Si ton modèle a correctement prédit 90 échantillons sur 100,  
\( \text{Accuracy} = 0.90 \) → soit **90 % de précision**.

⚠️ Attention : l’accuracy n’est pas toujours représentative, notamment si les classes sont déséquilibrées.

---

## 4. Score F1 (F1-Score)

Le **F1-score** est une **moyenne harmonique** entre la **précision (precision)** et le **rappel (recall)**.  
Il est souvent utilisé lorsque les classes sont déséquilibrées.

### 🧮 Formules :
\[
\text{Precision} = \frac{\text{Vrais positifs}}{\text{Vrais positifs} + \text{Faux positifs}}
\]
\[
\text{Recall} = \frac{\text{Vrais positifs}}{\text{Vrais positifs} + \text{Faux négatifs}}
\]
\[
\text{F1} = 2 * \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}
\]

### 🧠 Exemple :
- Precision = 0.8  
- Recall = 0.6  
- F1 = 2 × (0.8×0.6)/(0.8+0.6) = 0.6857 → **≈ 68,6 %**

---

## 5. Suivi pendant l’entraînement

Durant l’entraînement d’un réseau de neurones, on suit souvent plusieurs métriques :

| Époque | Perte (Loss) | Accuracy | F1-score |
|:------:|:-------------:|:---------:|:---------:|
| 1 | 0.89 | 0.62 | 0.58 |
| 5 | 0.45 | 0.79 | 0.76 |
| 10 | 0.23 | 0.90 | 0.88 |

🔹 La **perte** doit généralement **diminuer** au fil des époques.  
🔹 Les **métriques de performance (Accuracy, F1)** doivent **augmenter** jusqu’à un point de stabilisation.  

---

## 6. Courbes d’apprentissage

Les courbes permettent de **visualiser la progression du modèle** :

- **Loss vs Epoch** → pour vérifier la convergence.  
- **Accuracy vs Epoch** → pour voir si le modèle apprend bien ou s’il sur-apprend (*overfitting*).

📈 Exemple attendu :
- La courbe de *loss* descend rapidement puis se stabilise.  
- La courbe d’*accuracy* monte jusqu’à un plateau.  

---

## 7. Résumé

> 🔹 **Époque (epoch)** : un passage complet sur le jeu de données.  
> 🔹 **Loss** : mesure de l’erreur à minimiser.  
> 🔹 **Accuracy** : proportion de bonnes prédictions.  
> 🔹 **F1-score** : équilibre entre précision et rappel, utile pour les classes déséquilibrées.

---

## 8. Bonnes pratiques

- Surveille toujours la **courbe de validation** (validation loss/accuracy).  
- Arrête l’entraînement avant le sur-apprentissage (technique d’**early stopping**).  
- Utilise le **F1-score** plutôt que l’accuracy quand les classes sont inégales.  
- Garde les **métriques par époque** pour comprendre la progression du modèle.

---

🧠 En résumé :  
Ces métriques sont les **indicateurs essentiels** pour évaluer et améliorer la qualité d’un réseau de neurones comme ton **MLP**.
