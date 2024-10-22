
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
# Importation du dataset dans un dataframe
data=pd.read_csv('Social_Network_Ads.csv')
# Affichage dataset
print(data)
# Obtenir le nombre de colonnes dans la dataset
nombre_colonnes = len(data.columns)
print(nombre_colonnes)
#Renommer les colonnes du dataframe diabetes
Columns = ['ID-USer', 'H-gender','Age','ESTsalary','Purchase' ]
data.columns=Columns
print(data)
# Affichage de la taille de la dataset
print("Taille du jeu de données :")
print(data.shape)
# Vérification des valeurs manquantes dans le dataset
data.isnull().sum()
# description pour les colomuns numuriques
data.describe()
# Affichage des informations sur les colonnes
data.info()
# ululisation de d'un boucle pour separer les colomuns categories des autres 
nocategories = [var for var in data.columns if data[var].dtype!='O']

print('il y a {} nocategories variables\n'.format(len(nocategories)))
print('les nocategories variables sont :\n', nocategories)
# affichage de la dataset sans coloums de ctegories
print(data[nocategories])


# ululisation de d'un boucle pour separer les colomuns categories des autres 
categories = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categories )))
# affichage de le nombre de lignes  de ctegories 
print(data[categories ].value_counts())
# affichage de le le pourcentage  des lignes  de ctegories
print(data[categories ].value_counts()/np.float_(len(data)))

#affichage de rapartition binaire de purchase grace a la fonction hist  
plt.figure(figsize=(10, 6))
data['Purchase'].hist(color='skyblue', bins=10) # bins l'epesseur de de hist  
plt.title('Repartition des frequenes d achats des ads')
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.show()
#La corrélation linéaire, ou corrélation de Pearson, vise donc à établir une mesure de l'association linéaire, ou la force d'un lien entre deux variables X et Y (sur un diagramme, X est placé en abscisse et Y en ordonnée). Cette mesure est nommée coefficient de corrélation, ou r dans un rapport de corrélation
corr=data.iloc[:,1:].corr(method="pearson")
cmap = sns.diverging_palette(250, 0, 100, center='dark', as_cmap=True)#diverging_palette choix de couleurs
plt.figure(figsize=(8, 6))
sns.heatmap(corr, vmax=1, vmin=-1, cmap=cmap, square=True, annot=False, linewidths=0.2)# sns.heatma choix de cartes





#affichage de rapartition de l'histogramme de l'age et salary grace a la fonction hissplot

fig, axes = plt.subplots(1, 2, figsize=(10, 4) )
sns.histplot(data, ax=axes[0], x='Age', kde=True, color='r')
axes[0].set_title('Histogramme de Age')
sns.histplot(data, ax=axes[1], x='ESTsalary', kde=True, color='g')
axes[1].set_title('Histogramme de salary')




# Diviser l'ensemble de données en ensemble d'entraînement et ensemble de test 90/10
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)
# grace a la   StandardScaler Le Feature Scaling permet de préparer les données quand elles ont des échelles différentes.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# grace a la bib sklearn.naive_bayes Entraînement du modèle Naive Bayes sur l'ensemble d'entraînement
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Créer la matrice de confusion
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
#grace a la bib roc-curve Les courbes ROC (fonctions d'efficacité du récepteur) sont un outil important pour évaluer les performances d'un modèle de Machine Learning. 
plt.figure(figsize=(4, 6))
plt.plot(fpr, tpr, color='orange', lw=2, label='Courbede Roc'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2)
plt.title('Courbe ROC pour le modèle Naive Bayes')
plt.show()  