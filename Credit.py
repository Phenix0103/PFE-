# Je vais d'abord importer les bibliothèques nécessaires pour travailler avec les données et les visualisations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Charger les données
file_path = 'EtudeSB - Copie.xlsx'
all_data = pd.read_excel(file_path, sheet_name=None)

# Initialiser un DataFrame pour le taux de croissance
credit_growth = pd.DataFrame()

# Calculer le taux de croissance du crédit pour chaque année et banque
for year in range(2011, 2023):
    current_year_data = all_data.get(f'BILAN {year}')
    previous_year_data = all_data.get(f'BILAN {year - 1}')
    
    if previous_year_data is not None:
        current_credits = current_year_data.set_index('En milliers de dinars').loc['Créances sur clientèle']
        previous_credits = previous_year_data.set_index('En milliers de dinars').loc['Créances sur clientèle']
        
        for bank in ['WIB', 'BARAKA']:
            if bank not in current_credits:
                current_credits[bank] = pd.NA
            if bank not in previous_credits:
                previous_credits[bank] = pd.NA
        
        growth_rate = (((current_credits / previous_credits) ** (1/1)) - 1) * 100
        credit_growth[year] = growth_rate

credit_growth.fillna(0, inplace=True)
credit_growth = credit_growth.round(2)
credit_growth = credit_growth.T

# Enregistrer les données calculées
output_file_path = 'Updated_EtudeSBF.xlsx'
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    credit_growth.to_excel(writer, sheet_name='Credit2')


## Visualisation 1: Taux de croissance du crédit
fig = px.line(credit_growth, labels={'value':'Taux de Croissance (%)', 'variable':'Banque', 'index':'Année'},
              title='Taux de Croissance Annuel du Crédit Bancaire par Banque')
fig.update_layout(xaxis_title='Année', yaxis_title='Taux de Croissance (%)', legend_title='Banque')


## Visualisation 2: Distribution des taux de croissance
for bank in credit_growth.columns:
    sns.histplot(credit_growth[bank], kde=True)
    plt.title(f'Distribution des taux de croissance pour {bank}')
    plt.xlabel('Taux de Croissance (%)')
    plt.ylabel('Fréquence')
    plt.show()


## Visualisation 3: Matrice de corrélation
correlation_matrix = credit_growth.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matrice de corrélation des taux de croissance du crédit entre les banques')
plt.show()
## Visualisation 4: Boxplot de la distribution des taux de croissance
sns.boxplot(data=credit_growth)
plt.title("Distribution des Taux de Croissance du Crédit par Année")
plt.ylabel("Taux de Croissance (%)")
plt.xlabel("Année")
plt.xticks(rotation=45)
plt.show()
import pandas as pd

# Enregistrer les statistiques descriptives dans un fichier Excel
description_file = 'description_credit_growth_banque.xlsx'
with pd.ExcelWriter(description_file) as writer:
    print("Statistiques descriptives des taux de croissance du crédit :")
    print(credit_growth.describe())
    credit_growth.describe().to_excel(writer, sheet_name='Descriptives')

    # Classification de la croissance
    def classify_growth(rate):
        if rate < 5:
            return 'faible'
        elif rate < 15:
            return 'moyen'
        else:
            return 'fort'

    classified_growth = credit_growth.applymap(classify_growth)
    print(classified_growth)
    classified_growth.to_excel(writer, sheet_name='Classification')

print(f"Les statistiques descriptives et la classification ont été enregistrées dans '{description_file}'.")

# Visualisation de la classification
mapped_values = classified_growth.replace({'faible': 1, 'moyen': 2, 'fort': 3})
plt.figure(figsize=(12, 8))
sns.heatmap(mapped_values, annot=True, fmt=".0f", cmap='RdYlGn', cbar_kws={'ticks': [1, 2, 3]}, cbar=True)
plt.xlabel('Année')
plt.ylabel('Banque')
plt.title('Classification de la Croissance des Banques par Année')
cbar = plt.gca().collections[0].colorbar
cbar.set_ticklabels(['faible', 'moyen', 'fort'])
plt.show()
import pandas as pd

# Calcul de la moyenne de croissance et classification finale
mean_growth = credit_growth.mean(axis=0)
classification = mean_growth.apply(classify_growth)
final_table = pd.DataFrame({'Moyenne de Croissance': mean_growth, 'Classification': classification})

# Enregistrer la table finale dans un fichier Excel
final_table_file = 'final_table_credit_growth.xlsx'
final_table.to_excel(final_table_file, sheet_name='Final Table')

print(f"La table finale a été enregistrée dans '{final_table_file}'.")
print(final_table)
# Visualisation finale: Nombre de banques par catégorie de croissance
classification_count = classification.value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=classification_count.index, y=classification_count.values, palette='RdYlGn')
plt.xlabel('Classification')
plt.ylabel('Nombre de Banques')
plt.title('Nombre de Banques par Catégorie de Croissance')
plt.show()
# Graphique à barres pour la moyenne de croissance et classification des banques
banques = final_table.index
moyenne_croissance = final_table['Moyenne de Croissance']
couleurs = final_table['Classification'].map({'faible': 'red', 'moyen': 'yellow', 'fort': 'green'})
plt.figure(figsize=(10, 8))
bars = plt.bar(banques, moyenne_croissance, color=couleurs)
plt.legend(handles=[plt.Rectangle((0,0),1,1, color=color) for color in ['red', 'yellow', 'green']],
           labels=['faible', 'moyen', 'fort'])
plt.title('Moyenne de Croissance et Classification des Banques')
plt.xlabel('Banque')
plt.ylabel('Moyenne de Croissance (%)')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
file_path = 'Updated_EtudeSBF.xlsx'
credit_growth = pd.read_excel(file_path, sheet_name='Credit2', index_col=0)

# Classer la croissance selon des seuils prédéfinis
def classify_growth(growth_rate):
    if growth_rate < 5:
        return 'Faible'
    elif growth_rate < 15:
        return 'Moyenne'
    else:
        return 'Élevée'

classified_growth = credit_growth.applymap(classify_growth)

# Reformater les données pour la visualisation
classified_growth = classified_growth.stack().reset_index()
classified_growth.columns = ['Année', 'Banque', 'Classification']

# Visualiser la classification de la croissance du crédit
plt.figure(figsize=(10, 6))
sns.countplot(data=classified_growth, x='Année', hue='Classification')
plt.title('Classification de la Croissance du Crédit par Année')
plt.xlabel('Année')
plt.ylabel('Nombre de Banques')
plt.legend(title='Classification')
plt.show()


import pandas as pd

# Charger le fichier Excel
file_path = 'EtudeSB - Copie.xlsx'

# Lire toutes les feuilles du fichier Excel dans un dictionnaire
# Lire toutes les feuilles du fichier Excel dans un dictionnaire pour comprendre sa structure
all_data = pd.read_excel(file_path, sheet_name=None)

# Afficher les noms des feuilles pour comprendre la structure des données
sheet_names = list(all_data.keys())
sheet_names
# Initialiser un dictionnaire pour stocker le total des crédits par année
total_credits_per_year = {}

# Itérer sur chaque feuille de bilan pour calculer le total des crédits par année
for year in range(2010, 2023):
    sheet_name = f'BILAN {year}'
    
    # Vérifier si la feuille existe dans le fichier Excel
    if sheet_name in sheet_names:
        # Lire les données de la feuille de bilan de l'année courante
        current_year_data = all_data[sheet_name]
        
        # Vérifier si 'Créances sur clientèle' est dans les index après les avoir définis
        # Si oui, sommer toutes les valeurs pour obtenir le total des crédits de l'année
        if 'En milliers de dinars' in current_year_data.columns:
            current_year_data.set_index('En milliers de dinars', inplace=True)
            if 'Créances sur clientèle' in current_year_data.index:
                total_credits = current_year_data.loc['Créances sur clientèle'].sum()
                total_credits_per_year[year] = total_credits

# Convertir le dictionnaire en DataFrame pour un traitement ultérieur
total_credits_df = pd.DataFrame(list(total_credits_per_year.items()), columns=['Year', 'Total Credits'])

# Calculer le taux de croissance annuel du crédit
total_credits_df['Growth Rate'] = total_credits_df['Total Credits'].pct_change() * 100

total_credits_df
import matplotlib.pyplot as plt

# Créer un graphique pour visualiser la croissance du crédit bancaire par année
plt.figure(figsize=(12, 6))
plt.plot(total_credits_df['Year'], total_credits_df['Growth Rate'], marker='o', linestyle='-', color='b')
plt.title('Croissance Annuelle du Crédit Bancaire', fontsize=16)
plt.xlabel('Année', fontsize=14)
plt.ylabel('Taux de Croissance (%)', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(total_credits_df['Year'], rotation=45)
plt.tight_layout()

# Sauvegarder le graphique
#plt_path = '/mnt/data/credit_growth_annual.png'
#plt.savefig(plt_path)

plt.show() #plt_path
import pandas as pd

# Votre code existant pour calculer les taux de croissance annuels et créer un DataFrame
# (Assumant que total_credits_df contient les données que vous souhaitez enregistrer)

# Spécifiez le chemin où vous souhaitez enregistrer le fichier Excel
excel_path = 'croissance_credit_annuel.xlsx'

# Enregistrez les données dans un fichier Excel
total_credits_df.to_excel(excel_path, index=False)

# Affichez un message de confirmation
print(f"Les données ont été enregistrées dans '{excel_path}'.")


total_credits_df
import seaborn as sns

# Créer un histogramme pour visualiser la distribution des taux de croissance globaux
plt.figure(figsize=(10, 6))
sns.histplot(total_credits_df['Growth Rate'].dropna(), kde=True, color='skyblue')
plt.title('Distribution des Taux de Croissance Annuelle Globale du Crédit Bancaire', fontsize=14)
plt.xlabel('Taux de Croissance (%)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Sauvegarder le graphique

plt.show()

print(credit_growth)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Imaginons que `credit_growth` soit votre DataFrame existant contenant les taux de croissance du crédit par année jusqu'en 2022
# Voici un exemple de structure de ce DataFrame
data = {
    'Année': np.arange(2011, 2023),
    'Taux de Croissance (%)': np.random.rand(12) * 10  # Généré aléatoirement pour l'exemple
}
credit_growth = pd.DataFrame(data)

# Préparation des données pour le modèle de prédiction
X = credit_growth['Année'].values.reshape(-1, 1)  # Années
y = credit_growth['Taux de Croissance (%)'].values  # Taux de croissance

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédiction des taux de croissance de 2023 à 2035
future_years = np.arange(2023, 2036).reshape(-1, 1)
predicted_growth = model.predict(future_years)

# Création d'un nouveau DataFrame pour les prédictions
predictions_df = pd.DataFrame({
    'Année': np.arange(2023, 2036),
    'Taux de Croissance Prévu (%)': predicted_growth
})

# Affichage des prédictions
print(predictions_df)

# Enregistrement des prédictions dans un fichier Excel
# Note: Changez le chemin en fonction de votre système d'exploitation et de votre environnement
output_file_path = 'predictions_croissance_credit.xlsx'

predictions_df.to_excel(output_file_path, index=False)

print(f"Le fichier a été enregistré avec succès à l'emplacement suivant: {output_file_path}")

import pandas as pd

# Charger le fichier Excel
file_path = 'EtudeSB - Copie.xlsx'

# Lire toutes les feuilles du fichier Excel dans un dictionnaire
all_data = pd.read_excel(file_path, sheet_name=None)

# Initialiser un dictionnaire pour stocker le total des crédits par année
total_credits_per_year = {}

# Itérer sur chaque feuille de bilan pour calculer le total des crédits par année
for year in range(2010, 2023):
    sheet_name = f'BILAN {year}'
    
    # Vérifier si la feuille existe dans le fichier Excel
    if sheet_name in all_data:
        # Lire les données de la feuille de bilan de l'année courante
        current_year_data = all_data[sheet_name]
        
        # Vérifier si 'Créances sur clientèle' est dans les index après les avoir définis
        # Si oui, sommer toutes les valeurs pour obtenir le total des crédits de l'année
        if 'En milliers de dinars' in current_year_data.columns:
            current_year_data.set_index('En milliers de dinars', inplace=True)
            if 'Créances sur clientèle' in current_year_data.index:
                total_credits = current_year_data.loc['Créances sur clientèle'].sum()
                total_credits_per_year[year] = total_credits

# Convertir le dictionnaire en DataFrame pour un traitement ultérieur
total_credits_df = pd.DataFrame(list(total_credits_per_year.items()), columns=['Year', 'Total Credits'])

# Calculer le taux de croissance annuel du crédit
total_credits_df['Growth Rate'] = total_credits_df['Total Credits'].pct_change() * 100

# Classification de la croissance
def classify_growth(rate):
    if rate < 5:
        return 'faible'
    elif rate < 15:
        return 'moyen'
    else:
        return 'fort'

# Appliquer la classification au DataFrame
total_credits_df['Classification'] = total_credits_df['Growth Rate'].apply(classify_growth)

# Spécifier le chemin où vous souhaitez enregistrer le fichier Excel
excel_path = 'croissance_credit_annuel_année.xlsx'

# Enregistrer les données dans un fichier Excel
with pd.ExcelWriter(excel_path) as writer:
    # Enregistrer le DataFrame de taux de croissance
    total_credits_df.to_excel(writer, sheet_name='growth_rate', index=False)

    # Enregistrer la feuille de classification
    classification_sheet = total_credits_df[['Year', 'Classification']]
    classification_sheet.to_excel(writer, sheet_name='Classification', index=False)

# Afficher un message de confirmation
print(f"Les données ont été enregistrées dans '{excel_path}'.")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Supposons que `credit_growth` soit votre DataFrame existant contenant les taux de croissance du crédit par année jusqu'en 2022
# Voici un exemple de structure de ce DataFrame
data = {
    'Année': np.arange(2011, 2023),
    'Taux de Croissance (%)': np.random.rand(12) * 10  # Généré aléatoirement pour l'exemple
}
credit_growth = pd.DataFrame(data)

# Préparation des données pour le modèle de prédiction
X = credit_growth['Année'].values.reshape(-1, 1)  # Années
y = credit_growth['Taux de Croissance (%)'].values  # Taux de croissance

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédiction des taux de croissance de 2023 à 2035
future_years = np.arange(2023, 2036).reshape(-1, 1)
predicted_growth = model.predict(future_years)

# Création d'un nouveau DataFrame pour les prédictions
predictions_df = pd.DataFrame({
    'Année': np.arange(2023, 2036),
    'Taux de Croissance Prévu (%)': predicted_growth
})

# Classification de la croissance prédite
def classify_predicted_growth(rate):
    if rate < 5:
        return 'faible'
    elif rate < 15:
        return 'moyen'
    else:
        return 'fort'

predictions_df['Classification'] = predictions_df['Taux de Croissance Prévu (%)'].apply(classify_predicted_growth)

# Affichage des prédictions
print("Prévisions de taux de croissance par année :")
print(predictions_df)

# Enregistrement des prédictions dans un fichier Excel
output_file_path = 'predictions_croissance_credit.xlsx'
predictions_df.to_excel(output_file_path, index=False)

print(f"Le fichier a été enregistré avec succès à l'emplacement suivant: {output_file_path}")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Supposons que credit_growth est votre DataFrame contenant la croissance du crédit pour chaque banque et chaque année
# Assurez-vous que le DataFrame est structuré de sorte que chaque colonne représente une banque et chaque ligne une année

# Création d'un modèle de prédiction pour chaque banque
predictions_df_list = []
for bank in credit_growth.columns:
    # Préparation des données pour le modèle de prédiction
    X = np.arange(2011, 2023).reshape(-1, 1)  # Les années
    y = credit_growth[bank].values  # Taux de croissance pour la banque courante
    
    # Création et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    
    # Prédiction des taux de croissance de 2023 à 2035 pour la banque courante
    future_years = np.arange(2023, 2036).reshape(-1, 1)
    predicted_growth = model.predict(future_years)
    
    # Création d'un DataFrame pour les prédictions de la banque courante
    bank_predictions_df = pd.DataFrame({
        'Année': np.arange(2023, 2036),
        f'Taux de Croissance Prévu (%) - {bank}': predicted_growth
    })
    
    predictions_df_list.append(bank_predictions_df)

# Concaténation de tous les DataFrames de prédictions dans un seul DataFrame
final_predictions_df = pd.concat(predictions_df_list, axis=1)

# Affichage des prédictions
print(final_predictions_df)

# Enregistrement des prédictions dans un fichier Excel
output_file_path = 'predictions_croissance_credit_par_banques.xlsx'
final_predictions_df.to_excel(output_file_path, index=False)
print(f"Le fichier a été enregistré avec succès à l'emplacement suivant: {output_file_path}")

import pandas as pd

# Charger le fichier Excel
file_path = 'Credit/Credit.xlsx'
credit_growth_df = pd.read_excel(file_path)

# Afficher les premières lignes pour vérifier la structure
credit_growth_df.head()

# Initialisation du modèle de régression linéaire
model = LinearRegression()

# Dictionnaire pour stocker les prédictions de chaque banque
predictions = {}

# Boucle sur chaque colonne (banque) du DataFrame, sauf la première colonne des années
for bank in credit_growth_df.columns[1:]:
    # Préparation des données
    X = credit_growth_df['Unnamed: 0'].values.reshape(-1, 1)  # Les années
    y = credit_growth_df[bank].values  # Taux de croissance du crédit pour la banque courante
    
    # Entraînement du modèle
    model.fit(X, y)
    
    # Prédiction des taux de croissance du crédit de 2023 à 2035 pour la banque courante
    predicted_growth_rates = model.predict(future_years)
    
    # Stockage des prédictions
    predictions[bank] = predicted_growth_rates

# Conversion du dictionnaire de prédictions en DataFrame
predictions_df = pd.DataFrame(predictions, index=np.arange(2023, 2036))

predictions_df

# Chemin vers le fichier de sortie
file_path = 'Credit/CreditBanque.xlsx'

# Enregistrer le DataFrame dans un fichier Excel
predictions_df.to_excel(file_path, index=True)
