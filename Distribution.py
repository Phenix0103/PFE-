import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pmdarima as pm
import os
from datetime import datetime

print("Bibliothèques importées avec succès.")

# Chemin d'accès au fichier Excel
excel_path = 'EtudeSB - Copie.xlsx'
xls = pd.ExcelFile(excel_path)

# Affichage des noms des feuilles pour comprendre la structure du classeur
sheet_names = xls.sheet_names
print(f"Feuilles disponibles dans le fichier: {sheet_names}")

# Chargement d'une feuille spécifique pour examiner sa structure
balance_sheet_2022 = pd.read_excel(excel_path, sheet_name='BILAN 2022')
balance_sheet_2022.head()

# Nettoyage des données en supprimant les valeurs négatives et NaN pour la colonne 'Secteur'
balance_sheet_2022['Secteur'] = pd.to_numeric(balance_sheet_2022['Secteur'], errors='coerce').fillna(0)
cleaned_data = balance_sheet_2022[balance_sheet_2022['Secteur'] >= 0]
cleaned_data.head()

# Exemple: Affichage de statistiques descriptives de la colonne 'Secteur'
cleaned_data['Secteur'].describe()

# Exemple de visualisation: Distribution des actifs bancaires
plt.figure(figsize=(10, 6))
plt.hist(cleaned_data['Secteur'], bins=20, color='skyblue')
plt.title('Distribution des Actifs Bancaires en 2022')
plt.xlabel('Actifs')
plt.ylabel('Fréquence')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Initialize a list to hold aggregated data
aggregated_data_list = []

# Load and aggregate data from each year
for year in range(2010, 2023):  # From 2010 to 2022
    sheet_name = f'BILAN {year}'
    df = pd.read_excel(excel_path, sheet_name=sheet_name).dropna(subset=['Secteur'])
    # Summarize total assets for the sector
    total_assets = df['Secteur'].sum()
    # Append the total to the aggregated data list
    aggregated_data_list.append({'Year': year, 'Total Assets': total_assets})

# Create DataFrame from the list of dictionaries
# Create DataFrame from the list of dictionaries
aggregated_data = pd.DataFrame(aggregated_data_list)

# Set 'Year' as the index
aggregated_data.set_index('Year', inplace=True)

# Utilisation de Plotly pour créer un graphique filtrable dynamique
fig = px.line(aggregated_data, x=aggregated_data.index, y='Total Assets',
              title='Total Banking Sector Assets in Tunisia (2010-2022)',
              labels={'x': 'Year', 'Total Assets': 'Total Assets (in thousands of dinars)'})

# Ajouter un slider et des boutons pour filtrer dynamiquement l'année
fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
    dict(count=1, label='1y', step='year', stepmode='backward'),
    dict(count=2, label='2y', step='year', stepmode='backward'),
    dict(count=5, label='5y', step='year', stepmode='backward'),
    dict(step='all')
])),
    rangeslider=dict(visible=True),
    type='date'
))
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Chemin d'accès à votre fichier Excel
excel_path = 'EtudeSB - Copie.xlsx'
xls = pd.ExcelFile(excel_path)

# Liste des banques sans doublons
banks = ['ATB', 'BNA', 'ATTIBK', 'BT', 'BTL', 'BTS', 'ABC', 'WIB', 'AMENBK', 'BIAT', 'STB', 'UBCI', 'UIB', 'BARAKA', 'BH', 'BTK', 'TSB', 'QNB', 'BTE']

# Initialiser un dictionnaire pour stocker les données pour chaque banque
bank_data = {bank: {'Years': [], 'Total Assets': []} for bank in banks}

# Années à itérer
years = range(2010, 2023)

# Itérer sur chaque année et extraire les données pour chaque banque
for year in years:
    sheet_name = f'BILAN {year}'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    for bank in banks:
        if bank in df.columns:
            total_assets = pd.to_numeric(df[bank], errors='coerce').fillna(0).sum()
        else:
            total_assets = np.nan  # Utiliser np.nan pour les années où les données de la banque sont absentes

        # Stocker les données dans le dictionnaire
        bank_data[bank]['Years'].append(year)
        bank_data[bank]['Total Assets'].append(total_assets)

# Créer un graphique pour chaque banque
for bank in banks:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bank_data[bank]['Years'], y=bank_data[bank]['Total Assets'], mode='lines+markers', name=bank))
    
    fig.update_layout(title=f'Total Assets Trend for {bank} (2010-2022)',
                      xaxis_title='Year',
                      yaxis_title='Total Assets',
                      legend_title='Bank')
    
    fig.show()

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Chemin d'accès à votre fichier Excel - à ajuster selon votre environnement
excel_path = 'EtudeSB - Copie.xlsx'
xls = pd.ExcelFile(excel_path)

# Liste des banques sans doublons
banks = ['ATB', 'BNA', 'ATTIBK', 'BT', 'BTS', 'ABC', 'WIB', 'AMENBK', 'BIAT', 'STB', 'UBCI', 'UIB', 'BARAKA', 'BH', 'BTK', 'QNB', 'BTE']

# Définir les années pour itérer
years = range(2010, 2023)

# Initialiser un DataFrame pour stocker les données de chaque banque
bank_data_df = pd.DataFrame(index=years)

# Itérer sur chaque année et extraire les données pour chaque banque
for year in years:
    sheet_name = f'BILAN {year}'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    for bank in banks:
        if bank in df.columns:
            total_assets = pd.to_numeric(df[bank], errors='coerce').fillna(0).sum()
            bank_data_df.loc[year, bank] = total_assets
        else:
            bank_data_df.loc[year, bank] = np.nan

# Création du graphique interactif avec Plotly
fig = go.Figure()

for bank in banks:
    # Ici, bank_data_df.index est utilisé pour l'axe des x
    fig.add_trace(go.Scatter(x=bank_data_df.index, y=bank_data_df[bank], mode='lines+markers', name=bank))

fig.update_layout(title='Total Assets Trend for Banks (2010-2022)',
                  xaxis_title='Year',
                  yaxis_title='Total Assets',
                  legend_title='Bank')

import pandas as pd
import matplotlib.pyplot as plt

# Chemin d'accès au fichier Excel original
excel_path = 'EtudeSB - Copie.xlsx'
# Chemin d'accès pour sauvegarder le nouveau fichier Excel
output_file_path = 'dat.xlsx'

# Initialiser une liste pour stocker les données agrégées
aggregated_data_list = []

# Charger et agréger les données de chaque année
for year in range(2010, 2023):  # De 2010 à 2022
    sheet_name = f'BILAN {year}'
    df = pd.read_excel(excel_path, sheet_name=sheet_name).dropna(subset=['Secteur'])
    # Résumer les actifs totaux pour le secteur
    total_assets = df['Secteur'].sum()
    # Ajouter le total à la liste des données agrégées
    aggregated_data_list.append({'Year': year, 'Total Assets': total_assets})

# Créer un DataFrame à partir de la liste de dictionnaires
aggregated_data = pd.DataFrame(aggregated_data_list)

# Définir 'Year' comme l'index
aggregated_data.set_index('Year', inplace=True)

# Affichage sous forme de tableau (dans la console ou le notebook Jupyter)
print(aggregated_data)

# Sauvegarder le DataFrame dans un fichier Excel
aggregated_data.to_excel(output_file_path)

# Graphique
plt.figure(figsize=(12, 6))
plt.plot(aggregated_data.index, aggregated_data['Total Assets'], marker='o', linestyle='-', color='b')
plt.title('Total des actifs du secteur bancaire en Tunisie (2010-2022)')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux (en milliers de dinars)')
plt.grid(True)
plt.xticks(aggregated_data.index, rotation=45)  # Rotation des étiquettes d'année pour une meilleure lisibilité
plt.tight_layout()  # Ajustement de la mise en page

# Afficher le graph

import pandas as pd
import matplotlib.pyplot as plt

# Chemin d'accès au fichier Excel
excel_path = 'EtudeSB - Copie.xlsx'

# Initialiser un dictionnaire pour stocker les données agrégées par banque
aggregated_data_by_bank = {}

# Charger et agréger les données de chaque année
for year in range(2010, 2023):  # De 2010 à 2022
    sheet_name = f'BILAN {year}'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # Itérer sur chaque colonne (banque) dans le DataFrame
    for bank in df.columns[1:]:  # Supposons que la première colonne est 'Secteur' et les autres sont les banques
        # Convertir les valeurs de la colonne banque en numérique, en ignorant les erreurs et en remplaçant les non numériques par NaN
        assets_numeric = pd.to_numeric(df[bank], errors='coerce')
        # Calculer la somme des actifs pour l'année en cours, en excluant les NaN
        total_assets_year = assets_numeric.sum()
        # Vérifier si la colonne (banque) existe déjà dans le dictionnaire
        if bank not in aggregated_data_by_bank:
            aggregated_data_by_bank[bank] = 0
        # Ajouter les actifs de l'année courante à la banque correspondante
        aggregated_data_by_bank[bank] += total_assets_year

# Conversion du dictionnaire en DataFrame pour une manipulation plus aisée
aggregated_data_df = pd.DataFrame(list(aggregated_data_by_bank.items()), columns=['Bank', 'Total Assets'])

# Tri des données par 'Total Assets' pour une meilleure visualisation
aggregated_data_df = aggregated_data_df.sort_values(by='Total Assets', ascending=False)
aggregated_data.to_excel(output_file_path)

# Graphique
plt.figure(figsize=(12, 8))
plt.bar(aggregated_data_df['Bank'], aggregated_data_df['Total Assets'], color='skyblue')
plt.title('Distribution des Actifs par Banque dans le Secteur Bancaire Tunisien (2010-2022)')
plt.xlabel('Banque')
plt.ylabel('Actifs Totaux')  # Modification ici pour enlever la référence spécifique à "en milliers de dinars"
plt.xticks(rotation=45, ha='right')  # Rotation des noms des banques pour une meilleure lisibilité
plt.tight_layout()

# Affichage du graphique
plt.show()

# Affichage du DataFrame
print(aggregated_data_df)
output_file_path = 'datBanques.xlsx'
aggregated_data_df.to_excel(output_file_path, index=False)


import matplotlib.pyplot as plt
import pandas as pd

# Chemin d'accès au fichier Excel - ajustez selon votre environnement
excel_path = 'EtudeSB - Copie.xlsx'

# Charger et agréger les données de chaque année
aggregated_data_list = []
for year in range(2010, 2023):  # De 2010 à 2022
    sheet_name = f'BILAN {year}'
    df = pd.read_excel(excel_path, sheet_name=sheet_name).dropna(subset=['Secteur'])
    total_assets = df['Secteur'].sum()
    aggregated_data_list.append({'Year': year, 'Total Assets': total_assets})

aggregated_data = pd.DataFrame(aggregated_data_list)
aggregated_data.set_index('Year', inplace=True)

# Visualisation de la tendance des actifs totaux
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Total Assets'], marker='o')
plt.title('Evolution des Actifs Totaux du Secteur Bancaire Tunisien (2010-2022)')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux')
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px

# Prepare the data for modeling
X = aggregated_data.index.values.reshape(-1, 1)  # Reshape years for sklearn
y = aggregated_data['Total Assets'].values
output_file_path = 'PredictionBA.xlsx'

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict for future years, e.g., 2023 to 2025
future_years = np.array([[2023], [2024], [2025],[2026], [2027], [2028],[2029], [2030], [2031],[2032],[2033],[2034],[2035]])
predictions = model.predict(future_years)
aggregated_data.to_excel(output_file_path)

# Add predictions to the plot
for i, year in enumerate(future_years.flatten()):
    aggregated_data.loc[year] = predictions[i]

# Update Plotly figure to include predictions
fig = px.line(aggregated_data, x=aggregated_data.index, y='Total Assets',
              title='Total Banking Sector Assets in Tunisia (2010-2025 Predicted)',
              labels={'x': 'Year', 'Total Assets': 'Total Assets (in thousands of dinars)'})
# (Reconfigure the slider and buttons as needed to include the new years)

fig.show()
# Création d'un DataFrame pour les prédictions
predictionsDA_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Total Assets': predictions
})

# Affichage du DataFrame
print(predictionsDA_df)
output_file_path = 'PredictionAnnée.xlsx'
predictionsDA_df.to_excel(output_file_path, index=False)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Chemin d'accès au fichier Excel
excel_path = 'EtudeSB - Copie.xlsx'
xls = pd.ExcelFile(excel_path)

# Charger les données agrégées
aggregated_data_path = 'dat.xlsx'
aggregated_data = pd.read_excel(aggregated_data_path, index_col='Year')

# Visualisation des actifs totaux sur la période étudiée
plt.figure(figsize=(12, 6))
plt.plot(aggregated_data.index, aggregated_data['Total Assets'], marker='o', linestyle='-', color='b')
plt.title('Total des actifs du secteur bancaire en Tunisie (2010-2022)')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux (en milliers de dinars)')
plt.grid(True)
plt.xticks(aggregated_data.index, rotation=45)
plt.tight_layout()
plt.show()

# Préparation des données pour le modèle
X = aggregated_data.index.values.reshape(-1, 1)  # Remodeler les années pour sklearn
y = aggregated_data['Total Assets'].values

# Créer et ajuster le modèle de régression polynomiale de degré 2
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# Prédire pour les années futures
future_years = np.array([[2023], [2024], [2025], [2026], [2027], [2028], [2029], [2030], [2031], [2032], [2033], [2034], [2035]])
predictions = model.predict(future_years)

# Ajouter les prédictions au DataFrame
aggregated_data_future = pd.DataFrame(index=future_years.flatten(), columns=['Predicted Total Assets'])
aggregated_data_future['Predicted Total Assets'] = predictions

# Concaténer les données prédites avec les données existantes
aggregated_data_combined = pd.concat([aggregated_data, aggregated_data_future])

# Afficher les prédictions
print("Prédictions pour les années 2023-2032:")
print(aggregated_data_future)

# Mettre à jour le graphique
plt.figure(figsize=(12, 6))
plt.plot(aggregated_data_combined.index, aggregated_data_combined['Total Assets'], marker='o', linestyle='-', color='b')
plt.title('Total des actifs du secteur bancaire en Tunisie (2010-2032)')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux (en milliers de dinars)')
plt.grid(True)
plt.xticks(aggregated_data_combined.index, rotation=45)
plt.tight_layout()
plt.show()


# Création d'un DataFrame pour les prédictions
predictionsDA_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Total Assets': predictions
})

# Affichage du DataFrame
print(predictionsDA_df)

# Analyse des tendances
# Calcul du taux de croissance annuel des actifs
growth_rates = aggregated_data['Total Assets'].pct_change().dropna()
print("Taux de croissance annuel des actifs :", growth_rates)

# Identifier les années avec une forte croissance ou réduction
print("Années avec les taux de croissance les plus élevés :", growth_rates.nlargest(3))
print("Années avec les taux de croissance les plus faibles :", growth_rates.nsmallest(3))

# Affichage du DataFrame
# Création d'un DataFrame pour les prédictions
aggregated_data_combined = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Total Assets': predictions
})

# Affichage du DataFrame
print(predictionsDA_df)

# Création d'un DataFrame pour les années de 2010 à 2035 avec les valeurs prédites
years_all = np.arange(2010, 2036)
predictions_all = model.predict(years_all.reshape(-1, 1))

# Création d'un DataFrame avec les années et les valeurs prédites
all_data = pd.DataFrame({
    'Year': years_all,
    'Predicted Total Assets': predictions_all
})

# Affichage du DataFrame
print(all_data)
output_file_path = 'TAB/Alldatafusion.xlsx'
all_data.to_excel(output_file_path, index=False)
from sklearn.model_selection import train_test_split

# Partitionnement des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Erreur absolue moyenne (MAE) :", mae)
print("Coefficient de détermination (R2) :", r2)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Préparation des données
X = all_data.index.values.reshape(-1, 1)  # Les années
y = all_data['Predicted Total Assets'].values  # Les actifs totaux

# Création d'une instance de modèle de régression polynomiale
degree = 3  # Vous pouvez ajuster le degré en fonction de la complexité des données
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Entraînement du modèle
polyreg.fit(X, y)

# Prédiction sur l'ensemble des données (pour visualisation)
y_pred_poly = polyreg.predict(X)

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue')  # Données réelles
plt.plot(X, y_pred_poly, color='red')  # Prédiction du modèle
plt.title('Régression Polynomiale sur les Actifs Totaux du Secteur Bancaire')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux')
plt.show()

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# Convertir l'index en série temporelle
ts = all_data['Predicted Total Assets']

# Définir et ajuster le modèle ARIMA
# p=2, d=1, q=2 sont des paramètres de départ, vous devrez peut-être les ajuster
model = ARIMA(ts, order=(2, 1, 2))
results = model.fit()

# Prévision
preds = results.forecast(steps=5)  # Prévoir les 5 prochaines années

# Affichage des prévisions
plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts, label='Données Réelles')
plt.plot(np.arange(ts.index[-1] + 1, ts.index[-1] + 6), preds, label='Prévisions ARIMA', linestyle='--')
plt.title('Prévision des Actifs Totaux avec ARIMA')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux')
plt.legend()
plt.show()

import pmdarima as pm

# Chargement de la série temporelle
ts = all_data['Predicted Total Assets']

# Modélisation ARIMA avec auto_arima pour trouver automatiquement un bon ensemble de paramètres
model = pm.auto_arima(ts, start_p=1, start_q=1,
                      test='adf',       # Utilisation du test ADF pour trouver l'ordre de différenciation 'd'
                      max_p=5, max_q=5, # Plages maximales pour 'p' et 'q'
                      m=1,              # Fréquence de la série temporelle (m=1 pour des données annuelles)
                      d=None,           # Laisser auto_arima déterminer l'ordre de différenciation
                      seasonal=False,   # Pas de saisonnalité
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

# Prévision
forecast = model.predict(n_periods=5)

# Affichage des prévisions
future_years = np.arange(ts.index[-1] + 1, ts.index[-1] + 6)
plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts, label='Données Historiques')
plt.plot(future_years, forecast, label='Prévisions ARIMA', color='red', linestyle='--')
plt.title('Prévisions des Actifs Totaux avec ARIMA Optimisé')
plt.xlabel('Année')
plt.ylabel('Actifs Totaux')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
banks = ['ATB', 'BNA', 'ATTIBK', 'BT', 'AMENBK', 'BIAT', 'STB', 'UBCI', 'UIB']  # Ajoutez plus de banques selon vos besoins

# Exemple de données (à remplacer par vos données réelles)
data = {
    'Year': np.repeat(np.arange(2010, 2035), len(banks)),
    'Bank': np.tile(banks, 2035-2010),
    'GDP': np.random.rand((2035-2010)*len(banks)),  # Exemple de données PIB
    'InterestRate': np.random.rand((2035-2010)*len(banks)),  # Exemple de taux d'intérêt
    'TotalAssets': np.random.rand((2035-2010)*len(banks)) * 10000  # Exemple d'actifs totaux
}

bank_data_df = pd.DataFrame(data)

# Encoder 'Bank' en variables numériques
bank_data_df = pd.get_dummies(bank_data_df, columns=['Bank'])
bank_data_df
# Séparation des caractéristiques et de la cible
X = bank_data_df.drop('TotalAssets', axis=1)
y = bank_data_df['TotalAssets']

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)
# Calcul du R²
r2 = r2_score(y_test, y_pred)
# Calcul de la MAE
mae = mean_absolute_error(y_test, y_pred)
# Calcul de la MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) :", mse)
print("Mean Absolute Error (MAE) :", mae)
print("Coefficient de détermination R² :", r2)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Chemin d'accès au fichier Excel - ajustez selon votre environnement
excel_path = 'EtudeSB - Copie.xlsx'
xls = pd.ExcelFile(excel_path)

# Liste des banques pour itérer
banks = ['ATB', 'BNA', 'ATTIBK', 'BT', 'AMENBK', 'BIAT', 'STB', 'UBCI', 'UIB']  # Exemple de banques

# Initialiser une liste pour stocker les performances des modèles
model_performance_list = []

# Itérer sur chaque banque pour entraîner un modèle et évaluer sa performance
for bank in banks:
    data_list = []
    for year in range(2010, 2023):  # De 2010 à 2022
        sheet_name = f'BILAN {year}'
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        if bank in df.columns:
            total_assets = pd.to_numeric(df[bank], errors='coerce').fillna(0).sum()
            data_list.append({'Year': year, 'Total Assets': total_assets})
    
    # Créer un DataFrame pour la banque actuelle
    bank_data = pd.DataFrame(data_list)
    
    # Si la banque a des données disponibles
    if not bank_data.empty:
        X = bank_data[['Year']] - bank_data['Year'].min()  # Normalisation de l'année
        y = bank_data['Total Assets']
        
        # Division des données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Création et entraînement du modèle de régression linéaire
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Prédiction sur l'ensemble de test et calcul du score R²
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Stocker la performance du modèle dans la liste
        model_performance_list.append({'Bank': bank, 'R2_Score': r2})

# Convertir la liste en DataFrame
model_performance = pd.DataFrame(model_performance_list)

# Afficher les performances des modèles
print(model_performance)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Supposons X_train, X_test, y_train, y_test sont déjà définis

models = {
    "Linear Regression": LinearRegression(),
    "KNN Regression": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(),
    # Ajoutez ou remplacez par d'autres modèles ici
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)  # Entraînement du modèle
    predictions = model.predict(X_test)  # Prédiction sur l'ensemble de test
    mse = mean_squared_error(y_test, predictions)  # Calcul du MSE
    r2 = r2_score(y_test, predictions)  # Calcul du score R2
    
    results[name] = {"MSE": mse, "R2": r2}  # Stockage des résultats

# Visualisation des résultats
labels, mse_scores, r2_scores = [], [], []
for model_name, metrics in results.items():
    labels.append(model_name)
    mse_scores.append(metrics["MSE"])
    r2_scores.append(metrics["R2"])

x = np.arange(len(labels))  # labels de localisation
width = 0.35  # largeur des barres

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mse_scores, width, label='MSE')
rects2 = ax.bar(x + width/2, r2_scores, width, label='R2')

# Ajout de textes pour labels, titre, axes ticks, etc.
ax.set_xlabel('Modèles')
ax.set_ylabel('Scores')
ax.set_title('Scores par modèle')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
