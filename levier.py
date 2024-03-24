import pandas as pd
import matplotlib.pyplot as plt
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xl = pd.ExcelFile(file_path)
# Function to extract financial leverage data from the balance sheet
# This function will try to handle cases where the lookup fails by returning None
def extract_leverage_data_updated(sheet_name, file_path):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    try:
        # Attempting to extract 'Total passifs' and 'TOTAL CAPITAUX PROPRES'
        total_liabilities = data.loc[data['En milliers de dinars'].str.contains('Total passifs', na=False), 'Secteur'].values[0]
        total_equity = data.loc[data['En milliers de dinars'].str.contains('TOTAL CAPITAUX PROPRES', na=False), 'Secteur'].values[0]
        return total_liabilities, total_equity
    except IndexError:
        # If the expected rows are not found, return None to indicate failure
        return None, None

# Initialize a DataFrame to store leverage results
leverage_results_updated = pd.DataFrame(columns=['Year', 'Financial Leverage'])

# Loop through each year and calculate the financial leverage
for year in range(2010, 2023):
    bilan_sheet_name = f'BILAN {year}'
    # Check if the sheet exists in the Excel file
    if bilan_sheet_name in xl.sheet_names:
        total_liabilities, total_equity = extract_leverage_data_updated(bilan_sheet_name, file_path)
        if total_liabilities is not None and total_equity is not None:
            # Calculate financial leverage if data was successfully extracted
            financial_leverage = total_liabilities / total_equity
            # Append to results DataFrame
            leverage_results_updated = pd.concat([leverage_results_updated, pd.DataFrame({
                'Year': [year],
                'Financial Leverage': [financial_leverage]
            })], ignore_index=True)

# Set 'Year' as the index
leverage_results_updated.set_index('Year', inplace=True)


# Définir le chemin complet du fichier Excel de sortie pour les résultats de l'effet de levier financier
output_leverage_excel_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/Levier/leverage_results.xlsx'

# Exporter le DataFrame des résultats de l'effet de levier financier dans un fichier Excel
leverage_results_updated.to_excel(output_leverage_excel_path)

# Plotting the financial leverage over the years
plt.figure(figsize=(10, 6))
plt.bar(leverage_results_updated.index, leverage_results_updated['Financial Leverage'], color='skyblue')
plt.title('Financial Leverage from 2010 to 2022')
plt.xlabel('Year')
plt.ylabel('Financial Leverage')
plt.grid(True)
plt.xticks(range(2010, 2023))
plt.tight_layout()
plt.show()

# Display the results as a table
print("Tableau des résultats de l'effet de levier financier :")
print(leverage_results_updated)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Charger les données
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/Levier/leverage_results.xlsx'
leverage_data = pd.read_excel(file_path, index_col='Year')

# Préparation des données
X = leverage_data.index.values.reshape(-1, 1)  # Années comme caractéristique
y = leverage_data['Financial Leverage'].values  # Effet de levier financier comme cible

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
print(f"RMSE sur l'ensemble de test: {rmse}")
print(f"R2 sur l'ensemble de test: {r2}")

# Tracé des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Valeurs réelles')
plt.plot(X_test, y_pred, color='red', label='Prédiction linéaire')
plt.title('Prédiction de l\'effet de levier financier')
plt.xlabel('Année')
plt.ylabel('Effet de levier financier')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Créer et entraîner le modèle de forêt aléatoire
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 arbres dans la forêt
model_rf.fit(X_train, y_train)

# Prédiction sur l'ensemble de test avec le modèle de forêt aléatoire
y_pred_rf = model_rf.predict(X_test)

# Évaluation du modèle de forêt aléatoire
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

# Affichage des résultats pour le modèle de forêt aléatoire
print(f"RMSE (Forêt Aléatoire) sur l'ensemble de test: {rmse_rf}")
print(f"R2 (Forêt Aléatoire) sur l'ensemble de test: {r2_rf}")

# Tracé des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Valeurs réelles')
plt.scatter(X_test, y_pred_rf, color='red', label='Prédiction Forêt Aléatoire')
plt.title('Prédiction de l\'effet de levier financier avec Forêt Aléatoire')
plt.xlabel('Année')
plt.ylabel('Effet de levier financier')
plt.legend()
plt.grid(True)
plt.show()

XGBOOST
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Création du modèle XGBoost
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Entraînement du modèle
model_xgb.fit(X_train, y_train)

# Prédictions
y_pred_xgb = model_xgb.predict(X_test)

# Évaluation
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"RMSE (XGBoost) sur l'ensemble de test: {rmse_xgb}")
print(f"R2 (XGBoost) sur l'ensemble de test: {r2_xgb}")

Arima
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Assurez-vous que 'Year' est de type datetime si ce n'est pas déjà le cas
leverage_results_updated.index = pd.to_datetime(leverage_results_updated.index, format='%Y')

# Effectuer la décomposition saisonnière
result = seasonal_decompose(leverage_results_updated['Financial Leverage'], model='additive')

# Tracer les composants décomposés
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()

from statsmodels.tsa.stattools import adfuller

result = adfuller(leverage_results_updated['Financial Leverage'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Interprétation
if result[1] > 0.05:
    print("La série n'est pas stationnaire.")
else:
    print("La série est stationnaire.")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Tracé de l'ACF et PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
plot_acf(leverage_results_updated['Financial Leverage'], ax=axes[0])
plot_pacf(leverage_results_updated['Financial Leverage'], ax=axes[1])
plt.show()

from statsmodels.tsa.arima.model import ARIMA
# Ou SARIMAX pour SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima.model import ARIMA

p = 1
d = 0
q = 1

model = ARIMA(leverage_results_updated['Financial Leverage'], order=(p, d, q))
model_fit = model.fit()

# Affichage du résumé du modèle
print(model_fit.summary())



# Prévisions
forecast = model_fit.forecast(steps=5)  # Prévoir 5 périodes dans le futur
print(forecast)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Obtenir les prédictions rétrospectives (in-sample forecast)
# Cela nous donnera les prédictions pour l'ensemble des données utilisées pour l'entraînement du modèle
in_sample_forecast = model_fit.predict(start=leverage_results_updated.index.min(), end=leverage_results_updated.index.max())

# Calculer le RMSE sur l'ensemble d'entraînement
rmse = np.sqrt(mean_squared_error(leverage_results_updated['Financial Leverage'], in_sample_forecast))
print("RMSE sur l'ensemble d'entraînement:", rmse)

# Calculer le R² sur l'ensemble d'entraînement
r2 = r2_score(leverage_results_updated['Financial Leverage'], in_sample_forecast)
print("R² sur l'ensemble d'entraînement:", r2)

# Tracer les valeurs réelles contre les prédictions rétrospectives
plt.figure(figsize=(10, 6))
plt.plot(leverage_results_updated.index, leverage_results_updated['Financial Leverage'], label='Valeurs Réelles')
plt.plot(leverage_results_updated.index, in_sample_forecast, label='Prédictions Rétrospectives', linestyle='--')
plt.title('Prédictions Rétrospectives avec le Modèle ARIMA')
plt.xlabel('Année')
plt.ylabel('Effet de levier financier')
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Define the path to the Excel file
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xl = pd.ExcelFile(file_path)

# Helper function to extract financial data
def extract_financial_data(sheet_name, file_path):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    metrics = {}
    try:
        metrics['total_liabilities'] = data.loc[data['En milliers de dinars'].str.contains('Total passifs', na=False), 'Secteur'].values[0]
        metrics['total_equity'] = data.loc[data['En milliers de dinars'].str.contains('TOTAL CAPITAUX PROPRES', na=False), 'Secteur'].values[0]
        metrics['total_assets'] = metrics['total_liabilities'] + metrics['total_equity']
        # Add more financial metrics if needed
    except IndexError:
        metrics = None
    return metrics

# Initialize a DataFrame to store enhanced financial analysis results
financial_analysis_results = pd.DataFrame()

# Loop through each sheet in the Excel file
for sheet_name in xl.sheet_names:
    if sheet_name.startswith('BILAN'):
        financial_data = extract_financial_data(sheet_name, file_path)
        if financial_data:
            year = int(sheet_name.split(' ')[1])
            financial_leverage = financial_data['total_liabilities'] / financial_data['total_equity']
            debt_to_equity = financial_data['total_liabilities'] / financial_data['total_equity']
            # Append to results DataFrame
            financial_analysis_results = pd.concat([financial_analysis_results, pd.DataFrame({
                'Year': [year],
                'Financial Leverage': [financial_leverage],
                'Debt-to-Equity Ratio': [debt_to_equity]
            })], ignore_index=True)

# Sort the results DataFrame by 'Year'
financial_analysis_results.sort_values('Year', inplace=True)
financial_analysis_results.set_index('Year', inplace=True)

# Export the enhanced financial analysis to a new Excel file
output_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/enhanced_financial_analysis.xlsx' 
financial_analysis_results.to_excel(output_path)

# Plotting multiple financial metrics over the years
plt.figure(figsize=(14, 8))
plt.subplot(311)
plt.plot(financial_analysis_results.index, financial_analysis_results['Financial Leverage'], marker='o', linestyle='-', color='blue')
plt.title('Financial Leverage over Years')
plt.xlabel('Year')
plt.ylabel('Financial Leverage')
plt.grid(True)


plt.subplot(313)
plt.plot(financial_analysis_results.index, financial_analysis_results['Debt-to-Equity Ratio'], marker='o', linestyle='-', color='green')
plt.title('Debt-to-Equity Ratio over Years')
plt.xlabel('Year')
plt.ylabel('Debt-to-Equity Ratio')
plt.grid(True)

plt.tight_layout()
plt.show()

# Return the path to the saved Excel file for download
output_path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Simulons des données pour l'exemple, remplacez ceci par votre propre préparation de données
X = financial_analysis_results.drop('Financial Leverage', axis=1)
y = financial_analysis_results['Financial Leverage']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R^2: {r2}")

# Optimisation: Cette étape peut inclure l'ajustement des hyperparamètres, le choix d'un modèle différent, ou la transformation des caractéristiques.

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
# Assuming `financial_analysis_results` is your DataFrame with historical data
# For demonstration, let's create a simple DataFrame
years = np.arange(2010, 2023).reshape(-1, 1)  # Example years
financial_leverage = np.random.rand(len(years))  # Example financial leverage data

# Fit the model
model = LinearRegression().fit(years, financial_leverage)
future_years = np.arange(2023, 2036).reshape(-1, 1)  # Years to predict
predicted_leverage = model.predict(future_years)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Financial Leverage': predicted_leverage
})
# Plot historical data
plt.plot(years, financial_leverage, label='Historical Financial Leverage')

# Plot predicted data
plt.plot(future_years, predicted_leverage, label='Predicted Financial Leverage', linestyle='--')

plt.xlabel('Year')
plt.ylabel('Financial Leverage')
plt.title('Financial Leverage Prediction from 2023 to 2035')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example historical data (simulated)
np.random.seed(0)  # For reproducible output
years = np.arange(2000, 2023).reshape(-1, 1)
financial_leverage = np.random.rand(len(years)) * 0.1 + np.linspace(0.5, 1.5, len(years))

# Fit the linear regression model
model = LinearRegression()
model.fit(years, financial_leverage)
# Years to predict
future_years = np.arange(2023, 2036).reshape(-1, 1)

# Make predictions
predicted_leverage = model.predict(future_years)

# Create a DataFrame for predictions
predictions_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Financial Leverage': predicted_leverage
})

print(predictions_df)
# Simulate "actual" future financial leverage for evaluation
actual_future_leverage = np.random.rand(len(future_years)) * 0.1 + np.linspace(1.5, 2.5, len(future_years))

# Calculate and print RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(actual_future_leverage, predicted_leverage))
print(f"RMSE: {rmse:.4f}")

# Calculate R² on historical data
r_squared = model.score(years, financial_leverage)
print(f"R² on historical data: {r_squared:.4f}")

# Simulate "actual" future financial leverage for evaluation (as before)
actual_future_leverage = np.random.rand(len(future_years)) * 0.1 + np.linspace(1.5, 2.5, len(future_years))

from sklearn.metrics import r2_score

# Calculate R² for future predictions
r_squared_future = r2_score(actual_future_leverage, predicted_leverage)
print(f"R² on future predictions: {r_squared_future:.4f}")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Simulated historical data
np.random.seed(0)  # For reproducible output
historical_years = np.arange(2010, 2023).reshape(-1, 1)
historical_leverage = np.random.rand(len(historical_years)) * 0.1 + np.linspace(0.5, 1.5, len(historical_years))

# Fit the linear regression model
model = LinearRegression()
model.fit(historical_years, historical_leverage)

# Future years to predict
future_years = np.arange(2023, 2036).reshape(-1, 1)
predicted_leverage = model.predict(future_years)

# Combine historical and future data
all_years = np.vstack((historical_years, future_years))
all_leverage = np.hstack((historical_leverage, predicted_leverage))

# Create DataFrame
df = pd.DataFrame({
    'Year': all_years.flatten(),
    'Financial Leverage': all_leverage
})

print(df)

Banque
def extract_leverage_data_by_bank(sheet_name, file_path):
    # Load the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Skip the 'En milliers de dinars' and 'Secteur' columns if they are not relevant
    relevant_columns = [col for col in df.columns if col not in ['En milliers de dinars', 'Secteur']]
    
    # Create a dictionary to hold your leverage data
    leverage_data = {}
    for bank_name in relevant_columns:
        # Assuming the leverage data you need is in the first row, adjust as necessary
        leverage_data[bank_name] = df[bank_name][0]
        
    return leverage_data

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assuming your existing function to extract leverage data remains the same.
# extract_leverage_data_by_bank(sheet_name, file_path)

# Load the Excel file
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xl = pd.ExcelFile(file_path)

# Create a dictionary to store data for each bank
data_by_bank = {}

# Loop through each sheet (year) and aggregate data by bank
for sheet_name in xl.sheet_names:
    if 'BILAN' in sheet_name:  # This checks if the sheet name contains 'BILAN'
        year = sheet_name.split()[-1]  # Extract the year from the sheet name
        leverage_data = extract_leverage_data_by_bank(sheet_name, file_path)
        for bank, leverage in leverage_data.items():
            if bank not in data_by_bank:
                data_by_bank[bank] = []
            data_by_bank[bank].append((year, leverage))

# Convert the aggregated data into a DataFrame for each bank and perform linear regression
for bank, data in data_by_bank.items():
    df = pd.DataFrame(data, columns=['Year', 'Financial Leverage']).dropna()
    df['Year'] = pd.to_numeric(df['Year'])
    
    if not df.empty:
        X = df[['Year']]
        y = df['Financial Leverage']
        model = LinearRegression().fit(X, y)
        
        # Predict future leverage
        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 11).reshape(-1, 1)
        predicted_leverage = model.predict(future_years)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df['Year'], y, label='Historical Financial Leverage')
        plt.plot(future_years.flatten(), predicted_leverage, label='Predicted Financial Leverage', linestyle='--')
        plt.title(f'Financial Leverage Prediction for {bank}')
        plt.xlabel('Year')
        plt.ylabel('Financial Leverage')
        plt.legend()
        plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulating historical financial leverage data for three banks
np.random.seed(0)  # For reproducible output
banks = ['ATB', 'BIAT', 'UBCI']
num_years = 23
data = []

# Générer des données de levier financier simulées pour chaque banque
for bank in banks:
    years = np.arange(2000, 2000 + num_years).reshape(-1, 1)
    financial_leverage = np.random.rand(num_years) * 0.1 + np.linspace(0.5, 1.5, num_years)
    for i in range(num_years):
        data.append([bank, years[i][0], financial_leverage[i]])

# Créer un DataFrame à partir des données
df = pd.DataFrame(data, columns=['Banque', 'Année', 'Levier Financier'])

# Initialiser les prédictions et les erreurs RMSE
predictions = []
rmses = []

# Entraîner un modèle de régression linéaire pour chaque banque et faire des prédictions
for bank in banks:
    bank_data = df[df['Banque'] == bank]
    model = LinearRegression()
    model.fit(bank_data[['Année']], bank_data['Levier Financier'])
    
    # Années à prédire
    future_years = np.arange(2023, 2023 + 10).reshape(-1, 1)
    predicted_leverage = model.predict(future_years)
    
    # Créer un DataFrame pour les prédictions pour chaque banque
    predictions_df = pd.DataFrame({
        'Année': future_years.flatten(),
        'Levier Financier Prévu': predicted_leverage
    })
    print(f"Prédictions pour {bank}:")
    print(predictions_df, "\n")
    
    # Ajouter les prédictions au tableau global des prédictions
    predictions.append((bank, predictions_df))
    
    # Simuler "réel" futur levier financier pour évaluation (pour exemple)
    actual_future_leverage = np.random.rand(len(future_years)) * 0.1 + np.linspace(1.5, 2.5, len(future_years))
    rmse = np.sqrt(mean_squared_error(actual_future_leverage, predicted_leverage))
    rmses.append((bank, rmse))

# Afficher l'erreur RMSE pour chaque banque
for bank, rmse in rmses:
    print(f"RMSE pour {bank}: {rmse:.4f}")


# Calculer R² pour chaque banque
for bank in banks:
    bank_data = df[df['Banque'] == bank]
    model = LinearRegression()
    X = bank_data[['Année']]
    y = bank_data['Levier Financier']
    model.fit(X, y)
    
    # Calculer R² sur les données historiques
    r_squared = model.score(X, y)
    print(f"R² pour {bank} sur les données historiques: {r_squared:.4f}")
# Pour chaque banque, ajuster un modèle, faire des prédictions et évaluer
for bank in banks:
    bank_data = df[df['Banque'] == bank]
    model = LinearRegression()
    X = bank_data[['Année']]
    y = bank_data['Levier Financier']
    model.fit(X, y)
    
    future_years = np.arange(2023, 2023 + 10).reshape(-1, 1)
    predicted_leverage = model.predict(future_years)
    
    # Simuler les valeurs réelles futures spécifiques à chaque banque pour l'évaluation
    np.random.seed(len(bank))  # Utiliser la longueur du nom de la banque comme graine pour la reproductibilité
    actual_future_leverage = np.random.rand(len(future_years)) * 0.1 + np.linspace(1.5, 2.5, len(future_years))
    
from sklearn.model_selection import cross_val_score

# Exemple d'utilisation de la validation croisée pour évaluer un modèle linéaire
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"Scores de validation croisée (R²) : {scores}")
print(f"Moyenne des scores de validation croisée (R²) : {np.mean(scores):.4f}")

from sklearn.cluster import KMeans

# Convertir les données de levier financier en un tableau 2D pour K-Means
X = leverage_results_updated.values.reshape(-1, 1)

# Déterminer le nombre optimal de clusters en utilisant la méthode du coude
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

# Tracer le graphique du coude pour trouver le nombre optimal de clusters
plt.figure(figsize=(8, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Distortion')
plt.title('Méthode du coude pour le nombre optimal de clusters')
plt.show()

# Basé sur le graphique du coude, choisissons un nombre de clusters (par exemple, 3)
num_clusters = 3

# Appliquer K-Means avec le nombre de clusters choisi
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Ajouter les étiquettes de cluster aux données de levier financier
leverage_results_updated['Cluster'] = kmeans.labels_

# Afficher les résultats du clustering
print("Résultats du clustering basé sur l'effet de levier financier :")
print(leverage_results_updated)

# Plotting the clusters
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_data = leverage_results_updated[leverage_results_updated['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['Financial Leverage'], label=f'Cluster {cluster+1}')

plt.title('Clustering des années en fonction de l\'effet de levier financier')
plt.xlabel('Année')
plt.ylabel('Effet de levier financier')
plt.xticks(range(2010, 2023))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Actual evaluation metrics from the code snippets you provided
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
r2_lr = r2_score(y_test, y_pred)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# Assuming ARIMA RMSE and R² are calculated similarly
# Please replace `y_arima_pred` with your actual ARIMA predictions
rmse_arima = np.sqrt(mean_squared_error(leverage_results_updated['Financial Leverage'], in_sample_forecast))
r2_arima = r2_score(leverage_results_updated['Financial Leverage'], in_sample_forecast)

# Data preparation
data = {
    'Method': ['Linear Regression', 'Random Forest', 'XGBoost', 'ARIMA'],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb, rmse_arima],
    'R2': [r2_lr, r2_rf, r2_xgb, r2_arima]
}

df = pd.DataFrame(data)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for RMSE
ax1.set_xlabel('Method')
ax1.set_ylabel('RMSE', color='tab:red')
ax1.bar(df['Method'], df['RMSE'], color='tab:red', width=0.4, label='RMSE')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a twin Axes sharing the xaxis for R²
ax2 = ax1.twinx()
ax2.set_ylabel('R²', color='tab:blue')
ax2.plot(df['Method'], df['R2'], color='tab:blue', marker='o', label='R²')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Title and legend
fig.tight_layout()
plt.title('Comparison of Evaluation Methods Based on Real Values')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()
