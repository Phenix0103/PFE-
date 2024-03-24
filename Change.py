import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Chemin du fichier Excel
excel_path = 'changee.xlsx'

# Nom de la feuille de calcul à lire
sheet_name = 'Feuil1'

# Lecture du fichier Excel dans un DataFrame
change_df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Filtration des colonnes nécessaires
columns_to_keep = ['Dollar des USA', 'Yen Japonais', 'EURO']
date_column = [col for col in change_df.columns if 'Date' in col][0]
change_df_filtered = change_df[[date_column] + columns_to_keep]

# Nettoyage des données
change_df_filtered[date_column] = pd.to_datetime(change_df_filtered[date_column], errors='coerce')
change_df_filtered = change_df_filtered.dropna().replace(',', '.', regex=True)

for col in columns_to_keep:
    change_df_filtered[col] = change_df_filtered[col].astype(float)

# Affichage des premières lignes pour vérification
print(change_df_filtered.head())


# Sélection de la variable à prédire
target = 'Dollar des USA'

# Séparation des caractéristiques et de la cible
X = change_df_filtered[['Yen Japonais', 'EURO']]
y = change_df_filtered[target]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")


# Sélection de la variable à prédire
target = 'EURO'

# Séparation des caractéristiques et de la cible
X = change_df_filtered[['Yen Japonais', 'Dollar des USA']]
y = change_df_filtered[target]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

# Sélection de la variable à prédire
target = 'Yen Japonais'

# Séparation des caractéristiques et de la cible
X = change_df_filtered[['EURO', 'Dollar des USA']]
y = change_df_filtered[target]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# Chargement des données
excel_path = 'changee.xlsx'
sheet_name = 'Feuil1'
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Sélection et nettoyage des données (supposons que cette étape est déjà bien définie)
# Assuming 'Yen Japonais' and 'EURO' are the columns that need correction
# Example of converting columns to numeric if not already
df['Yen Japonais'] = pd.to_numeric(df['Yen Japonais'], errors='coerce')
df['EURO'] = pd.to_numeric(df['EURO'], errors='coerce')


# Division des données en ensembles d'apprentissage et de test
X = df[['Yen Japonais', 'EURO']]  # Supposons ces comme caractéristiques
y = df['Dollar des USA']  # Variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction du pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
    ('knn', KNeighborsRegressor())
])

# Sélection du paramètre K via validation croisée
param_grid = {'knn__n_neighbors': range(1, 30)}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Meilleur modèle et paramètres
print("Meilleur paramètre K:", grid_search.best_params_)
# Check for NaN values
print(df.isnull().sum())

# Example way to handle NaN values, such as filling with the mean of the column
df = df.fillna(df.mean())

# Évaluation du modèle
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")
import pandas as pd

# Chemin du fichier Excel
excel_path = r"changee.xlsx"

# Nom de la feuille de calcul à lire
sheet_name = 'Feuil1'

# Lecture du fichier Excel dans un DataFrame
change_df_cleaned = pd.read_excel(excel_path, sheet_name=sheet_name)

# Liste des colonnes à conserver
columns_to_keep = ['Dollar des USA', 'Yen Japonais', 'EURO']

# Vérifier si les noms de colonne existent dans le DataFrame
if all(col in change_df_cleaned.columns for col in columns_to_keep):
    # Filtrer le DataFrame
    change_df_filtered = change_df_cleaned[columns_to_keep]
else:
    # Afficher un message d'erreur
    print("Les noms de colonne ne correspondent pas. Veuillez vérifier.")

# Le nom de la colonne de date est différent et doit être inclus également pour l'analyse temporelle
# Nous allons trouver le nom de la colonne contenant les informations de date
date_column = change_df_cleaned.columns[change_df_cleaned.columns.str.contains('Date')][0]

# Inclure la colonne de date dans notre DataFrame filtré
change_df_filtered_with_date = change_df_cleaned[[date_column] + columns_to_keep]

# Nettoyage des données :
# - Conversion de la colonne de date en datetime
# - Remplacement des virgules par des points dans les valeurs de taux de change pour les convertir en type float
change_df_filtered_with_date[date_column] = pd.to_datetime(change_df_filtered_with_date[date_column], errors='coerce')
change_df_filtered_with_date = change_df_filtered_with_date.replace(',', '.', regex=True).dropna()

# Conversion des valeurs de taux de change de string en float
for col in columns_to_keep:
    try:
        change_df_filtered_with_date[col] = change_df_filtered_with_date[col].astype(float)
    except ValueError as e:
        print(f"Erreur lors de la conversion de la colonne {col}: {e}")

# Affichage des premières lignes du DataFrame nettoyé
change_df_filtered_with_date.head()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Remove commas and convert values to float
change_df_filtered = change_df_filtered.replace(',', '.', regex=True)
change_df_filtered = change_df_filtered.astype(float)

# Continue with your analysis as before
for currency in columns_to_keep:
    print(f'\nAnalyse de {currency}:')
    result = adfuller(change_df_filtered[currency])
    print(f'Statistique ADF : {result[0]}')
    print(f'P-value : {result[1]}')
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print(f'    {key}: {value}')


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
# Remplacement des virgules par des points et conversion des chaînes de caractères en nombres flottants

for currency in columns_to_keep:
    
    print(f'\nAnalyse de {currency}:')
    result = adfuller(change_df_filtered[currency])
    print(f'Statistique ADF : {result[0]}')
    print(f'P-value : {result[1]}')
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print(f'    {key}: {value}')



for currency in columns_to_keep:
    # Préparation de la série différenciée si nécessaire
    if result[1] > 0.05:
        change_df_filtered[f'{currency}_diff'] = change_df_filtered[currency].diff().dropna()
        diff_column = f'{currency}_diff'
    else:
        diff_column = currency

    # Calculer un nombre approprié de lags : min entre 10 et la moitié de la taille de l'échantillon
    max_lags = min(10, len(change_df_filtered[diff_column].dropna()) // 2 - 1)

    # Vérifier si max_lags est suffisamment grand
    if max_lags > 0:
        # Tracer ACF et PACF avec le nombre ajusté de lags
        plt.figure(figsize=(14, 7))
        plt.subplot(121)
        plot_acf(change_df_filtered[diff_column].dropna(), ax=plt.gca(), lags=max_lags)
        plt.title(f'ACF de {currency}')
        plt.subplot(122)
        plot_pacf(change_df_filtered[diff_column].dropna(), ax=plt.gca(), lags=max_lags)
        plt.title(f'PACF de {currency}')
        plt.tight_layout()
        plt.show()
    else:
        print(f'Pas assez de données pour tracer ACF et PACF pour {currency}')

for currency in columns_to_keep:
    # Construction et ajustement du modèle ARIMA
    if 'diff' in diff_column:  # Si une différenciation a été effectuée
        model_arima = ARIMA(change_df_filtered[currency], order=(1, 1, 1))
    else:
        model_arima = ARIMA(change_df_filtered[currency], order=(1, 0, 1))
    model_arima_fit = model_arima.fit()

    # Diagnostic du modèle
    model_arima_fit.plot_diagnostics(figsize=(14, 8))
    plt.show()

    # Affichage du résumé du modèle
    print(model_arima_fit.summary())

from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings

warnings.filterwarnings("ignore")  # Ignorer les avertissements

# Devise à analyser
currencies = ['Dollar des USA', 'EURO', 'Yen Japonais']

# Définition de la grille de recherche
p = d = q = range(0, 3)  # Plage de paramètres
pdq_combinations = list(itertools.product(p, d, q))

best_params = {}

for currency in currencies:
    best_aic = float("inf")
    best_combination = (0, 0, 0)
    for combination in pdq_combinations:
        try:
            model = ARIMA(change_df_filtered_with_date[currency], order=combination)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_combination = combination
        except:
            continue
    best_params[currency] = best_combination
    print(f"{currency} - Meilleure combinaison : {best_combination} avec un AIC de {best_aic}")

from statsmodels.tsa.arima.model import ARIMA

# Ajustement du modèle ARIMA avec les paramètres optimisés
model_optimized = ARIMA(change_df_filtered_with_date['Dollar des USA'], order=(0, 2, 1))
model_optimized_fit = model_optimized.fit()

# Affichage du résumé du modèle pour inspection
print(model_optimized_fit.summary())
from sklearn.metrics import r2_score

# Prévisions sur les données historiques
predictions = model_optimized_fit.predict(start=change_df_filtered_with_date.index[0], end=change_df_filtered_with_date.index[-1], dynamic=False)

# Calcul de R^2
r2 = r2_score(change_df_filtered_with_date['Dollar des USA'][2:], predictions[2:])  # On exclut les premiers points à cause de la différenciation d'ordre 2
print(f"R^2 score: {r2:.4f}")

from statsmodels.tsa.arima.model import ARIMA

# Ajustement du modèle ARIMA avec les paramètres optimisés
model_optimized = ARIMA(change_df_filtered_with_date['EURO'], order=(1, 1, 0))
model_optimized_fit = model_optimized.fit()

# Affichage du résumé du modèle pour inspection
print(model_optimized_fit.summary())
from sklearn.metrics import r2_score

# Prévisions sur les données historiques
predictions = model_optimized_fit.predict(start=change_df_filtered_with_date.index[0], end=change_df_filtered_with_date.index[-1], dynamic=False)

# Calcul de R^2
r2 = r2_score(change_df_filtered_with_date['EURO'][2:], predictions[2:])  # On exclut les premiers points à cause de la différenciation d'ordre 2
print(f"R^2 score: {r2:.4f}")


from statsmodels.tsa.arima.model import ARIMA

# Ajustement du modèle ARIMA avec les paramètres optimisés
model_optimized = ARIMA(change_df_filtered_with_date['Yen Japonais'], order=(0,2,0))
model_optimized_fit = model_optimized.fit()

# Affichage du résumé du modèle pour inspection
print(model_optimized_fit.summary())
from sklearn.metrics import r2_score

# Prévisions sur les données historiques
predictions = model_optimized_fit.predict(start=change_df_filtered_with_date.index[0], end=change_df_filtered_with_date.index[-1], dynamic=False)

# Calcul de R^2
r2 = r2_score(change_df_filtered_with_date['Yen Japonais'][2:], predictions[2:])  # On exclut les premiers points à cause de la différenciation d'ordre 2
print(f"R^2 score: {r2:.4f}")


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Je suppose que votre DataFrame est déjà nettoyé et contient les colonnes nécessaires
# et que vous avez déjà défini `change_df_filtered_with_date` avec l'index correct.

# Définir l'index de `change_df_filtered_with_date` sur la colonne de date si ce n'est pas déjà fait
change_df_filtered_with_date.set_index(date_column, inplace=True)

# Assurez-vous que l'index est de type datetime
change_df_filtered_with_date.index = pd.to_datetime(change_df_filtered_with_date.index)

# Le reste de votre code pour l'ARIMA et les prévisions reste le même
optimal_params = {
    'Dollar des USA': (0, 2, 1),
    'EURO': (1, 1, 0),
    'Yen Japonais': (0, 2, 0)
}

# Forecast steps
forecast_steps = 5
# Placeholder pour les valeurs prévues
forecasts = {}

# Prévision pour chaque devise
for currency, params in optimal_params.items():
    model = ARIMA(change_df_filtered_with_date[currency], order=params)
    model_fit = model.fit()
    forecasts[currency] = model_fit.forecast(steps=forecast_steps)

# Création d'un DataFrame à partir des prévisions
forecast_dates = pd.date_range(start=change_df_filtered_with_date.index[-1] + pd.DateOffset(days=1), 
                               periods=forecast_steps, freq='Y')
forecasts_df = pd.DataFrame(forecasts, index=forecast_dates)

# Tracé
fig, axs = plt.subplots(len(optimal_params), 1, figsize=(10, 15))
for i, (currency, forecast) in enumerate(forecasts.items()):
    axs[i].plot(change_df_filtered_with_date.index, change_df_filtered_with_date[currency], label='Historical')
    axs[i].plot(forecast_dates, forecast, label='Forecast', linestyle='--')
    axs[i].set_title(f'{currency} Forecast')
    axs[i].set_xlabel('Year')
    axs[i].set_ylabel('Exchange Rate')
    axs[i].legend()

plt.tight_layout()
# Afficher le DataFrame des prévisions
forecasts_df


from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# Génération de données fictives pour les devises
np.random.seed(42)  # Pour la reproductibilité
data_length = 100  # Total de points de données
train_length = 80  # Points de données pour l'entraînement

# Génération de séries temporelles aléatoires pour chaque devise
data_dollar = np.random.randn(data_length).cumsum()
data_euro = np.random.randn(data_length).cumsum()
data_yen = np.random.randn(data_length).cumsum()

# Division en ensembles d'entraînement et de test
train_dollar, test_dollar = data_dollar[:train_length], data_dollar[train_length:]
train_euro, test_euro = data_euro[:train_length], data_euro[train_length:]
train_yen, test_yen = data_yen[:train_length], data_yen[train_length:]

# Dictionnaire pour stocker les erreurs MSE pour chaque devise
mse_errors = {}

# Fonction pour ajuster un modèle ARIMA et calculer le MSE
def evaluate_arima(data_train, data_test, order=(1,1,1)):
    model = ARIMA(data_train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(data_test))
    mse = mean_squared_error(data_test, predictions)
    return mse

# Évaluation pour chaque devise
mse_errors['Dollar des USA'] = evaluate_arima(train_dollar, test_dollar)
mse_errors['EURO'] = evaluate_arima(train_euro, test_euro)
mse_errors['Yen Japonais'] = evaluate_arima(train_yen, test_yen)
# Dictionnaire pour stocker les scores R^2 pour chaque devise


mse_errors

from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA

# Fonction pour ajuster un modèle ARIMA et calculer le R^2
def evaluate_arima_r2(data_train, data_test, order=(1,1,1)):
    model = ARIMA(data_train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(data_test))
    r2 = r2_score(data_test, predictions)
    return r2

# Initialisation du dictionnaire pour stocker les scores R^2
r2_scores = {}

# Évaluation R^2 pour chaque devise
r2_scores['Dollar des USA'] = evaluate_arima_r2(train_dollar, test_dollar)
r2_scores['EURO'] = evaluate_arima_r2(train_euro, test_euro)
r2_scores['Yen Japonais'] = evaluate_arima_r2(train_yen, test_yen)

print(r2_scores)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Paramètres pour la simulation
np.random.seed(42)  # Pour la reproductibilité
n_samples = 100
slope = 3
intercept = 50
time = np.arange(n_samples).reshape(-1, 1)

# Simuler des données pour chaque devise
currencies = ['Dollar des USA', 'EURO', 'Yen Japonais']
data = {}
for currency in currencies:
    noise = np.random.normal(0, 5, n_samples)  # Bruit ajouté
    data[currency] = slope * time.flatten() + intercept + noise

# Conversion en DataFrame
df = pd.DataFrame(data, columns=currencies)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

r2_scores = {}
forecasts_df = {}

for currency in currencies:
    # Préparation des données
    X = time  # variable indépendante
    y = df[currency].values.reshape(-1, 1)  # variable dépendante
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    
    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Création et ajustement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédiction sur les données de test
    y_pred = model.predict(X_test)
    r2_scores[currency] = r2_score(y_test, y_pred)
    
    # Stockage des prédictions
    forecasts_df[currency] = model.predict(X_scaled).flatten()

# Affichage des R²
print("R² Scores:", r2_scores)

# Conversion des prédictions en DataFrame pour affichage
forecasts_df = pd.DataFrame(forecasts_df, columns=currencies)

from sklearn.metrics import mean_squared_error

# Stockage des MSE
mse_scores = {}

for currency in currencies:
    # Calcul de la MSE
    mse = mean_squared_error(y_scaled, model.predict(X_scaled))
    mse_scores[currency] = mse

# Affichage des MSE
print("MSE Scores:", mse_scores)

import pandas as pd

# Chemin du fichier Excel
excel_path = r"changee.xlsx"

# Nom de la feuille de calcul à lire
sheet_name = 'Feuil1'

# Lecture du fichier Excel dans un DataFrame
change_df_cleaned = pd.read_excel(excel_path, sheet_name=sheet_name)

# Liste des colonnes à conserver
columns_to_keep = ['Dollar des USA', 'Yen Japonais', 'EURO']

# Vérifier si les noms de colonne existent dans le DataFrame
if all(col in change_df_cleaned.columns for col in columns_to_keep):
    # Filtrer le DataFrame
    change_df_filtered = change_df_cleaned[columns_to_keep]
else:
    # Afficher un message d'erreur
    print("Les noms de colonne ne correspondent pas. Veuillez vérifier.")

# Le nom de la colonne de date est différent et doit être inclus également pour l'analyse temporelle
# Nous allons trouver le nom de la colonne contenant les informations de date
date_column = change_df_cleaned.columns[change_df_cleaned.columns.str.contains('Date')][0]

# Inclure la colonne de date dans notre DataFrame filtré
change_df_filtered_with_date = change_df_cleaned[[date_column] + columns_to_keep]

# Nettoyage des données :
# - Conversion de la colonne de date en datetime
# - Remplacement des virgules par des points dans les valeurs de taux de change pour les convertir en type float
change_df_filtered_with_date[date_column] = pd.to_datetime(change_df_filtered_with_date[date_column], errors='coerce')
change_df_filtered_with_date = change_df_filtered_with_date.replace(',', '.', regex=True).dropna()

# Conversion des valeurs de taux de change de string en float
for col in columns_to_keep:
    try:
        change_df_filtered_with_date[col] = change_df_filtered_with_date[col].astype(float)
    except ValueError as e:
        print(f"Erreur lors de la conversion de la colonne {col}: {e}")

# Affichage des premières lignes du DataFrame nettoyé
change_df_filtered_with_date.head()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# Assurez-vous d'avoir le DataFrame 'change_df_filtered_with_date' prêt comme spécifié dans votre code initial

# Conversion de la colonne de date en une valeur numérique (nombre de jours depuis une date)
change_df_filtered_with_date['date_numeric'] = (change_df_filtered_with_date[date_column] - pd.Timestamp("1970-01-01")).dt.days

results = [] # Pour stocker les résultats

for currency in columns_to_keep:
    X = change_df_filtered_with_date[['date_numeric']]  # Features
    y = change_df_filtered_with_date[currency]  # Cible

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construction et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul des métriques d'évaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Stockage des résultats
    results.append({
        'Currency': currency,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'Predictions': y_pred,
        'Actual': y_test.values,
        'Test Dates': X_test['date_numeric'].apply(lambda x: pd.Timestamp("1970-01-01") + pd.Timedelta(days=x)).dt.year
    })

# Affichage des résultats pour chaque devise
for result in results:
    print(f"\nCurrency: {result['Currency']}")
    df_results = pd.DataFrame({
        'Year': result['Test Dates'],
        'Actual': result['Actual'],
        'Predicted': result['Predictions']
    })
    print(df_results.head())  # Affichage des premières lignes pour l'exemple
    print(f"RMSE: {result['RMSE']}, R2: {result['R2']}, MAE: {result['MAE']}")


# Assumons que 'change_df_filtered_with_date' et 'columns_to_keep' sont déjà préparés
# Assurons-nous également que la colonne des dates est au format datetime
change_df_filtered_with_date[date_column] = pd.to_datetime(change_df_filtered_with_date[date_column])

# Ajout d'une colonne pour l'année
change_df_filtered_with_date['Year'] = change_df_filtered_with_date[date_column].dt.year

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Initialisation d'un DataFrame pour stocker les prédictions futures
future_years = np.arange(2024, 2036)
predictions = pd.DataFrame(future_years, columns=['Year'])

# Boucle sur chaque devise pour entraîner un modèle et faire des prédictions
for currency in columns_to_keep:
    # Sélection des données pour la devise actuelle
    X = change_df_filtered_with_date[['Year']]
    y = change_df_filtered_with_date[currency]

    # Conversion de l'année en format numérique pour l'entraînement
    X = X.values.reshape(-1, 1)  # Convertir en format attendu par sklearn
    
    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)

    # Préparation des données pour les futures prédictions (2023 à 2028)
    future_X = future_years.reshape(-1, 1)

    # Prédiction pour les années futures
    future_preds = model.predict(future_X)
    predictions[currency] = future_preds

# Affichage des prédictions pour chaque devise
print(predictions)
predictions.to_excel("predictions_devises.xlsx", index=False)

# Save the predictions to an Excel file
predictions_output_path = 'future_predictions.xlsx'
predictions.to_excel(predictions_output_path, index=False)

# Tracé des prédictions futures pour chaque devise
for currency in columns_to_keep:
    plt.plot(predictions['Year'], predictions[currency], label=currency)

plt.title('Prédictions des taux de change de 2023 à 2028')
plt.xlabel('Année')
plt.ylabel('Taux de change')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Initialiser un dictionnaire pour stocker les métriques pour chaque devise
metrics = {}

for currency in columns_to_keep:
    # Préparation des données
    X = change_df_filtered_with_date['Year'].values.reshape(-1, 1)
    y = change_df_filtered_with_date[currency].values
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construction et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Stockage des métriques
    metrics[currency] = {'R2': r2, 'MSE': mse, 'MAE': mae}

# Affichage des métriques pour chaque devise
for currency, metric_values in metrics.items():
    print(f"Devise: {currency}")
    for metric, value in metric_values.items():
        print(f"{metric}: {value}")
    print()  # Ligne vide pour la séparation

import matplotlib.pyplot as plt

# Assurez-vous que 'columns_to_keep' et 'change_df_filtered_with_date' sont définis comme avant

for currency in columns_to_keep:
    # Préparation des données
    X = change_df_filtered_with_date['Year'].values.reshape(-1, 1)
    y = change_df_filtered_with_date[currency].values
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Tracé des données réelles vs prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Données réelles', alpha=0.6)
    plt.scatter(X_test, y_pred, color='red', label='Prédictions', alpha=0.6)
    plt.title(f"Données réelles vs Prédictions pour {currency}")
    plt.xlabel('Année')
    plt.ylabel(f'Taux de change pour {currency}')
    plt.legend()
    plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Chemin du fichier Excel
excel_path = 'changee.xlsx'

# Nom de la feuille de calcul à lire
sheet_name = 'Feuil1'

# Lecture du fichier Excel dans un DataFrame
change_df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Filtration des colonnes nécessaires
columns_to_keep = ['Dollar des USA', 'Yen Japonais', 'EURO']
date_column = [col for col in change_df.columns if 'Date' in col][0]
change_df_filtered = change_df[[date_column] + columns_to_keep]

# Nettoyage des données
change_df_filtered[date_column] = pd.to_datetime(change_df_filtered[date_column], errors='coerce')
change_df_filtered = change_df_filtered.dropna().replace(',', '.', regex=True)

for col in columns_to_keep:
    change_df_filtered[col] = change_df_filtered[col].astype(float)

# Affichage des premières lignes pour vérification
print(change_df_filtered.head())

# Sélection de la variable à prédire
target = 'Dollar des USA'

# Séparation des caractéristiques et de la cible
X = change_df_filtered[['Yen Japonais', 'EURO']]
y = change_df_filtered[target]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor

# Création du modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

# Sélection de la variable à prédire
target = 'EURO'

# Séparation des caractéristiques et de la cible
X = change_df_filtered[['Yen Japonais', 'Dollar des USA']]
y = change_df_filtered[target]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

# Sélection de la variable à prédire
target = 'Yen Japonais'

# Séparation des caractéristiques et de la cible
X = change_df_filtered[['EURO', 'Dollar des USA']]
y = change_df_filtered[target]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neighbors import KNeighborsRegressor

# Assurez-vous d'avoir le DataFrame 'change_df_filtered_with_date' prêt comme spécifié dans votre code initial

# Définition des colonnes à conserver pour l'analyse
columns_to_keep = ['Dollar des USA', 'EURO', 'Yen Japonais']

# Initialisation d'un dictionnaire pour stocker les métriques pour chaque devise et méthode
metrics = {}

# Régression linéaire
for currency in columns_to_keep:
    X = change_df_filtered_with_date[['Year']].values
    y = change_df_filtered_with_date[currency].values
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construction et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Stockage des métriques
    metrics[f'{currency} - Linear Regression'] = {'R2': r2, 'MSE': mse, 'MAE': mae}

# ARIMA
for currency in columns_to_keep:
    # Ajustement du modèle ARIMA avec les paramètres optimisés
    model_optimized = ARIMA(change_df_filtered_with_date[currency], order=(1, 1, 0))
    model_optimized_fit = model_optimized.fit()

    # Prévisions sur les données historiques
    predictions = model_optimized_fit.predict(start=change_df_filtered_with_date.index[0],
                                               end=change_df_filtered_with_date.index[-1], dynamic=False)

    # Calcul de R^2
    r2 = r2_score(change_df_filtered_with_date[currency][1:], predictions[1:])  # On exclut les premiers points à cause de la différenciation d'ordre 1

    # Stockage des métriques
    metrics[f'{currency} - ARIMA'] = {'R2': r2}

# KNN
for currency in columns_to_keep:
    # Sélection des données pour la devise actuelle
    X = change_df_filtered_with_date[['Year']].values
    y = change_df_filtered_with_date[currency].values

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création et ajustement du modèle KNN
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = knn.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Stockage des métriques
    metrics[f'{currency} - KNN'] = {'R2': r2, 'MSE': mse, 'MAE': mae}

# Affichage des métriques pour chaque devise et méthode
for method, metric_values in metrics.items():
    print(f"Method: {method}")
    for metric, value in metric_values.items():
        print(f"{metric}: {value}")
    print()  # Ligne vide pour la séparation

import matplotlib.pyplot as plt

# Initialisation des listes pour stocker les métriques
methods = []
r2_scores = []
mse_scores = []
mae_scores = []

# Parcourir les métriques et les stocker dans les listes
for method, metric_values in metrics.items():
    methods.append(method)
    r2_scores.append(metric_values['R2'])
    mse_scores.append(metric_values.get('MSE', 0))  # Si le score MSE n'est pas disponible, on utilise 0
    mae_scores.append(metric_values.get('MAE', 0))  # Si le score MAE n'est pas disponible, on utilise 0

# Création du graphique comparatif
plt.figure(figsize=(10, 6))

# R² Scores
plt.barh(methods, r2_scores, color='skyblue', label='R²')

# MSE Scores
if any(mse_scores):  # Vérifie s'il y a des scores MSE disponibles
    plt.barh(methods, mse_scores, color='lightgreen', label='MSE')

# MAE Scores
if any(mae_scores):  # Vérifie s'il y a des scores MAE disponibles
    plt.barh(methods, mae_scores, color='salmon', label='MAE')

# Ajout de titres et de légendes
plt.xlabel('Scores')
plt.title('Comparaison des performances des méthodes de prédiction')
plt.legend()

# Affichage du graphique
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Récupération des méthodes et des scores
methods = list(metrics.keys())
r2_scores = [metric['R2'] for metric in metrics.values()]
mse_scores = [metric.get('MSE', 0) for metric in metrics.values()]
mae_scores = [metric.get('MAE', 0) for metric in metrics.values()]

# Création des positions des barres
bar_width = 0.25
index = np.arange(len(methods))
opacity = 0.8

# Création de la figure et des sous-graphiques
fig, ax = plt.subplots(figsize=(12, 8))

# Barres pour R²
rects1 = ax.bar(index, r2_scores, bar_width, alpha=opacity, color='b', label='R²')

# Barres pour MSE
if any(mse_scores):
    rects2 = ax.bar(index + bar_width, mse_scores, bar_width, alpha=opacity, color='g', label='MSE')

# Barres pour MAE
if any(mae_scores):
    rects3 = ax.bar(index + 2 * bar_width, mae_scores, bar_width, alpha=opacity, color='r', label='MAE')

# Ajout des étiquettes, titres et légendes
ax.set_xlabel('Méthodes')
ax.set_ylabel('Scores')
ax.set_title('Comparaison des performances des méthodes de prédiction')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(methods)
ax.legend()

# Affichage du graphique
plt.tight_layout()
plt.show()
