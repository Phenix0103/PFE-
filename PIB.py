import pandas as pd

# Charger les données
data_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIBDone/PIB/PIBNew.xlsx'
df = pd.read_excel(data_path)

# Afficher les premières lignes pour comprendre la structure des données
print(df.head())

import pandas as pd

# Charger les données
data_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIB/PIBNew.xlsx'
df = pd.read_excel(data_path)

# Afficher les premières lignes des données pour comprendre la structure
df.head()

# Transformation des données pour la modélisation de séries temporelles
df_tunisie_pib = df.melt(id_vars=["Country Name"], 
                         var_name="Year", 
                         value_name="PIB")

# Conversion du type de données pour 'Year' en entier pour faciliter l'analyse
df_tunisie_pib['Year'] = df_tunisie_pib['Year'].astype(int)

# Affichage des premières lignes après transformation
print(df_tunisie_pib.head())

df_tunisie_pib['Year'] = df_tunisie_pib['Year'].astype(int)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Visualisation de la corrélation entre les variables numériques
plt.figure(figsize=(10, 6))
sns.heatmap(df_tunisie_pib.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Corrélation entre les variables numériques')
plt.show()

# Visualisation des données temporelles
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='PIB', data=df_tunisie_pib)
plt.title('Évolution du PIB au fil des années')
plt.xlabel('Année')
plt.ylabel('PIB')
plt.show()

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
# Assuming df_tunisie_pib['PIB'] contains the strings with commas as decimal separators
df_tunisie_pib['PIB'] = df_tunisie_pib['PIB'].str.replace(',', '.').astype(float)

# Tester la stationnarité
result_adfuller = adfuller(df_tunisie_pib['PIB'])

# Calculer l'ACF et PACF pour déterminer les paramètres p et q
lag_acf = acf(df_tunisie_pib['PIB'], nlags=5)
lag_pacf = pacf(df_tunisie_pib['PIB'], nlags=5, method='ols')

# Tracer l'ACF et PACF avec le nombre ajusté de décalages
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].stem(lag_acf)
axes[0].axhline(y=0, linestyle='--', color='gray')
axes[0].axhline(y=-1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[0].axhline(y=1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[0].set_title('Fonction d\'Autocorrélation (ACF)')

axes[1].stem(lag_pacf)
axes[1].axhline(y=0, linestyle='--', color='gray')
axes[1].axhline(y=-1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[1].axhline(y=1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[1].set_title('Fonction d\'Autocorrélation Partielle (PACF)')

plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller, acf, pacf

# Stationarity Test - Dickey-Fuller Test
result = adfuller(df_tunisie_pib['PIB'])
dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
from pmdarima.arima import auto_arima

# Fit a model using auto_arima to find the best ARIMA parameters
auto_model = auto_arima(df_tunisie_pib['PIB'], start_p=0, start_q=0,
                        test='adf',       # Use adftest to find optimal 'd'
                        max_p=5, max_q=5, # Maximum p and q
                        m=1,              # Frequency of the series
                        d=None,           # Let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

# Summary of the fitted model
auto_model.summary()

# Convertir la série PIB au format numérique et vérifier les valeurs manquantes
df_tunisie_pib['PIB'] = pd.to_numeric(df_tunisie_pib['PIB'], errors='coerce')

# Vérifier s'il y a des valeurs NaN après la conversion
nan_check = df_tunisie_pib['PIB'].isnull().sum()

# Afficher le nombre de valeurs NaN trouvées, si elles existent
nan_check

from statsmodels.tsa.arima.model import ARIMA

# Ajustement du modèle
model = ARIMA(df_tunisie_pib['PIB'], order=(0,2,0))
model_fit = model.fit()

# Prédictions
preds = model_fit.forecast(steps=5)  # Prédire les 5 prochaines années

# Affichage des prédictions
print(preds)

import numpy as np
from sklearn.metrics import r2_score

# Assuming 'model_fit' is your fitted ARIMA model and 'df_tunisie_pib' is your DataFrame
# Ensure your model has been fitted with something like:
p, d, q = 0, 2, 0
model = ARIMA(df_tunisie_pib['PIB'], order=(p, d, q))
model_fit = model.fit()

# Generating predictions
# Assuming you want to compare these predictions with the actual series
# Let's say we're predicting the same number of steps as our data length for simplicity
predictions = model_fit.predict(start=0, end=len(df_tunisie_pib['PIB'])-1)

# Calculating R^2
r2 = r2_score(df_tunisie_pib['PIB'], predictions)

print(f'R²: {r2}')

Regression Linéaire
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualisation de l'autocorrélation
plt.figure(figsize=(12, 6))
plot_acf(df_tunisie_pib['PIB'], lags=min(20, len(df_tunisie_pib)-1), title='Autocorrélation (ACF) du PIB')
plt.xlabel('Lag')
plt.ylabel('Autocorrélation')
plt.show()

# Visualisation de l'autocorrélation partielle
plt.figure(figsize=(12, 6))
plot_pacf(df_tunisie_pib['PIB'], lags=min(20, len(df_tunisie_pib)//2), title='Autocorrélation partielle (PACF) du PIB')
plt.xlabel('Lag')
plt.ylabel('Autocorrélation partielle')
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np
# Modélisation
X = df_tunisie_pib['Year'].values.reshape(-1, 1)
y = df_tunisie_pib['PIB'].values

model = LinearRegression()
model.fit(X, y)

# Prédiction pour les prochaines années
years_predict = np.array([[2022],[2023], [2024], [2025], [2026], [2027]])
pib_predictions = model.predict(years_predict)


# Affichage des prédictions
predictions = pd.DataFrame({'Year': years_predict.flatten(), 'Predicted PIB': pib_predictions})
print(predictions)
# Calcul du R² en utilisant la méthode score() du modèle
from sklearn.metrics import r2_score

# Calcul du R² en utilisant la fonction r2_score
r2_linear = r2_score(y, model.predict(X))
print("R² pour le modèle de régression linéaire :", r2_linear)

import pandas as pd

# Adjust pandas display options
pd.set_option('display.float_format', '{:.2f}'.format)

# Convert 'Predicted PIB' back to float before formatting
predictions['Predicted PIB'] = predictions['Predicted PIB'].astype(float)

# Now apply the formatting
predictions['Predicted PIB'] = predictions['Predicted PIB'].apply(lambda x: f"{x:.2f}")

# Assuming predictions is your DataFrame and you've just performed some operations on it

# Step 1: Convert 'Predicted PIB' to float
predictions['Predicted PIB'] = predictions['Predicted PIB'].astype(float)

# Step 2: Format 'Predicted PIB' as a string with two decimal places
predictions['Predicted PIB'] = predictions['Predicted PIB'].apply(lambda x: f"{x:.2f}")

print(predictions)

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


predictions['Predicted PIB'] = predictions['Predicted PIB'].astype(float)

# Format 'Predicted PIB' for display
predictions['Predicted PIB'] = predictions['Predicted PIB'].apply(lambda x: f"{x:.2f}")

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(df_tunisie_pib['Year'], df_tunisie_pib['PIB'], label='Historical PIB', marker='o', color='blue')
plt.plot(years_predict.flatten(), pib_predictions, label='Predicted PIB', linestyle='--', marker='x', color='red')
plt.title('PIB Prediction for Tunisia')
plt.xlabel('Year')
plt.ylabel('PIB')
plt.legend()
plt.grid(True)
plt.show()

predictions, r2_linear

file_path = 'C:\\Users\\ASUS TUF I5\\Desktop\\PFE\\ETUDE DU SECTEUR\\KPI\\PIB\\predictions.xlsx'
predictions.to_excel(file_path, index=False)

print("Predictions saved to Excel file successfully.")
Forest Random
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)

# Préparation des données d'entraînement
X_train = df_tunisie_pib['Year'].values.reshape(-1, 1)
y_train = df_tunisie_pib['PIB'].values

# Concaténation des années d'entraînement et de prédiction
years_all = np.concatenate([X_train.flatten(), years_predict.flatten()])

# Ré-entraînement du modèle avec les données mises à jour
forest_model.fit(X_train, y_train)

# Prédiction pour toutes les années
pib_predictions_forest = forest_model.predict(years_all.reshape(-1, 1))

# Séparation des prédictions pour les années futures
pib_predictions_future = pib_predictions_forest[len(X_train):]

# Affichage des prédictions
predictions_forest = pd.DataFrame({'Year': years_predict.flatten(), 'Predicted PIB': pib_predictions_future})
print(predictions_forest)

# Calcul de R²
r2_forest = forest_model.score(X_train, y_train)
print("R² pour le modèle RandomForestRegressor :", r2_forest)


# Visualisation des prédictions
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='PIB', data=df_tunisie_pib, label='Données réelles')
sns.lineplot(x='Year', y='Predicted PIB', data=predictions_forest, label='Prédictions')
plt.title('Prédictions du PIB avec RandomForestRegressor')
plt.xlabel('Année')
plt.ylabel('PIB')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# Assumptions: df_tunisie_pib is your DataFrame containing 'Year' and 'PIB' columns

# Linear Regression
X = df_tunisie_pib['Year'].values.reshape(-1, 1)
y = df_tunisie_pib['PIB'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_lr = linear_model.predict(X)
rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
r2_lr = r2_score(y, y_pred_lr)

# RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)
forest_model.fit(X, y)
y_pred_rf = forest_model.predict(X)
rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
r2_rf = r2_score(y, y_pred_rf)

# ARIMA
p, d, q = 0, 2, 0  # Assuming these parameters for ARIMA
model = ARIMA(df_tunisie_pib['PIB'], order=(p, d, q))
model_fit = model.fit()
predictions_arima = model_fit.predict(start=0, end=len(df_tunisie_pib['PIB'])-1)
rmse_arima = np.sqrt(mean_squared_error(df_tunisie_pib['PIB'], predictions_arima))
r2_arima = r2_score(df_tunisie_pib['PIB'], predictions_arima)

# Data preparation
data = {
    'Method': ['Linear Regression', 'Random Forest', 'ARIMA'],
    'RMSE': [rmse_lr, rmse_rf, rmse_arima],
    'R2': [r2_lr, r2_rf, r2_arima]
}
df = pd.DataFrame(data)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for RMSE
ax1.set_xlabel('Method')
ax1.set_ylabel('RMSE', color='tab:red')
ax1.bar(df['Method'], df['RMSE'], color='tab:red', width=0.4, label='RMSE')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a twin Axes sharing the x-axis for R²
ax2 = ax1.twinx()
ax2.set_ylabel('R²', color='tab:blue')
ax2.plot(df['Method'], df['R2'], color='tab:blue', marker='o', label='R²')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Title and legend
fig.tight_layout()
plt.title('Comparison of Evaluation Methods Based on Real Values')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Placeholder for loading the DataFrame
# Replace 'path_to_your_file.xlsx' with the actual path to your Excel file
path_to_your_file = 'C:/Users\/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIB/PIB/fusion.xlsx'
predictions = pd.read_excel(path_to_your_file, usecols=['Country Name', 'Year', 'Value'])

# Convert 'Value' to numeric, coercing errors
predictions['Value'] = pd.to_numeric(predictions['Value'], errors='coerce')

# Sort by 'Year'
predictions.sort_values('Year', inplace=True)

# Calculate percentage change in 'Value'
predictions['Percentage Change'] = predictions['Value'].pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(predictions['Year'], predictions['Percentage Change'], marker='o', linestyle='-', color='blue')
plt.title('Variation annuelle du PIB tunisien en pourcentage')
plt.xlabel('Année')
plt.ylabel('Pourcentage de variation du PIB')
plt.grid(True)
plt.show()
predictions

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

# Placeholder for loading the DataFrame
# Replace 'path_to_your_file.xlsx' with the actual path to your Excel file
path_to_your_file = 'C:/Users\/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIB/PIB/fusion.xlsx'
# Assuming `predictions` is your DataFrame
# Ensure 'Year' and 'Value' are sorted
predictions.sort_values(['Country Name', 'Year'], inplace=True)

# Convert 'Value' to numeric, coercing errors to NaN
predictions['Value'] = pd.to_numeric(predictions['Value'], errors='coerce')

# Calculate percentage change and handle NaN for the first year
predictions['Percentage Change'] = predictions.groupby('Country Name')['Value'].pct_change().fillna(0) * 100

# If you specifically want to set the first row or a specific year's percentage change to a certain value (e.g., 0 for 2010), you can do so as follows:
predictions.loc[predictions['Year'] == 2010, 'Percentage Change'] = 0

print(predictions)

# Enregistrer le DataFrame dans un fichier Excel spécifique
chemin_enregistrement = r'C:\Users\ASUS TUF I5\Desktop\PFE\ETUDE DU SECTEUR\KPI\PIB\PIB\PIBPourcentafe.xlsx'
predictions.to_excel(chemin_enregistrement, index=False)

print(f'Le DataFrame a été enregistré avec succès à : {chemin_enregistrement}')
import pandas as pd

# Chemin vers le fichier Excel
fichier_excel = "C:\\Users\\ASUS TUF I5\\Desktop\\PFE\\ETUDE DU SECTEUR\\KPI\\PIB\\PIB\\fusion.xlsx"

# Charger le fichier Excel dans un DataFrame
df = pd.read_excel(fichier_excel)

# Convertir la colonne "Value" en nombres décimaux
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Sauvegarder le DataFrame modifié dans un nouveau fichier Excel
df.to_excel("C:\\Users\\ASUS TUF I5\\Desktop\\PFE\\ETUDE DU SECTEUR\\KPI\\PIB\\PIB\\fusion_modifie.xlsx", index=False)
import pandas as pd

# Charger les données
data_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIBDone/PIB/PIBNew.xlsx'
df = pd.read_excel(data_path)

# Afficher les premières lignes pour comprendre la structure des données
print(df.head())

import pandas as pd

# Charger les données
data_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIB/PIBNew.xlsx'
df = pd.read_excel(data_path)

# Afficher les premières lignes des données pour comprendre la structure
df.head()

# Transformation des données pour la modélisation de séries temporelles
df_tunisie_pib = df.melt(id_vars=["Country Name"], 
                         var_name="Year", 
                         value_name="PIB")

# Conversion du type de données pour 'Year' en entier pour faciliter l'analyse
df_tunisie_pib['Year'] = df_tunisie_pib['Year'].astype(int)

# Affichage des premières lignes après transformation
print(df_tunisie_pib.head())

df_tunisie_pib['Year'] = df_tunisie_pib['Year'].astype(int)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Visualisation de la corrélation entre les variables numériques
plt.figure(figsize=(10, 6))
sns.heatmap(df_tunisie_pib.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Corrélation entre les variables numériques')
plt.show()

# Visualisation des données temporelles
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='PIB', data=df_tunisie_pib)
plt.title('Évolution du PIB au fil des années')
plt.xlabel('Année')
plt.ylabel('PIB')
plt.show()

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
# Assuming df_tunisie_pib['PIB'] contains the strings with commas as decimal separators
df_tunisie_pib['PIB'] = df_tunisie_pib['PIB'].str.replace(',', '.').astype(float)

# Tester la stationnarité
result_adfuller = adfuller(df_tunisie_pib['PIB'])

# Calculer l'ACF et PACF pour déterminer les paramètres p et q
lag_acf = acf(df_tunisie_pib['PIB'], nlags=5)
lag_pacf = pacf(df_tunisie_pib['PIB'], nlags=5, method='ols')

# Tracer l'ACF et PACF avec le nombre ajusté de décalages
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].stem(lag_acf)
axes[0].axhline(y=0, linestyle='--', color='gray')
axes[0].axhline(y=-1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[0].axhline(y=1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[0].set_title('Fonction d\'Autocorrélation (ACF)')

axes[1].stem(lag_pacf)
axes[1].axhline(y=0, linestyle='--', color='gray')
axes[1].axhline(y=-1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[1].axhline(y=1.96/np.sqrt(len(df_tunisie_pib['PIB'])), linestyle='--', color='gray')
axes[1].set_title('Fonction d\'Autocorrélation Partielle (PACF)')

plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller, acf, pacf

# Stationarity Test - Dickey-Fuller Test
result = adfuller(df_tunisie_pib['PIB'])
dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
from pmdarima.arima import auto_arima

# Fit a model using auto_arima to find the best ARIMA parameters
auto_model = auto_arima(df_tunisie_pib['PIB'], start_p=0, start_q=0,
                        test='adf',       # Use adftest to find optimal 'd'
                        max_p=5, max_q=5, # Maximum p and q
                        m=1,              # Frequency of the series
                        d=None,           # Let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

# Summary of the fitted model
auto_model.summary()

# Convertir la série PIB au format numérique et vérifier les valeurs manquantes
df_tunisie_pib['PIB'] = pd.to_numeric(df_tunisie_pib['PIB'], errors='coerce')

# Vérifier s'il y a des valeurs NaN après la conversion
nan_check = df_tunisie_pib['PIB'].isnull().sum()

# Afficher le nombre de valeurs NaN trouvées, si elles existent
nan_check

from statsmodels.tsa.arima.model import ARIMA

# Ajustement du modèle
model = ARIMA(df_tunisie_pib['PIB'], order=(0,2,0))
model_fit = model.fit()

# Prédictions
preds = model_fit.forecast(steps=5)  # Prédire les 5 prochaines années

# Affichage des prédictions
print(preds)

import numpy as np
from sklearn.metrics import r2_score

# Assuming 'model_fit' is your fitted ARIMA model and 'df_tunisie_pib' is your DataFrame
# Ensure your model has been fitted with something like:
p, d, q = 0, 2, 0
model = ARIMA(df_tunisie_pib['PIB'], order=(p, d, q))
model_fit = model.fit()

# Generating predictions
# Assuming you want to compare these predictions with the actual series
# Let's say we're predicting the same number of steps as our data length for simplicity
predictions = model_fit.predict(start=0, end=len(df_tunisie_pib['PIB'])-1)

# Calculating R^2
r2 = r2_score(df_tunisie_pib['PIB'], predictions)

print(f'R²: {r2}')

Regression Linéaire
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualisation de l'autocorrélation
plt.figure(figsize=(12, 6))
plot_acf(df_tunisie_pib['PIB'], lags=min(20, len(df_tunisie_pib)-1), title='Autocorrélation (ACF) du PIB')
plt.xlabel('Lag')
plt.ylabel('Autocorrélation')
plt.show()

# Visualisation de l'autocorrélation partielle
plt.figure(figsize=(12, 6))
plot_pacf(df_tunisie_pib['PIB'], lags=min(20, len(df_tunisie_pib)//2), title='Autocorrélation partielle (PACF) du PIB')
plt.xlabel('Lag')
plt.ylabel('Autocorrélation partielle')
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np
# Modélisation
X = df_tunisie_pib['Year'].values.reshape(-1, 1)
y = df_tunisie_pib['PIB'].values

model = LinearRegression()
model.fit(X, y)

# Prédiction pour les prochaines années
years_predict = np.array([[2022],[2023], [2024], [2025], [2026], [2027]])
pib_predictions = model.predict(years_predict)


# Affichage des prédictions
predictions = pd.DataFrame({'Year': years_predict.flatten(), 'Predicted PIB': pib_predictions})
print(predictions)
# Calcul du R² en utilisant la méthode score() du modèle
from sklearn.metrics import r2_score

# Calcul du R² en utilisant la fonction r2_score
r2_linear = r2_score(y, model.predict(X))
print("R² pour le modèle de régression linéaire :", r2_linear)

import pandas as pd

# Adjust pandas display options
pd.set_option('display.float_format', '{:.2f}'.format)

# Convert 'Predicted PIB' back to float before formatting
predictions['Predicted PIB'] = predictions['Predicted PIB'].astype(float)

# Now apply the formatting
predictions['Predicted PIB'] = predictions['Predicted PIB'].apply(lambda x: f"{x:.2f}")

# Assuming predictions is your DataFrame and you've just performed some operations on it

# Step 1: Convert 'Predicted PIB' to float
predictions['Predicted PIB'] = predictions['Predicted PIB'].astype(float)

# Step 2: Format 'Predicted PIB' as a string with two decimal places
predictions['Predicted PIB'] = predictions['Predicted PIB'].apply(lambda x: f"{x:.2f}")

print(predictions)

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


predictions['Predicted PIB'] = predictions['Predicted PIB'].astype(float)

# Format 'Predicted PIB' for display
predictions['Predicted PIB'] = predictions['Predicted PIB'].apply(lambda x: f"{x:.2f}")

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(df_tunisie_pib['Year'], df_tunisie_pib['PIB'], label='Historical PIB', marker='o', color='blue')
plt.plot(years_predict.flatten(), pib_predictions, label='Predicted PIB', linestyle='--', marker='x', color='red')
plt.title('PIB Prediction for Tunisia')
plt.xlabel('Year')
plt.ylabel('PIB')
plt.legend()
plt.grid(True)
plt.show()

predictions, r2_linear

file_path = 'C:\\Users\\ASUS TUF I5\\Desktop\\PFE\\ETUDE DU SECTEUR\\KPI\\PIB\\predictions.xlsx'
predictions.to_excel(file_path, index=False)

print("Predictions saved to Excel file successfully.")
Forest Random
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)

# Préparation des données d'entraînement
X_train = df_tunisie_pib['Year'].values.reshape(-1, 1)
y_train = df_tunisie_pib['PIB'].values

# Concaténation des années d'entraînement et de prédiction
years_all = np.concatenate([X_train.flatten(), years_predict.flatten()])

# Ré-entraînement du modèle avec les données mises à jour
forest_model.fit(X_train, y_train)

# Prédiction pour toutes les années
pib_predictions_forest = forest_model.predict(years_all.reshape(-1, 1))

# Séparation des prédictions pour les années futures
pib_predictions_future = pib_predictions_forest[len(X_train):]

# Affichage des prédictions
predictions_forest = pd.DataFrame({'Year': years_predict.flatten(), 'Predicted PIB': pib_predictions_future})
print(predictions_forest)

# Calcul de R²
r2_forest = forest_model.score(X_train, y_train)
print("R² pour le modèle RandomForestRegressor :", r2_forest)


# Visualisation des prédictions
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='PIB', data=df_tunisie_pib, label='Données réelles')
sns.lineplot(x='Year', y='Predicted PIB', data=predictions_forest, label='Prédictions')
plt.title('Prédictions du PIB avec RandomForestRegressor')
plt.xlabel('Année')
plt.ylabel('PIB')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# Assumptions: df_tunisie_pib is your DataFrame containing 'Year' and 'PIB' columns

# Linear Regression
X = df_tunisie_pib['Year'].values.reshape(-1, 1)
y = df_tunisie_pib['PIB'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_lr = linear_model.predict(X)
rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
r2_lr = r2_score(y, y_pred_lr)

# RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)
forest_model.fit(X, y)
y_pred_rf = forest_model.predict(X)
rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
r2_rf = r2_score(y, y_pred_rf)

# ARIMA
p, d, q = 0, 2, 0  # Assuming these parameters for ARIMA
model = ARIMA(df_tunisie_pib['PIB'], order=(p, d, q))
model_fit = model.fit()
predictions_arima = model_fit.predict(start=0, end=len(df_tunisie_pib['PIB'])-1)
rmse_arima = np.sqrt(mean_squared_error(df_tunisie_pib['PIB'], predictions_arima))
r2_arima = r2_score(df_tunisie_pib['PIB'], predictions_arima)

# Data preparation
data = {
    'Method': ['Linear Regression', 'Random Forest', 'ARIMA'],
    'RMSE': [rmse_lr, rmse_rf, rmse_arima],
    'R2': [r2_lr, r2_rf, r2_arima]
}
df = pd.DataFrame(data)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for RMSE
ax1.set_xlabel('Method')
ax1.set_ylabel('RMSE', color='tab:red')
ax1.bar(df['Method'], df['RMSE'], color='tab:red', width=0.4, label='RMSE')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a twin Axes sharing the x-axis for R²
ax2 = ax1.twinx()
ax2.set_ylabel('R²', color='tab:blue')
ax2.plot(df['Method'], df['R2'], color='tab:blue', marker='o', label='R²')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Title and legend
fig.tight_layout()
plt.title('Comparison of Evaluation Methods Based on Real Values')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Placeholder for loading the DataFrame
# Replace 'path_to_your_file.xlsx' with the actual path to your Excel file
path_to_your_file = 'C:/Users\/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIB/PIB/fusion.xlsx'
predictions = pd.read_excel(path_to_your_file, usecols=['Country Name', 'Year', 'Value'])

# Convert 'Value' to numeric, coercing errors
predictions['Value'] = pd.to_numeric(predictions['Value'], errors='coerce')

# Sort by 'Year'
predictions.sort_values('Year', inplace=True)

# Calculate percentage change in 'Value'
predictions['Percentage Change'] = predictions['Value'].pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(predictions['Year'], predictions['Percentage Change'], marker='o', linestyle='-', color='blue')
plt.title('Variation annuelle du PIB tunisien en pourcentage')
plt.xlabel('Année')
plt.ylabel('Pourcentage de variation du PIB')
plt.grid(True)
plt.show()
predictions

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

# Placeholder for loading the DataFrame
# Replace 'path_to_your_file.xlsx' with the actual path to your Excel file
path_to_your_file = 'C:/Users\/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/PIB/PIB/fusion.xlsx'
# Assuming `predictions` is your DataFrame
# Ensure 'Year' and 'Value' are sorted
predictions.sort_values(['Country Name', 'Year'], inplace=True)

# Convert 'Value' to numeric, coercing errors to NaN
predictions['Value'] = pd.to_numeric(predictions['Value'], errors='coerce')

# Calculate percentage change and handle NaN for the first year
predictions['Percentage Change'] = predictions.groupby('Country Name')['Value'].pct_change().fillna(0) * 100

# If you specifically want to set the first row or a specific year's percentage change to a certain value (e.g., 0 for 2010), you can do so as follows:
predictions.loc[predictions['Year'] == 2010, 'Percentage Change'] = 0

print(predictions)

# Enregistrer le DataFrame dans un fichier Excel spécifique
chemin_enregistrement = r'C:\Users\ASUS TUF I5\Desktop\PFE\ETUDE DU SECTEUR\KPI\PIB\PIB\PIBPourcentafe.xlsx'
predictions.to_excel(chemin_enregistrement, index=False)

print(f'Le DataFrame a été enregistré avec succès à : {chemin_enregistrement}')
import pandas as pd

# Chemin vers le fichier Excel
fichier_excel = "C:\\Users\\ASUS TUF I5\\Desktop\\PFE\\ETUDE DU SECTEUR\\KPI\\PIB\\PIB\\fusion.xlsx"

# Charger le fichier Excel dans un DataFrame
df = pd.read_excel(fichier_excel)

# Convertir la colonne "Value" en nombres décimaux
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Sauvegarder le DataFrame modifié dans un nouveau fichier Excel
df.to_excel("C:\\Users\\ASUS TUF I5\\Desktop\\PFE\\ETUDE DU SECTEUR\\KPI\\PIB\\PIB\\fusion_modifie.xlsx", index=False)
