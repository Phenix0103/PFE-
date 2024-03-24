import pandas as pd

# Chemin vers le fichier Excel
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xls = pd.ExcelFile(file_path)

# Initialiser un DataFrame pour stocker les ROA de toutes les banques pour chaque année
roa_df = pd.DataFrame()

# Itérer sur les années de 2010 à 2022
for year in range(2010, 2023):
    try:
        bilan_df = pd.read_excel(xls, f'BILAN {year}')
        resultat_df = pd.read_excel(xls, f'E Rslt {year}')

        # Utilisation d'une condition pour gérer les variations dans les noms de colonnes ou les libellés
        col_name = 'En milliers de dinars'  # ou adaptez selon les variations observées
        idx_benefice_net = resultat_df.index[resultat_df[col_name].str.contains("RESULTAT NET DE l'EXERCICE", na=False)].tolist()[0]
        idx_total_actifs = bilan_df.index[bilan_df[col_name].str.contains('Total actif', na=False)].tolist()[0]

        benefice_net = pd.to_numeric(resultat_df.iloc[idx_benefice_net, 1:].dropna(), errors='coerce')
        total_actifs = pd.to_numeric(bilan_df.iloc[idx_total_actifs, 1:].dropna(), errors='coerce')

        roa_percentage = (benefice_net / total_actifs) * 100
        roa_df[year] = roa_percentage
    except Exception as e:
        print(f"Erreur pour l'année {year}: {e}")

# Transposer le DataFrame pour avoir les années en lignes et les banques en colonnes
roa_df = roa_df.T

print(roa_df)
roa_df.style.format("{:.2f}")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Créer un index de prévision avec uniquement les années
forecast_years = range(2023, 2036)  # Crée une liste d'années de 2023 à 2036
forecast_index = pd.Index(forecast_years)

# Définir X_future en utilisant forecast_index pour la prédiction
X_future = np.array(forecast_index).reshape(-1, 1)

# Initialiser un DataFrame pour stocker les prédictions de ROE pour chaque banque de 2023 à 2028
roa_predictions_df = pd.DataFrame(index=forecast_index)

# Appliquer la régression linéaire pour chaque colonne (banque) dans le DataFrame ROE d'origine
for column in roa_df.columns:
    # Extraire les données pour la banque actuelle
    bank_data = roa_df[column].dropna()
    X = np.array(bank_data.index).reshape(-1, 1)  # Années
    y = bank_data.values  # ROE

    # Vérifier s'il existe des valeurs non numériques dans y
    if np.isnan(y).any():
        print(f"Skipping bank '{column}' due to non-numeric values in ROE data.")
        continue

    # Si la banque a suffisamment de données, entraîner le modèle de régression linéaire et prédire
    if len(X) > 1:  # Vérifier s'il y a plus d'une année de données
        lr_model = LinearRegression().fit(X, y)
        predictions = lr_model.predict(X_future)
        predictions = np.round(predictions, 2)  # Arrondir les prédictions à deux chiffres après la virgule

    else:
        # S'il n'y a pas suffisamment de données, remplir avec NaN
        predictions = np.full(len(forecast_years), np.nan)

    # Add predictions to the DataFrame
    roa_predictions_df[column] = predictions

# Display the ROE predictions for each bank from 2023 to 2028
roa_predictions_df
# Transposer le DataFrame pour avoir les années en lignes et les banques en colonnes
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/roapred.xlsx'
roa_predictions_df.to_excel(output_file_path)

# Afficher un aperçu des données de ROA
roa_predictions_df

# Fusionner les DataFrames roe_df et roe_predictions_df
merged_roa_df = pd.concat([roa_df, roa_predictions_df])

# Afficher un aperçu du DataFrame fusionné
print(merged_roa_df.head())  # Afficher seulement les premières lignes du DataFrame fusionné pour un aperçu

# Chemin vers le fichier Excel de sortie pour le DataFrame fusionné
output_merged_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/merged_roa.xlsx'

# Exporter le DataFrame fusionné au format Excel
merged_roa_df.to_excel(output_merged_file_path)

# Afficher un message de confirmation
print(f"Le DataFrame fusionné a été exporté avec succès au chemin : {output_merged_file_path}")

import matplotlib.pyplot as plt

def evaluate_roa(dataframe):
    # Calcul de la moyenne annuelle du ROE pour chaque banque
    roa_mean = dataframe.mean(axis=0)
    
    # Tracer un graphique pour visualiser l'évolution du ROE moyen au fil des années
    plt.figure(figsize=(10, 6))
    plt.plot(roa_mean.index, roa_mean.values, marker='o', linestyle='-')
    plt.title('Évolution du ROE moyen')
    plt.xlabel('Année')
    plt.ylabel('ROE moyen (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Afficher les statistiques descriptives du ROE moyen
    print("\nStatistiques descriptives du ROE moyen :")
    print(roa_mean.describe())

evaluate_roa(merged_roa_df)

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Initialize a dictionary to store models for each bank
models = {}

# Train a model for each bank
for bank in roa_df.columns:
    X = np.array(range(2010, 2036)).reshape(-1, 1)  # Years
    y = merged_roa_df[bank].values  # ROE values for the bank
    
    # Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the model
    models[bank] = model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemin vers le fichier Excel à évaluer
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/merged_roa.xlsx'

# Charger les données à partir du fichier Excel
roa_df_eval = pd.read_excel(output_file_path, index_col=0)

# Initialiser un dictionnaire pour stocker les évaluations pour chaque banque
evaluations = {}

# Évaluer les données
for bank, model in models.items():
    X_real = np.array(range(2010, 2023)).reshape(-1, 1)  # Années avec données réelles
    X_future = np.array(range(2023, 2036)).reshape(-1, 1)  # Années futures sans données réelles
    y_real = roa_df_eval.loc[2010:2022, bank].values  # Valeurs de ROE pour les années avec données réelles
    
    # Prédiction des valeurs de ROE pour les années avec données réelles
    y_pred_real = model.predict(X_real)
    
    # Calcul des métriques d'évaluation uniquement pour les années avec données réelles
    r2 = r2_score(y_real, y_pred_real)
    mse = mean_squared_error(y_real, y_pred_real)
    mae = mean_absolute_error(y_real, y_pred_real)
    rmse = np.sqrt(mse)
    
    # Stocker les évaluations
    evaluations[bank] = {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}

    # Prédiction des valeurs de ROE pour les années futures
    y_pred_future = model.predict(X_future)

    # Tracer les valeurs réelles et prédictions (incluant les années futures)
    plt.figure(figsize=(10, 6))
    plt.plot(range(2010, 2023), y_real, label='Valeurs Réelles jusqu\'en 2022')
    plt.plot(range(2023, 2036), y_pred_future, '--', label='Prédictions 2023-2035')
    plt.title(f'Prédictions vs Valeurs Réelles pour {bank}')
    plt.xlabel('Année')
    plt.ylabel('ROE')
    plt.legend()
    plt.show()

# Convertir les évaluations en DataFrame pour affichage
evaluations_df = pd.DataFrame(evaluations).T
print("\nÉvaluation du ROE des données actuelles pour chaque banque :")
print(evaluations_df)


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemin vers le fichier Excel à évaluer
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/merged_roa.xlsx'

# Charger les données à partir du fichier Excel
roa_df_eval = pd.read_excel(output_file_path, index_col=0)

# Initialiser un dictionnaire pour stocker les évaluations pour chaque banque
evaluations = {}

# Évaluer les données
for bank, model in models.items():
    X = np.array(range(2010, 2036)).reshape(-1, 1)  # Années
    y = roa_df_eval[bank].values  # Valeurs de ROE pour la banque
    
    # Prédiction des valeurs de ROE
    y_pred = model.predict(X)
    
    # Calcul des métriques d'évaluation
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Stocker les évaluations
    evaluations[bank] = {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# Convertir les évaluations en DataFrame pour affichage
evaluations_df = pd.DataFrame(evaluations)

# Imprimer une explication sur la méthode de calcul de R2
print("Le coefficient de détermination R2 est calculé comme le carré du coefficient de corrélation Pearson entre les valeurs réelles et prédites. Cela mesure la proportion de la variance dans les valeurs réelles qui peut être expliquée par les valeurs prédites.")

print("\nÉvaluation du ROE des données actuelles pour chaque banque :")
print(evaluations_df)

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Chemin vers le fichier Excel à évaluer
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/merged_roa.xlsx'

# Charger les données à partir du fichier Excel
roa_df_eval = pd.read_excel(output_file_path, index_col=0)

# Initialiser un dictionnaire pour stocker les évaluations pour chaque banque
evaluations = {}

# Évaluer les données pour chaque banque
for bank, model in models.items():
    X = np.array(range(2010, 2036)).reshape(-1, 1)  # Années
    y = roa_df_eval[bank].values  # Valeurs de ROE pour la banque
    
    # Prédiction des valeurs de ROE
    y_pred = model.predict(X)
    
    # Calcul des métriques d'évaluation
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Stocker les évaluations pour chaque banque
    evaluations[bank] = {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# Convertir les évaluations individuelles en DataFrame pour affichage
evaluations_df = pd.DataFrame(evaluations).T

# Calculer les métriques d'évaluation moyennes pour le DataFrame complet
mean_r2 = evaluations_df['R2'].mean()
mean_mse = evaluations_df['MSE'].mean()
mean_mae = evaluations_df['MAE'].mean()
mean_rmse = evaluations_df['RMSE'].mean()

# Ajouter l'évaluation globale
evaluations['Global'] = {'R2': mean_r2, 'MSE': mean_mse, 'MAE': mean_mae, 'RMSE': mean_rmse}

# Afficher les évaluations
print("\nÉvaluation du ROE des données actuelles pour chaque banque :")
print(evaluations_df)

print("\nÉvaluation globale moyenne pour toutes les banques :")
print(f"R2 Moyen: {mean_r2}, MSE Moyen: {mean_mse}, MAE Moyen: {mean_mae}, RMSE Moyen: {mean_rmse}")

# Calcul de la corrélation entre les variables
correlation_matrix = roa_df.corr()
print("Matrice de corrélation :")
print(correlation_matrix)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Suppose roe_df represents your DataFrame of ROE values for each bank over the years
# This is a simulated structure based on your description. Replace it with your actual DataFrame.
banks = ['ATB', 'BNA', 'ATTIBK', 'ABC', 'BT', 'AMENBK', 'BIAT', 'STB', 'UBCI', 'UIB','BH','BTK','QNB','BTE','BZ','BTS','Secteur']
years = list(range(2010, 2023))
data = np.random.rand(len(years), len(banks)) * 20 + 5  # Simulated ROE values
roa_df = pd.DataFrame(data, columns=banks, index=years)


# Calculate and display correlation matrix
correlation_matrix = roa_df.corr()
print("Correlation matrix between banks' ROEs:")
print(correlation_matrix)

# Plotting the ROE trends

# Function to compute residuals and perform regression for each bank
def compute_and_plot_residuals(dataframe):
    residuals_dict = {}
    for bank in dataframe.columns:
        X = dataframe.index.values.reshape(-1, 1)  # Years
        y = dataframe[bank].values  # ROE values
        
        # Linear regression
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        # Compute residuals
        residuals = y - y_pred
        residuals_dict[bank] = residuals
        
        # Plotting residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(X, residuals, label=f'Residuals for {bank}')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f'Residuals for {bank}')
        plt.xlabel('Year')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()
        
    return residuals_dict

# Compute and plot residuals for each bank
residuals_dict = compute_and_plot_residuals(roa_df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemin vers le fichier Excel à évaluer
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/merged_roa.xlsx'

# Charger les données à partir du fichier Excel
roa_df_eval = pd.read_excel(output_file_path, index_col=0)

# Initialiser un dictionnaire pour stocker les évaluations pour chaque banque
evaluations = {}

# Évaluer les données pour chaque banque
for bank, model in models.items():
    X = np.array(range(2010, 2036)).reshape(-1, 1)  # Années
    y = roa_df_eval[bank].values  # Valeurs de ROE pour la banque
    
    # Prédiction des valeurs de ROE
    y_pred = model.predict(X)
    
    # Calcul des métriques d'évaluation
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Stocker les évaluations pour chaque banque
    evaluations[bank] = {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# Convertir les évaluations individuelles en DataFrame pour affichage
evaluations_df = pd.DataFrame(evaluations).T

# Calculer les métriques d'évaluation moyennes pour le DataFrame complet
mean_r2 = evaluations_df['R2'].mean()
mean_mse = evaluations_df['MSE'].mean()
mean_mae = evaluations_df['MAE'].mean()
mean_rmse = evaluations_df['RMSE'].mean()

# Ajouter l'évaluation globale
evaluations['Global'] = {'R2': mean_r2, 'MSE': mean_mse, 'MAE': mean_mae, 'RMSE': mean_rmse}

# Afficher les évaluations
print("\nÉvaluation du ROE des données actuelles pour chaque banque :")
print(evaluations_df)

print("\nÉvaluation globale moyenne pour toutes les banques :")
print(f"R2 Moyen: {mean_r2}, MSE Moyen: {mean_mse}, MAE Moyen: {mean_mae}, RMSE Moyen: {mean_rmse}")

# Tracer les métriques d'évaluation
plt.figure(figsize=(10, 6))
evaluations_df.plot(kind='bar', figsize=(10, 6))
plt.title('Évaluation du ROE pour chaque banque')
plt.xlabel('Banque')
plt.ylabel('Métrique')
plt.xticks(rotation=45)
plt.legend(title='Métrique')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemin vers le fichier Excel à évaluer
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/merged_roa.xlsx'

# Charger les données à partir du fichier Excel
roa_df_eval = pd.read_excel(output_file_path, index_col=0)

# Initialiser un dictionnaire pour stocker les évaluations pour chaque banque
evaluations = {}

# Évaluer les données pour chaque banque
for bank, model in models.items():
    X = np.array(range(2010, 2036)).reshape(-1, 1)  # Années
    y = roa_df_eval[bank].values  # Valeurs de ROE pour la banque
    
    # Prédiction des valeurs de ROE
    y_pred = model.predict(X)
    
    # Calcul des métriques d'évaluation
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Stocker les évaluations pour chaque banque
    evaluations[bank] = {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# Convertir les évaluations individuelles en DataFrame pour affichage
evaluations_df = pd.DataFrame(evaluations).T

# Calculer les métriques d'évaluation moyennes pour le DataFrame complet
mean_r2 = evaluations_df['R2'].mean()
mean_mse = evaluations_df['MSE'].mean()
mean_mae = evaluations_df['MAE'].mean()
mean_rmse = evaluations_df['RMSE'].mean()

# Ajouter l'évaluation globale
evaluations['Global'] = {'R2': mean_r2, 'MSE': mean_mse, 'MAE': mean_mae, 'RMSE': mean_rmse}

# Afficher les évaluations
print("\nÉvaluation du ROE des données actuelles pour chaque banque :")
print(evaluations_df)

print("\nÉvaluation globale moyenne pour toutes les banques :")
print(f"R2 Moyen: {mean_r2}, MSE Moyen: {mean_mse}, MAE Moyen: {mean_mae}, RMSE Moyen: {mean_rmse}")

# Tracer les métriques d'évaluation
plt.figure(figsize=(10, 6))
evaluations_df.plot(kind='bar', figsize=(10, 6))
plt.title('Évaluation du ROE pour chaque banque')
plt.xlabel('Banque')
plt.ylabel('Métrique')
plt.xticks(rotation=45)
plt.legend(title='Métrique')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
