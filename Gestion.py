
import pandas as pd

# Chemin d'accès au fichier Excel ajusté pour l'environnement actuel
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xl = pd.ExcelFile(file_path)

def calculate_ratios_for_year_adapted(bilan_df):
    try:
        # Sélectionner la dernière colonne pour les calculs, qui correspond aux totaux du secteur
        column_for_calculation = bilan_df.iloc[:, -1]
        
        # Correspondance des termes et somme des valeurs pour les postes spécifiques
        liquidites = bilan_df[bilan_df['En milliers de dinars'].str.contains('Caisse et avoirs auprès de la BCT CCP et TGT', na=False)]['Secteur'].sum()
        creances_sur_clientele = bilan_df[bilan_df['En milliers de dinars'].str.contains('Créances sur clientèle', na=False)]['Secteur'].sum()
        creances_financieres = bilan_df[bilan_df['En milliers de dinars'].str.contains('Créances sur établissement financier ou bancaires', na=False)]['Secteur'].sum()
        depots_et_avoirs_clients = bilan_df[bilan_df['En milliers de dinars'].str.contains('Dépôts et avoirs de la clientèle', na=False)]['Secteur'].sum()
        emprunts_et_ressources_speciales = bilan_df[bilan_df['En milliers de dinars'].str.contains('Emprunts et ressources spéciales', na=False)]['Secteur'].sum()
        autres_passifs = bilan_df[bilan_df['En milliers de dinars'].str.contains('Autres passifs', na=False)]['Secteur'].sum()

        # Calcul du passif à court terme
        passifs_court_terme = depots_et_avoirs_clients + emprunts_et_ressources_speciales + autres_passifs

        # Calcul du ratio de liquidité
        if passifs_court_terme == 0:
            ratio_liquidite = float('nan')  # Ou définir à 0 selon la préférence
        else:
            ratio_liquidite = ((liquidites + creances_sur_clientele + creances_financieres) / passifs_court_terme) * 100

    except Exception as e:
        print(f"Une erreur est survenue lors du traitement des données : {e}")
        return (0, 0, 0, 0, 0)  # Retourne des valeurs par défaut en cas d'erreur

    return (liquidites, creances_sur_clientele, creances_financieres, passifs_court_terme, ratio_liquidite)

# Initialiser un dictionnaire pour contenir les données collectées pour chaque année
yearly_data_adapted = {
    'Year': [],
    'Liquidites': [],
    'Creances_sur_Clientele': [],
    'Creances_Financieres': [],
    'Passifs_Court_Terme': [],
    'Ratio_Liquidite (%)': []
}

# Itérer sur chaque feuille de bilan de 2010 à 2022
for year in range(2010, 2023):
    sheet_name = f'BILAN {year}'
    try:
        bilan_df = xl.parse(sheet_name)
    except ValueError as ve:
        print(f"Erreur lors de la lecture de la feuille {sheet_name}: {ve}")
        continue

    liquidites, creances_sur_clientele, creances_financieres, passifs_ct, ratio_liquidite = calculate_ratios_for_year_adapted(bilan_df)
    
    yearly_data_adapted['Year'].append(year)
    yearly_data_adapted['Liquidites'].append(liquidites)
    yearly_data_adapted['Creances_sur_Clientele'].append(creances_sur_clientele)
    yearly_data_adapted['Creances_Financieres'].append(creances_financieres)
    yearly_data_adapted['Passifs_Court_Terme'].append(passifs_ct)
    yearly_data_adapted['Ratio_Liquidite (%)'].append(ratio_liquidite)
# Spécifier le chemin d'accès et le nom du fichier où vous souhaitez enregistrer le DataFrame
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/Gestion des passifs actifs/Yearly_Ratios_Summary.xlsx'

# Enregistrer le DataFrame sous forme de fichier Excel

# Convertir le dictionnaire adapté en DataFrame pour une visualisation et une analyse faciles
yearly_summary_df_adapted = pd.DataFrame(yearly_data_adapted)
yearly_summary_df_adapted.to_excel(output_file_path, sheet_name='Ratios Summary', index=False)

# Afficher le résumé des données adaptées pour vérifier les résultats, particulièrement pour 2021
yearly_summary_df_adapted

import plotly.express as px

# Créer un graphique interactif pour visualiser l'évolution du Ratio de Liquidité
fig = px.line(yearly_summary_df_adapted, x='Year', y='Ratio_Liquidite (%)', title='Évolution du Ratio de Liquidité de 2010 à 2022', markers=True)

# Ajouter des détails au graphique
fig.update_layout(xaxis_title='Année', yaxis_title='Ratio de Liquidité (%)', legend_title='Légende')

# Afficher le graphique
fig.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# Supposons que 'Year' a été remplacé par 'Bank' juste pour l'exemple
# Assurez-vous que 'Bank' est la colonne correcte à utiliser
# Sélection de la colonne numérique comme variable explicative
X = yearly_summary_df_adapted[['Liquidites', 'Creances_Financieres']].values
y = yearly_summary_df_adapted['Ratio_Liquidite (%)'].values  
# Assurez-vous d'abord que 'bank_summary_df' est correctement défini et contient les données attendues
# Si nécessaire, remplacez 'bank_summary_df' par le nom correct de votre DataFrame contenant les données

# Variable cible

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédictions
predictions = model.predict(X)

# Évaluation du modèle
print("Mean squared error: %.2f" % mean_squared_error(y, predictions))
print('R²: %.2f' % r2_score(y, predictions))
# Assuming X is 2D, we plot each feature against y in a loop for simplicity. Adjust as necessary.

# Plotting the real values against the predictions
plt.figure(figsize=(10, 6))

# Plotting the real values
plt.scatter(X[:, 0], y, color='blue', label='Real Values: Liquidités')
plt.scatter(X[:, 1], y, color='green', label='Real Values: Créances Financières')

# Plotting the predictions
plt.plot(X[:, 0], predictions, color='red', linewidth=2, label='Predictions: Liquidités')
plt.plot(X[:, 1], predictions, color='orange', linewidth=2, label='Predictions: Créances Financières')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Ratio de Liquidité (%)')
plt.title('Régression Linéaire: Réalité vs. Prédictions')
plt.legend()
plt.grid(True)

# Showing the plot
plt.show()

# Détermination du nombre optimal de clusters via la méthode du coude
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(yearly_summary_df_adapted[['Ratio_Liquidite (%)']])
    distortions.append(sum(np.min(cdist(yearly_summary_df_adapted[['Ratio_Liquidite (%)']], kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / yearly_summary_df_adapted.shape[0])

# Affichage de la méthode du coude
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('La méthode du coude montrant le nombre optimal de clusters')
plt.show()

# Supposons que le nombre optimal de clusters déterminé soit 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(yearly_summary_df_adapted[['Ratio_Liquidite (%)']])
labels = kmeans.labels_

# Évaluation avec le score de silhouette
silhouette_avg = silhouette_score(yearly_summary_df_adapted[['Ratio_Liquidite (%)']], labels)
print(f'Score de silhouette pour 3 clusters : {silhouette_avg}')

# Adapter le script pour calculer les ratios par banque à partir de la feuille "BILAN 2022"
# Lire les données de la feuille "BILAN 2022" pour examiner la structure
bilan_2022_df = xl.parse('BILAN 2022')

# Afficher les premières lignes pour comprendre la structure des données
bilan_2022_df.head()

# Initialiser un dictionnaire pour contenir les données collectées pour chaque banque
bank_data = {
    'Bank': [],
    'Liquidites': [],
    'Creances_sur_Clientele': [],
    'Creances_Financieres': [],
    'Passifs_Court_Terme': [],
    'Ratio_Liquidite (%)': []
}

# Itérer sur chaque colonne (banque) de la feuille de bilan, en excluant la première (descriptions) et la dernière (secteur total)
for bank in bilan_2022_df.columns[1:-1]:  # Exclure la première et la dernière colonne
    bilan_df = bilan_2022_df[['En milliers de dinars', bank]]
    
    # Renommer les colonnes pour simplifier l'accès
    bilan_df.columns = ['Postes', 'Valeurs']
    
    # Effectuer les calculs comme précédemment, mais pour chaque banque
    liquidites = bilan_df[bilan_df['Postes'].str.contains('Caisse et avoirs auprès de la BCT CCP et TGT', na=False)]['Valeurs'].sum()
    creances_sur_clientele = bilan_df[bilan_df['Postes'].str.contains('Créances sur clientèle', na=False)]['Valeurs'].sum()
    creances_financieres = bilan_df[bilan_df['Postes'].str.contains('Créances sur établissement financier ou bancaires', na=False)]['Valeurs'].sum()
    # Hypothèse: Utiliser les mêmes postes pour calculer les passifs à court terme
    depots_et_avoirs_clients = bilan_df[bilan_df['Postes'].str.contains('Dépôts et avoirs de la clientèle', na=False)]['Valeurs'].sum()
    emprunts_et_ressources_speciales = bilan_df[bilan_df['Postes'].str.contains('Emprunts et ressources spéciales', na=False)]['Valeurs'].sum()
    autres_passifs = bilan_df[bilan_df['Postes'].str.contains('Autres passifs', na=False)]['Valeurs'].sum()
    
    # Calcul du passif à court terme (sans indication claire, on utilise les mêmes postes hypothétiquement)
    passifs_court_terme = depots_et_avoirs_clients + emprunts_et_ressources_speciales + autres_passifs
    
    # Calcul du ratio de liquidité
    if passifs_court_terme == 0:
        ratio_liquidite = float('nan')  # Ou définir à 0 selon la préférence
    else:
        ratio_liquidite = ((liquidites + creances_sur_clientele + creances_financieres) / passifs_court_terme) * 100

    # Ajouter les résultats au dictionnaire
    bank_data['Bank'].append(bank)
    bank_data['Liquidites'].append(liquidites)
    bank_data['Creances_sur_Clientele'].append(creances_sur_clientele)
    bank_data['Creances_Financieres'].append(creances_financieres)
    bank_data['Passifs_Court_Terme'].append(passifs_court_terme)
    bank_data['Ratio_Liquidite (%)'].append(ratio_liquidite)
 #Spécifier le chemin d'accès et le nom du fichier où vous souhaitez enregistrer le DataFrame

# Enregistrer le DataFrame sous forme de fichier Excel
# Convertir le dictionnaire en DataFrame pour une visualisation et une analyse faciles
# Spécifier le chemin d'accès et le nom du fichier où vous souhaitez enregistrer le DataFrame
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/Gestion des passifs actifs/bANK_Ratios_Summary.xlsx'

# Convertir le dictionnaire en DataFrame pour une visualisation et une analyse faciles
bank_summary_df_adapted = pd.DataFrame(bank_data)

# Enregistrer le DataFrame sous forme de fichier Excel
bank_summary_df_adapted.to_excel(output_file_path, sheet_name='Ratios Summary', index=False)

# Convertir le dictionnaire adapté en DataFrame pour une visualisation et une analyse faciles

# Afficher le résumé des données pour vérifier les résultats
bank_summary_df_adapted.head()  # Afficher les premières lignes pour vérification

import plotly.express as px

# Créer un graphique à barres interactif pour visualiser le Ratio de Liquidité par banque
fig = px.bar(bank_summary_df_adapted, x='Bank', y='Ratio_Liquidite (%)', title='Ratio de Liquidité par Banque pour l\'année 2022', text='Ratio_Liquidite (%)')

# Ajouter des détails au graphique
fig.update_layout(xaxis_title='Banque', yaxis_title='Ratio de Liquidité (%)', legend_title='Légende')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# Afficher le graphique
fig.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# Supposons que 'Year' a été remplacé par 'Bank' juste pour l'exemple
# Assurez-vous que 'Bank' est la colonne correcte à utiliser
# Sélection de la colonne numérique comme variable explicative
X = bank_summary_df_adapted[['Liquidites', 'Creances_Financieres']].values
y = bank_summary_df_adapted['Ratio_Liquidite (%)'].values  

# Assurez-vous d'abord que 'bank_summary_df' est correctement défini et contient les données attendues
# Si nécessaire, remplacez 'bank_summary_df' par le nom correct de votre DataFrame contenant les données

# Variable cible

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédictions
predictions = model.predict(X)

# Évaluation du modèle
print("Mean squared error: %.2f" % mean_squared_error(y, predictions))
print('R²: %.2f' % r2_score(y, predictions))


# Plotting the real values against the predictions
plt.figure(figsize=(10, 6))

# Plotting the real values
plt.scatter(X[:, 0], y, color='blue', label='Real Values: Liquidités')
plt.scatter(X[:, 1], y, color='green', label='Real Values: Créances Financières')

# Plotting the predictions
plt.plot(X[:, 0], predictions, color='red', linewidth=2, label='Predictions: Liquidités')
plt.plot(X[:, 1], predictions, color='orange', linewidth=2, label='Predictions: Créances Financières')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Ratio de Liquidité (%)')
plt.title('Régression Linéaire: Réalité vs. Prédictions')
plt.legend()
plt.grid(True)

# Showing the plot
plt.show()

# Détermination du nombre optimal de clusters via la méthode du coude
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(bank_summary_df_adapted[['Ratio_Liquidite (%)']])
    distortions.append(sum(np.min(cdist(yearly_summary_df_adapted[['Ratio_Liquidite (%)']], kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / yearly_summary_df_adapted.shape[0])

# Affichage de la méthode du coude
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('La méthode du coude montrant le nombre optimal de clusters')
plt.show()

# Supposons que le nombre optimal de clusters déterminé soit 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(bank_summary_df_adapted[['Ratio_Liquidite (%)']])
labels = kmeans.labels_

# Évaluation avec le score de silhouette
silhouette_avg = silhouette_score(bank_summary_df_adapted[['Ratio_Liquidite (%)']], labels)
print(f'Score de silhouette pour 3 clusters : {silhouette_avg}')

import pandas as pd

# Assurez-vous que xl est votre objet ExcelFile chargé avec le fichier Excel

# Initialiser un dictionnaire pour contenir les données collectées pour chaque année et chaque banque
all_bank_data = {
    'Year': [],
    'Bank': [],
    'Liquidites': [],
    'Creances_sur_Clientele': [],
    'Creances_Financieres': [],
    'Passifs_Court_Terme': [],
    'Ratio_Liquidite (%)': []
}

# Boucler sur chaque année de 2010 à 2023
for year in range(2010, 2023):
    sheet_name = f'BILAN {year}'
    try:
        bilan_df = xl.parse(sheet_name)
    except ValueError as ve:
        print(f"Erreur lors de la lecture de la feuille {sheet_name}: {ve}")
        continue

    # Itérer sur chaque colonne (banque)
    for bank in bilan_df.columns[1:-1]:  # Exclure la première et la dernière colonne
        bilan_bank_df = bilan_df[['En milliers de dinars', bank]]
        bilan_bank_df.columns = ['Postes', 'Valeurs']
        
        # Convertir les valeurs en nombres flottants
        bilan_bank_df['Valeurs'] = pd.to_numeric(bilan_bank_df['Valeurs'], errors='coerce').fillna(0)
        
        liquidites = bilan_bank_df[bilan_bank_df['Postes'].str.contains('Caisse et avoirs auprès de la BCT CCP et TGT', na=False)]['Valeurs'].sum()
        creances_sur_clientele = bilan_bank_df[bilan_bank_df['Postes'].str.contains('Créances sur clientèle', na=False)]['Valeurs'].sum()
        creances_financieres = bilan_bank_df[bilan_bank_df['Postes'].str.contains('Créances sur établissement financier ou bancaires', na=False)]['Valeurs'].sum()
        depots_et_avoirs_clients = bilan_bank_df[bilan_bank_df['Postes'].str.contains('Dépôts et avoirs de la clientèle', na=False)]['Valeurs'].sum()
        emprunts_et_ressources_speciales = bilan_bank_df[bilan_bank_df['Postes'].str.contains('Emprunts et ressources spéciales', na=False)]['Valeurs'].sum()
        autres_passifs = bilan_bank_df[bilan_bank_df['Postes'].str.contains('Autres passifs', na=False)]['Valeurs'].sum()

        # Calcul du passif à court terme
        passifs_court_terme = depots_et_avoirs_clients + emprunts_et_ressources_speciales + autres_passifs

        # Calcul du ratio de liquidité
        if passifs_court_terme == 0:
            ratio_liquidite = float('nan')  # Ou définir à 0 selon la préférence
        else:
            ratio_liquidite = ((liquidites + creances_sur_clientele + creances_financieres) / passifs_court_terme) * 100

        # Ajouter les résultats au dictionnaire
        all_bank_data['Year'].append(year)
        all_bank_data['Bank'].append(bank)
        all_bank_data['Liquidites'].append(liquidites)
        all_bank_data['Creances_sur_Clientele'].append(creances_sur_clientele)
        all_bank_data['Creances_Financieres'].append(creances_financieres)
        all_bank_data['Passifs_Court_Terme'].append(passifs_court_terme)
        all_bank_data['Ratio_Liquidite (%)'].append(ratio_liquidite)
 #Spécifier le chemin d'accès et le nom du fichier où vous souhaitez enregistrer le DataFrame
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/Gestion des passifs actifs/YearlyBank_Ratios_Summary.xlsx'

# Enregistrer le DataFrame sous forme de fichier Excel
yearly_summary_df_adapted.to_excel(output_file_path, sheet_name='Ratios Summary', index=False)
# Convertir le dictionnaire en DataFrame pour une visualisation et une analyse faciles
banka_summary_df = pd.DataFrame(all_bank_data)

# Afficher le résumé des données pour vérifier les résultats
print(banka_summary_df.head())  # Afficher les premières lignes pour vérification

import plotly.express as px

# Création du graphique
fig = px.line(banka_summary_df,
              x='Year', y='Ratio_Liquidite (%)',
              color='Bank',
              title='Évolution du Ratio de Liquidité par Banque de 2010 à 2022',
              markers=True)

# Ajout de détails au graphique
fig.update_layout(xaxis_title='Année',
                  yaxis_title='Ratio de Liquidité (%)',
                  legend_title='Banque')

# Affichage du graphique
fig.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.impute import SimpleImputer

# Préparation des données financières avec imputation des valeurs NaN
imputer = SimpleImputer(strategy='mean')  # Création d'un imputeur qui remplace les valeurs NaN par la moyenne de chaque colonne
imputed_features = imputer.fit_transform(banka_summary_df[['Liquidites', 'Creances_sur_Clientele', 'Creances_Financieres', 'Passifs_Court_Terme', 'Ratio_Liquidite (%)']])

# Standardisation des données financières après imputation
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)

# Application de l'analyse de cluster K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Ajustez le nombre de clusters si nécessaire
banka_summary_df['Cluster'] = kmeans.fit_predict(scaled_features)
#Spécifier le chemin d'accès et le nom du fichier où vous souhaitez enregistrer le DataFrame
output_file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/KPI/Gestion des passifs actifs/cluster.xlsx'

# Enregistrer le DataFrame sous forme de fichier Excel
yearly_summary_df_adapted.to_excel(output_file_path, sheet_name='Ratios Summary', index=False)
# Affichage des premières lignes pour vérifier les clusters
print(banka_summary_df.head())

from sklearn.metrics import silhouette_score

# Assurez-vous que 'clusters' contient les étiquettes de cluster pour chaque banque
clusters = banka_summary_df['Cluster']

# Calcul du score de silhouette pour évaluer la qualité des clusters
silhouette_avg = silhouette_score(scaled_features, clusters)
print(f"Score de silhouette moyen : {silhouette_avg:.2f}")

import plotly.express as px

# Création d'un graphique à barres pour visualiser le Ratio de Liquidité par Banque
fig = px.bar(banka_summary_df, x='Bank', y='Ratio_Liquidite (%)', color='Cluster', text='Ratio_Liquidite (%)', title="Ratio de Liquidité par Banque")

# Ajout des détails au graphique
fig.update_layout(xaxis_title='Banque', yaxis_title='Ratio de Liquidité (%)', legend_title='Cluster')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# Pour afficher le nom des banques sur le graphique, ajustez la disposition des axes et des étiquettes
fig.update_xaxes(tickangle=-45)

# Affichage du graphique
fig.show()

from sklearn.decomposition import PCA

# Réduction de la dimensionnalité
pca = PCA(n_components=2)  # Réduire à 2 dimensions pour la visualisation
pca_features = pca.fit_transform(scaled_features)

# Ajouter les composantes principales comme nouvelles colonnes pour la visualisation
banka_summary_df['PCA1'] = pca_features[:, 0]
banka_summary_df['PCA2'] = pca_features[:, 1]

import plotly.express as px

# Création du graphique des clusters
fig = px.scatter(banka_summary_df, x='PCA1', y='PCA2',
                 color='Cluster',
                 title="Visualisation des Clusters des Banques",
                 labels={'PCA1': 'Composante Principale 1', 'PCA2': 'Composante Principale 2'},
                 hover_data=['Bank'])  # Afficher le nom de la banque au survol

# Ajouter des détails au graphique
fig.update_layout(legend_title_text='Cluster')
fig.show()


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Préparation des données pour la régression linéaire
X = yearly_summary_df_adapted[['Year']].values.reshape(-1, 1)  # Feature matrix
y = yearly_summary_df_adapted['Ratio_Liquidite (%)'].values  # Target variable

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédictions
predictions = model.predict(X)

# Évaluation du modèle de régression linéaire
r2 = r2_score(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

print(f"R² score: {r2}")
print(f"RMSE: {rmse}")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Supposons que 'Year' a été remplacé par 'Bank' juste pour l'exemple
# Assurez-vous que 'Bank' est la colonne correcte à utiliser

# Sélection de la colonne numérique comme variable explicative
X = yearly_summary_df_adapted[['Liquidites', 'Creances_Financieres']].values
y = yearly_summary_df_adapted['Ratio_Liquidite (%)'].values  

# Création et entraînement du modèle de régression linéaire
linear_model = LinearRegression()
linear_model.fit(X, y)

# Prédictions avec le modèle de régression linéaire
linear_predictions = linear_model.predict(X)

# Calcul des métriques d'évaluation pour le modèle de régression linéaire
linear_mse = mean_squared_error(y, linear_predictions)
linear_r2 = r2_score(y, linear_predictions)

# Création et entraînement du modèle de forêt aléatoire
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Prédictions avec le modèle de forêt aléatoire
rf_predictions = rf_model.predict(X)

# Calcul des métriques d'évaluation pour le modèle de forêt aléatoire
rf_mse = mean_squared_error(y, rf_predictions)
rf_r2 = r2_score(y, rf_predictions)

# Affichage des résultats
print("Métriques d'évaluation pour la régression linéaire :")
print("MSE :", linear_mse)
print("R² :", linear_r2)
print("\nMétriques d'évaluation pour le modèle de forêt aléatoire :")
print("MSE :", rf_mse)
print("R² :", rf_r2)

import matplotlib.pyplot as plt

# Plotting the real values against the predictions for both Linear Regression and Random Forest
plt.figure(figsize=(10, 6))

# Plotting Linear Regression
plt.scatter(y, linear_predictions, color='blue', label='Linear Regression')

# Plotting Random Forest
plt.scatter(y, rf_predictions, color='green', label='Random Forest')

# Adding labels and title
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Comparaison des Prédictions - Régression Linéaire vs. Forêt Aléatoire')
plt.legend()
plt.grid(True)

# Showing the plot
plt.show()

import matplotlib.pyplot as plt

# Définir la largeur des barres d'histogramme
bar_width = 0.35

# Positions des barres pour les deux ensembles de données
index = np.arange(len(y))

# Créer une figure et des axes
plt.figure(figsize=(10, 6))

# Tracer l'histogramme des valeurs réelles
plt.bar(index, y, bar_width, color='blue', label='Valeurs Réelles')

# Tracer l'histogramme des prédictions de la régression linéaire
plt.bar(index + bar_width, linear_predictions, bar_width, color='green', label='Prédictions Régression Linéaire')

# Tracer l'histogramme des prédictions de la forêt aléatoire
plt.bar(index + 2*bar_width, rf_predictions, bar_width, color='red', label='Prédictions Forêt Aléatoire')

# Ajouter des étiquettes sur l'axe x
plt.xlabel('Échantillons')
plt.ylabel('Valeurs')
plt.title('Comparaison des Prédictions - Régression Linéaire vs. Forêt Aléatoire')
plt.xticks(index + bar_width, index)
plt.legend()

# Afficher le graphique
plt.tight_layout()
plt.show()
