import pandas as pd
import matplotlib.pyplot as plt

# Chargement du fichier Excel
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xl = pd.ExcelFile(file_path)

# Initialisation d'un DataFrame pour stocker les résultats
results_df = pd.DataFrame()

for year in range(2010, 2023):  # Pour chaque année de 2010 à 2022
    bilan_sheet_name = f'BILAN {year}'
    resultat_sheet_name = f'E Rslt {year}'
    
    if bilan_sheet_name in xl.sheet_names and resultat_sheet_name in xl.sheet_names:
        # Chargement des données des feuilles de bilan et de résultat pour l'année courante
        bilan_data = pd.read_excel(file_path, sheet_name=bilan_sheet_name, usecols=["Secteur"])
        resultat_data = pd.read_excel(file_path, sheet_name=resultat_sheet_name, usecols=["Secteur"])
        
        # Calcul des ratios en utilisant les premières lignes pour une approximation
        actifs_courants_approx = bilan_data.iloc[0]['Secteur']
        passifs_courants_approx = bilan_data.iloc[1]['Secteur']
        capitaux_propres_approx = bilan_data.iloc[2]['Secteur']  # Approximation pour les capitaux propres
        resultat_exploitation_approx = resultat_data.iloc[0]['Secteur']
        charges_interet_approx = resultat_data.iloc[1]['Secteur']
        
        ratio_liquidite_courante_approx = actifs_courants_approx / passifs_courants_approx if passifs_courants_approx else pd.NA
        ratio_dette_sur_fonds_propres_approx = (actifs_courants_approx + passifs_courants_approx) / capitaux_propres_approx if capitaux_propres_approx else pd.NA
        ratio_couverture_interets_approx = resultat_exploitation_approx / charges_interet_approx if charges_interet_approx else pd.NA
        
        # Ajout des résultats approximatifs au DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'Année': [year],
            'Ratio de Liquidité Courante Approximatif': [ratio_liquidite_courante_approx],
            'Ratio Dette sur Fonds Propres Approximatif': [ratio_dette_sur_fonds_propres_approx],
            'Ratio de Couverture des Intérêts Approximatif': [ratio_couverture_interets_approx]
        })], ignore_index=True)

# Configuration de 'Année' comme index du DataFrame
results_df.set_index('Année', inplace=True)

# Affichage du DataFrame avec les résultats
print(results_df)

# Génération des graphiques pour visualiser les résultats
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

results_df['Ratio de Liquidité Courante Approximatif'].plot(kind='bar', ax=axs[0], color='skyblue')
axs[0].set_title('Ratio de Liquidité Courante Approximatif par Année')
axs[0].set_ylabel('Ratio')

results_df['Ratio Dette sur Fonds Propres Approximatif'].plot(kind='bar', ax=axs[1], color='lightgreen')
axs[1].set_title('Ratio Dette sur Fonds Propres Approximatif par Année')
axs[1].set_ylabel('Ratio')

results_df['Ratio de Couverture des Intérêts Approximatif'].plot(kind='bar', ax=axs[2], color='salmon')
axs[2].set_title('Ratio de Couverture des Intérêts Approximatif par Année')
axs[2].set_ylabel('Ratio')

plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du fichier Excel
file_path = 'C:/Users/ASUS TUF I5/Desktop/PFE/ETUDE DU SECTEUR/EtudeSB - Copie.xlsx'
xl = pd.ExcelFile(file_path)

# Initialisation d'un DataFrame pour stocker les résultats
results_df = pd.DataFrame()

for year in range(2010, 2023):  # Pour chaque année de 2010 à 2022
    bilan_sheet_name = f'BILAN {year}'
    resultat_sheet_name = f'E Rslt {year}'
    
    if bilan_sheet_name in xl.sheet_names and resultat_sheet_name in xl.sheet_names:
        # Chargement des données des feuilles de bilan et de résultat pour l'année courante
        bilan_data = pd.read_excel(file_path, sheet_name=bilan_sheet_name, usecols=["Secteur"])
        resultat_data = pd.read_excel(file_path, sheet_name=resultat_sheet_name, usecols=["Secteur"])
        
        # Calcul des ratios en utilisant les premières lignes pour une approximation
        actifs_courants_approx = bilan_data.iloc[0]['Secteur']
        passifs_courants_approx = bilan_data.iloc[1]['Secteur']
        capitaux_propres_approx = bilan_data.iloc[2]['Secteur']
        resultat_exploitation_approx = resultat_data.iloc[0]['Secteur']
        charges_interet_approx = resultat_data.iloc[1]['Secteur']
        
        ratio_liquidite_courante_approx = actifs_courants_approx / passifs_courants_approx if passifs_courants_approx else pd.NA
        ratio_dette_sur_fonds_propres_approx = (actifs_courants_approx + passifs_courants_approx) / capitaux_propres_approx if capitaux_propres_approx else pd.NA
        ratio_couverture_interets_approx = resultat_exploitation_approx / charges_interet_approx if charges_interet_approx else pd.NA
        
        # Ajout des résultats approximatifs au DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'Année': [year],
            'Ratio de Liquidité Courante Approximatif': [ratio_liquidite_courante_approx],
            'Ratio Dette sur Fonds Propres Approximatif': [ratio_dette_sur_fonds_propres_approx],
            'Ratio de Couverture des Intérêts Approximatif': [ratio_couverture_interets_approx]
        })], ignore_index=True)

# Normalisation des données avant le clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(results_df[['Ratio de Liquidité Courante Approximatif', 'Ratio Dette sur Fonds Propres Approximatif', 'Ratio de Couverture des Intérêts Approximatif']].fillna(0))

# Application de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
results_df['Cluster'] = kmeans.labels_

# Regroupement des années par cluster
results_df['Année'] = results_df['Année'].astype(str)  # Assurer que l'année est en format string pour l'affichage
grouped_years = results_df.groupby('Cluster')['Année'].apply(lambda x: ', '.join(x)).reset_index()

# Affichage des années regroupées par cluster
for index, row in grouped_years.iterrows():
    print(f"Cluster {row['Cluster']} contient les années suivantes : {row['Année']}")

# Optionnel: Visualisation des clusters
sns.pairplot(results_df, hue='Cluster', vars=['Ratio de Liquidité Courante Approximatif', 'Ratio Dette sur Fonds Propres Approximatif', 'Ratio de Couverture des Intérêts Approximatif'])
plt.title('Visualisation des Clusters')
plt.show()

from tabulate import tabulate

# Création de DataFrames pour chaque cluster
cluster_tables = []

for cluster_id in range(kmeans.n_clusters):
    cluster_years = grouped_years.iloc[cluster_id]['Année'].split(', ')
    cluster_values = results_df[results_df['Cluster'] == cluster_id].drop(columns=['Cluster'])
    cluster_data = []
    for year in cluster_years:
        row = cluster_values[cluster_values['Année'] == year].squeeze().to_dict()
        row['Année'] = year
        cluster_data.append(row)

    cluster_table = tabulate(cluster_data, headers='keys', tablefmt='grid')
    cluster_tables.append(cluster_table)

# Ajouter un message pour le cluster manquant
if len(cluster_tables) < kmeans.n_clusters:
    cluster_tables.append("Aucune donnée disponible pour le cluster 2.")

# Affichage des tableaux pour chaque cluster
for i, table in enumerate(cluster_tables):
    print(f"Cluster {i}:")
    print(table)
    print("\n")

from tabulate import tabulate

# Récupération des années associées au cluster 2
cluster_years = grouped_years[grouped_years['Cluster'] == 2]['Année'].values[0].split(', ')

# Récupération des données pour le cluster 2
cluster_values = results_df[results_df['Cluster'] == 2].drop(columns=['Cluster'])

# Création de la table pour le cluster 2
cluster_data = []
for year in cluster_years:
    row = cluster_values[cluster_values['Année'] == year].squeeze().to_dict()
    row['Année'] = year
    cluster_data.append(row)

cluster_table = tabulate(cluster_data, headers='keys', tablefmt='grid')

# Affichage de la table pour le cluster 2
print("Cluster 2:")
print(cluster_table)
