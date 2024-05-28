from pyspark.sql import SparkSession

# Create a SparkSession

spark = SparkSession.builder \

    .appName("Lina") \

    .getOrCreate()

 

data_file_path = r"C:\Users\SBS\Downloads\data_avec_mode_apresValManquantes_et_Aberrantes.csv"

# Load the data into a DataFrame

df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(data_file_path)

 

# Show the DataFrame to verify it's loaded correctly

df.show().toPandas()

 

from pyspark.sql.functions import col, explode, array, lit

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml import Pipeline

from pyspark.sql import DataFrame

 

def upsample_minority_class(data, class_column, major_class_label, minor_class_label):

    # Calculer le nombre de fois que la classe minoritaire doit être sur-échantillonnée

    major_df = data.filter(col(class_column) == major_class_label)

    minor_df = data.filter(col(class_column) == minor_class_label)

    ratio = major_df.count() / minor_df.count()

 

    # Créer une colonne qui contient un tableau de 'ratio' fois la même ligne

    repeat_col = array([lit(1)] * int(ratio))

    minor_df = minor_df.withColumn("repeat_col", explode(repeat_col))

 

    # Fusionner les deux DataFrames

    combined_df = major_df.unionAll(minor_df.drop('repeat_col'))

 

    return combined_df

 

# Convertir la colonne cible en type double (nécessaire pour la classification dans Spark)

df = df.withColumn("flag_v0", df["flag_v0"].cast("double"))

 

# Diviser les données en ensembles d'entraînement et de test

(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

 

# Appliquer la fonction d'équilibrage uniquement sur l'ensemble d'entraînement

train_balanced_df = upsample_minority_class(train_data, 'flag_v0', major_class_label=0, minor_class_label=1)

 

# Afficher la distribution après rééquilibrage

train_balanced_df.groupBy('flag_v0').count().show()

 

from pyspark.sql.functions import col

 

# Supposons que balanced_df est le DataFrame résultant après l'application de sur-échantillonnage

# Calculer les comptes pour chaque classe

balanced_class_counts = train_balanced_df.groupBy('flag_v0').count()

 

# Calculer le total des comptes pour normaliser

total_balanced_count = balanced_class_counts.groupBy().sum('count').collect()[0][0]

 

# Ajouter une colonne avec le taux normalisé pour chaque classe

balanced_class_rates = balanced_class_counts.withColumn('rate', col('count') / total_balanced_count)

 

# Afficher les résultats

balanced_class_rates.show()

 

 

# import numpy as np

from pyspark.ml.linalg import DenseMatrix  # Assurez-vous d'importer DenseMatrix si vous l'utilisez

 

def extract_correlated_features(correlation_matrix, feature_names, threshold):

    """

    Cette fonction extrait les paires de fonctionnalités corrélées à partir de la matrice de corrélation.

 

    :param correlation_matrix: Matrice de corrélation des caractéristiques.

    :param feature_names: Liste des noms des fonctionnalités.

    :param threshold: Seuil de corrélation pour considérer les caractéristiques comme corrélées.

    :return: Une liste de paires de noms de fonctionnalités corrélées.

    """

    # Vérifier si correlation_matrix est une instance de DenseMatrix et la convertir en ndarray si nécessaire

    if isinstance(correlation_matrix, DenseMatrix):

        correlation_matrix = correlation_matrix.toArray()

 

    # Créer un masque booléen où les éléments sont vrais pour les valeurs de corrélation supérieures au seuil

    mask = np.abs(correlation_matrix) > threshold

   

    # Extraire les indices des éléments vrais dans le masque

    indices = np.where(mask)

   

    # Créer une liste de paires de noms de fonctionnalités corrélées

    correlated_feature_names = [(feature_names[i], feature_names[j]) for i, j in zip(*indices) if i != j and i < j]

   

    return correlated_feature_names

 

# Exemple d'utilisation

# Supposons que vous avez déjà calculé votre matrice de corrélation et que vous l'avez stockée dans 'correlation_matrix'

# Supposons également que vous avez une liste de noms de fonctionnalités stockée dans 'feature_names'

# Et que vous avez un seuil de corrélation de 0.8

threshold = 0.8

 

# Liste des colonnes à extraire

colonnes_septembre = [

    'v1_sept1', 'v2_sept1', 'v3sept1', 'v4_sept1_winsorized', 'v5_sept1_winsorized', 'v6_sept1_winsorized',

    'v7_sept1_winsorized', 'v8_sept1_winsorized', 'v9_sept1_winsorized', 'v10_sept1_winsorized',

    'v11_sept1_winsorized', 'v12_sept1_winsorized', 'v13_sept1_winsorized', 'v14_sept1_winsorized',

    'v15_sept1_winsorized', 'v16_sept1_winsorized', 'v17_sept1_winsorized', 'v18_sept1_winsorized',

    'v19_sept1_winsorized', 'v20_sept1_winsorized', 'v21_sept1_winsorized', 'v22_sept1_winsorized',

    'v23_sept1_winsorized', 'v24_sept1_winsorized', 'v25_sept1_winsorized', 'v26_sept1_winsorized',

    'v27_sept1_winsorized', 'v28_sept1_winsorized', 'v29_sept1_winsorized', 'v30_sept1_winsorized',

    'v31_sept1_winsorized', 'v32_sept1_winsorized',

    'total_usage_voix_sept', 'total_in_and_out_contacts_sept',

    'total_nb_op_sept', 'total_revenu_op_sept', 'total_montant_achat_sos_sept',

    'total_montant_remb_sos_sept', 'total_recharge_sept', 'total_nombre_recharge_sept'

]

# 'flag_op_sept',

 

# Créer le DataFrame "septembre" en extrayant les colonnes spécifiées de df

septembre = df[colonnes_septembre]

 

# Afficher les premières lignes de septembre pour vérifier l'extraction

print(septembre.head())

 

septembre.show()

 

from pyspark.ml.stat import Correlation

from pyspark.ml.feature import VectorAssembler

from pyspark.sql.functions import col

 

# Convertissez les colonnes du DataFrame Spark en un vecteur d'assemblage requis par la fonction de corrélation.

# Supposons que toutes les colonnes du DataFrame `septembre` sont des variables numériques.

assembler = VectorAssembler(inputCols=[c for c in septembre.columns], outputCol="features")

 

# Transformez le DataFrame pour avoir une colonne de fonctionnalités de vecteur.

septembre_vector = assembler.transform(septembre).select("features")

 

# Calculez la matrice de corrélation.

matrix_septembre = Correlation.corr(septembre_vector, "features").head()[0]

 

# Affichez la matrice de corrélation.

print(matrix_septembre)

 

import numpy as np

 

correlated_feature_names_septembre = extract_correlated_features(matrix_septembre, septembre.columns, threshold)

print("Pairs de noms de fonctionnalités corrélées:", correlated_feature_names_septembre)

 

df.columns

 

# Liste des colonnes pour octobre

colonnes_octobre = [

'v1_oct1',

'v2_oct1',

'v3oct1',

'v4_oct1_winsorized',

'v5_oct1_winsorized',

'v6_oct1_winsorized',

'v7_oct1_winsorized',

'v8_oct1_winsorized',

'v9_oct1_winsorized',

'v10_oct1_winsorized',

'v11_oct1_winsorized',

'v12_oct1_winsorized',

'v13_oct1_winsorized',

'v14_oct1_winsorized',

'v15_oct1_winsorized',

'v16_oct1_winsorized',

'v17_oct1_winsorized',

'v18_oct1_winsorized',

'v19_oct1_winsorized',

'v20_oct1_winsorized',

'v21_oct1_winsorized',

'v22_oct1_winsorized',

'v23_oct1_winsorized',

'v24_oct1_winsorized',

'v25_oct1_winsorized',

'v26_oct1_winsorized',

'v27_oct1_winsorized',

'v28_oct1_winsorized',

'v29_oct1_winsorized',

'v30_oct1_winsorized',

'v31_oct1_winsorized',

'v32_oct1_winsorized',

'total_usage_voix_oct',

'total_in_and_out_contacts_oct',

'total_nb_op_oct',

'total_revenu_op_oct',

'total_montant_achat_sos_oct',

'total_montant_remb_sos_oct',

'total_recharge_oct',

'total_nombre_recharge_oct'

]

#  'flag_op_oct1'

 

# Sélectionner les colonnes spécifiées pour octobre

octobre = df[colonnes_octobre]

 

# Afficher les premières lignes de octobre pour vérifier l'extraction

print(octobre.head())

 

colonnes_octobre

assembler = VectorAssembler(inputCols=[c for c in octobre.columns], outputCol="features")

 

# Transformez le DataFrame pour avoir une colonne de fonctionnalités de vecteur.

octobre_vector = assembler.transform(octobre).select("features")

 

# Calculez la matrice de corrélation.

matrix_octobre = Correlation.corr(octobre_vector, "features").head()[0]

 

# Affichez la matrice de corrélation.

print(matrix_octobre)

 

correlated_feature_names_octobre = extract_correlated_features(matrix_octobre, octobre.columns, threshold)

print("Pairs de noms de fonctionnalités corrélées:", correlated_feature_names_octobre)

 

# Liste des colonnes pour novembre

colonnes_novembre = [

    'v1_nov1', 'v2_nov1', 'v3nov1', 'v4_nov1_winsorized', 'v5_nov1_winsorized', 'v6_nov1_winsorized',

    'v7_nov1_winsorized', 'v8_nov1_winsorized', 'v9_nov1_winsorized', 'v10_nov1_winsorized',

    'v11_nov1_winsorized', 'v12_nov1_winsorized', 'v13_nov1_winsorized', 'v14_nov1_winsorized',

    'v15_nov1_winsorized', 'v16_nov1_winsorized', 'v17_nov1_winsorized', 'v18_nov1_winsorized',

    'v19_nov1_winsorized', 'v20_nov1_winsorized', 'v21_nov1_winsorized', 'v22_nov1_winsorized',

    'v23_nov1_winsorized', 'v24_nov1_winsorized', 'v25_nov1_winsorized', 'v26_nov1_winsorized',

    'v27_nov1_winsorized', 'v28_nov1_winsorized', 'v29_nov1_winsorized', 'v30_nov1_winsorized',

    'v31_nov1_winsorized', 'v32_nov1_winsorized',

    'total_usage_voix_nov', 'total_in_and_out_contacts_nov',

    'total_nb_op_nov', 'total_revenu_op_nov', 'total_montant_achat_sos_nov',

    'total_montant_remb_sos_nov', 'total_recharge_nov', 'total_nombre_recharge_nov'

]

 

 

# Sélectionner les colonnes spécifiées pour octobre

novembre = df[colonnes_novembre]

 

# Afficher les premières lignes de octobre pour vérifier l'extraction

print(novembre.head())

 

assembler = VectorAssembler(inputCols=[c for c in novembre.columns], outputCol="features")

 

# Transformez le DataFrame pour avoir une colonne de fonctionnalités de vecteur.

novembre_vector = assembler.transform(novembre).select("features")

 

# Calculez la matrice de corrélation.

matrix_novembre = Correlation.corr(novembre_vector, "features").head()[0]

 

# Affichez la matrice de corrélation.

print(matrix_novembre)

correlated_feature_names_novembre = extract_correlated_features(matrix_novembre, novembre.columns, threshold)

print("Pairs de noms de fonctionnalités corrélées:", correlated_feature_names_novembre)

 

# Liste des colonnes pour décembre

colonnes_decembre = [

    'v1_dec1', 'v2_dec1', 'v3dec1', 'v4_dec1_winsorized', 'v5_dec1_winsorized', 'v6_dec1_winsorized',

    'v7_dec1_winsorized', 'v8_dec1_winsorized', 'v9_dec1_winsorized', 'v10_dec1_winsorized',

    'v11_dec1_winsorized', 'v12_dec1_winsorized', 'v13_dec1_winsorized', 'v14_dec1_winsorized',

    'v15_dec1_winsorized', 'v16_dec1_winsorized', 'v17_dec1_winsorized', 'v18_dec1_winsorized',

    'v19_dec1_winsorized', 'v20_dec1_winsorized', 'v21_dec1_winsorized', 'v22_dec1_winsorized',

    'v23_dec1_winsorized', 'v24_dec1_winsorized', 'v25_dec1_winsorized', 'v26_dec1_winsorized',

    'v27_dec1_winsorized', 'v28_dec1_winsorized', 'v29_dec1_winsorized', 'v30_dec1_winsorized',

    'v31_dec1_winsorized', 'v32_dec1_winsorized',

    'total_usage_voix_dec', 'total_in_and_out_contacts_dec',

    'total_nb_op_dec', 'total_revenu_op_dec', 'total_montant_achat_sos_dec',

    'total_montant_remb_sos_dec', 'total_recharge_dec', 'total_nombre_recharge_dec'

]

 

# Sélectionner les colonnes spécifiées pour décembre

decembre = df.select(colonnes_decembre)

 

# Afficher les premières lignes de décembre pour vérifier l'extraction

decembre.show()

 

assembler = VectorAssembler(inputCols=[c for c in decembre.columns], outputCol="features")

 

# Transformez le DataFrame pour avoir une colonne de fonctionnalités de vecteur.

decembre_vector = assembler.transform(decembre).select("features")

 

# Calculez la matrice de corrélation.

matrix_decembre = Correlation.corr(decembre_vector, "features").head()[0]

 

# Affichez la matrice de corrélation.

print(matrix_decembre)

 

correlated_feature_names_decembre = extract_correlated_features(matrix_decembre, decembre.columns, threshold)

print("Pairs de noms de fonctionnalités corrélées:", correlated_feature_names_decembre)

 

# Liste des colonnes à extraire

colonnes_janvier = [

    'v1_jan1', 'v2_jan1', 'v3jan1', 'v4_jan1_winsorized', 'v5_jan1_winsorized', 'v6_jan1_winsorized',

    'v7_jan1_winsorized', 'v8_jan1_winsorized', 'v9_jan1_winsorized', 'v10_jan1_winsorized',

    'v11_jan1_winsorized', 'v12_jan1_winsorized', 'v13_jan1_winsorized', 'v14_jan1_winsorized',

    'v15_jan1_winsorized', 'v16_jan1_winsorized', 'v17_jan1_winsorized', 'v18_jan1_winsorized',

    'v19_jan1_winsorized', 'v20_jan1_winsorized', 'v21_jan1_winsorized', 'v22_jan1_winsorized',

    'v23_jan1_winsorized', 'v24_jan1_winsorized', 'v25_jan1_winsorized', 'v26_jan1_winsorized',

    'v27_jan1_winsorized', 'v28_jan1_winsorized', 'v29_jan1_winsorized', 'v30_jan1_winsorized',

    'v31_jan1_winsorized', 'v32_jan1_winsorized',

    'total_usage_voix_jan', 'total_in_and_out_contacts_jan',

    'total_nb_op_jan', 'total_revenu_op_jan', 'total_montant_achat_sos_jan',

    'total_montant_remb_sos_jan', 'total_recharge_jan', 'total_nombre_recharge_jan'

]

janvier = df.select(colonnes_janvier)

 

# Afficher les premières lignes de janvier pour vérifier l'extraction

janvier.show()

 

assembler = VectorAssembler(inputCols=[c for c in janvier.columns], outputCol="features")

 

# Transformez le DataFrame pour avoir une colonne de fonctionnalités de vecteur.

janvier_vector = assembler.transform(janvier).select("features")

 

# Calculez la matrice de corrélation.

matrix_janvier = Correlation.corr(janvier_vector, "features").head()[0]

 

# Affichez la matrice de corrélation.

print(matrix_janvier)

 

correlated_feature_names_janvier = extract_correlated_features(matrix_janvier,janvier.columns, threshold)

print("Pairs de noms de fonctionnalités corrélées:", correlated_feature_names_janvier)

 

# Liste des colonnes à extraire

colonnes_fevrier = [

    'v1_fev1', 'v2_fev1', 'v3fev1', 'v4_fev1_winsorized', 'v5_fev1_winsorized', 'v6_fev1_winsorized',

    'v7_fev1_winsorized', 'v8_fev1_winsorized', 'v9_fev1_winsorized', 'v10_fev1_winsorized',

    'v11_fev1_winsorized', 'v12_fev1_winsorized', 'v13_fev1_winsorized', 'v14_fev1_winsorized',

    'v15_fev1_winsorized', 'v16_fev1_winsorized', 'v17_fev1_winsorized', 'v18_fev1_winsorized',

    'v19_fev1_winsorized', 'v20_fev1_winsorized', 'v21_fev1_winsorized', 'v22_fev1_winsorized',

    'v23_fev1_winsorized', 'v24_fev1_winsorized', 'v25_fev1_winsorized',

    'v27_fev1_winsorized', 'v28_fev1_winsorized', 'v29_fev1_winsorized', 'v30_fev1_winsorized',

    'v31_fev1_winsorized', 'v32_fev1_winsorized',

    'total_usage_voix_fev', 'total_in_and_out_contacts_fev',

    'total_nb_op_fev', 'total_revenu_op_fev', 'total_montant_achat_sos_fev',

    'total_montant_remb_sos_fev', 'total_recharge_fev', 'total_nombre_recharge_fev'

]

 

fevrier = df.select(colonnes_fevrier)

 

# Afficher les premières lignes de février pour vérifier l'extraction

fevrier.show()

 

assembler = VectorAssembler(inputCols=[c for c in fevrier.columns], outputCol="features")

 

# Transformez le DataFrame pour avoir une colonne de fonctionnalités de vecteur.

fevrier_vector = assembler.transform(fevrier).select("features")

 

# Calculez la matrice de corrélation.

matrix_fevrier = Correlation.corr(fevrier_vector, "features").head()[0]

 

# Affichez la matrice de corrélation.

print(matrix_fevrier)

correlated_feature_names_fevrier = extract_correlated_features(matrix_fevrier,fevrier.columns, threshold)

print("Pairs de noms de fonctionnalités corrélées:", correlated_feature_names_fevrier)

 

all_correlated_features = []

 

# Concaténation des listes de paires de noms de fonctionnalités corrélées pour chaque mois

all_correlated_features.extend(correlated_feature_names_septembre)

all_correlated_features.extend(correlated_feature_names_octobre)

all_correlated_features.extend(correlated_feature_names_novembre)

all_correlated_features.extend(correlated_feature_names_decembre)

all_correlated_features.extend(correlated_feature_names_janvier)

all_correlated_features.extend(correlated_feature_names_fevrier)

 

# Afficher la liste de toutes les paires de noms de fonctionnalités corrélées

print("Toutes les paires de noms de fonctionnalités corrélées :", all_correlated_features)

 

from pyspark.sql import DataFrame

 

def remove_correlated_features(df, correlated_features):

    for feature1, feature2 in correlated_features:

        if feature1 in df.columns and feature2 in df.columns:

            df = df.drop(feature2)  # Pas besoin de spécifier l'axe pour PySpark

    return df

 

# Supprimer les fonctionnalités corrélées de la DataFrame principale

df_balanced_sans_correlation = remove_correlated_features(train_balanced_df, all_correlated_features)

 

# Afficher la DataFrame sans les fonctionnalités corrélées

print (df_balanced_sans_correlation)

 

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

from pyspark.sql import SparkSession

import numpy as np

 

 

# Préparation du DataFrame pour PCA

assembler = VectorAssembler(inputCols=df_balanced_sans_correlation.columns, outputCol="features")

df_vect = assembler.transform(df_balanced_sans_correlation)

 

#Normalisation des données

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

scalerModel = scaler.fit(df_vect)

df_scaled = scalerModel.transform(df_vect)

 

# Ajustement du modèle PCA

pca = PCA(k=len(df_balanced_sans_correlation.columns), inputCol="scaledFeatures", outputCol="pcaFeatures")

model = pca.fit(df_scaled)

 

# Récupération et affichage de la variance expliquée et de la variance cumulée

explained_variance = model.explainedVariance.toArray()

cumulative_variance = explained_variance.cumsum()

 

# Affichage de la variance expliquée et de la variance cumulée

for i, (exp_var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):

    print("Composante principale {}: Variance expliquée: {:.4f}, Variance cumulée: {:.4f}".format(i + 1, exp_var, cum_var))

 

# Application du critère de Kaiser

n_features = len(df_balanced_sans_correlation.columns)

kaiser_criteria = np.where(explained_variance >= 1.0 / n_features)[0]

k_optimal = kaiser_criteria[-1] + 1 if kaiser_criteria.size > 0 else 0

 

print("Nombre de composantes selon le critère de Kaiser: {}".format(k_optimal))

 

# Variables explicatives

feature_columns = [c for c in df_balanced_sans_correlation.columns if c != 'flag_v0']

 

# Créer un nouveau DataFrame sans la colonne 'flag_v0'

X = df_balanced_sans_correlation.select(*feature_columns)

 

#variable cible = Sélectionner la variable cible

# Si df_supp_sans_corr est un autre DataFrame contenant la colonne 'flag_v0', sélectionnez cette colonne

y = df_balanced_sans_correlation.select('flag_v0')

 

# Utiliser VectorAssembler pour combiner les colonnes de caractéristiques en un seul vecteur

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

df_vect = assembler.transform(X)

 

# Normaliser les features

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

scalerModel = scaler.fit(df_vect)

df_scaled = scalerModel.transform(df_vect)

 

# Appliquer PCA

pca = PCA(k=44, inputCol="scaledFeatures", outputCol="pcaFeatures")

model = pca.fit(df_scaled)

# Variables explicatives

feature_columns = [c for c in df_balanced_sans_correlation.columns if c != 'flag_v0']

 

# Créer un nouveau DataFrame sans la colonne 'flag_v0'

X = df_balanced_sans_correlation.select(*feature_columns)

 

#variable cible = Sélectionner la variable cible

# Si df_supp_sans_corr est un autre DataFrame contenant la colonne 'flag_v0', sélectionnez cette colonne

y = df_balanced_sans_correlation.select('flag_v0')

# Transformer le DataFrame pour obtenir les nouvelles composantes

X_pca = model.transform(df_scaled)

 

# Variables explicatives

feature_columns = [c for c in df_balanced_sans_correlation.columns if c != 'flag_v0']

 

# Créer un nouveau DataFrame sans la colonne 'flag_v0'

X = df_balanced_sans_correlation.select(*feature_columns)

 

#variable cible = Sélectionner la variable cible

# Si df_supp_sans_corr est un autre DataFrame contenant la colonne 'flag_v0', sélectionnez cette colonne

y = df_balanced_sans_correlation.select('flag_v0')

 

 

from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

from pyspark.ml.linalg import Vectors

 

# Sélectionner les variables explicatives et assembler dans un vecteur de features

feature_columns = [c for c in df_balanced_sans_correlation.columns if c != 'flag_v0']

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

df_vect = assembler.transform(df_balanced_sans_correlation)

 

# Normaliser les features

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

scalerModel = scaler.fit(df_vect)

df_scaled = scalerModel.transform(df_vect)

 

# Appliquer PCA

pca = PCA(k=44, inputCol="scaledFeatures", outputCol="pcaFeatures")

model = pca.fit(df_scaled)

 

# Transformer le DataFrame pour obtenir les nouvelles composantes

X_pca = model.transform(df_scaled)

 

# Sélectionner les composantes et la variable cible pour l'entraînement du modèle

df_pca_final = X_pca.select("pcaFeatures", "flag_v0")

 

# Maintenant, df_pca_final est prêt à être utilisé pour l'entraînement des modèles

 

# Initialiser le RandomForestClassifier

rf = RandomForestClassifier(labelCol="flag_v0", featuresCol="pcaFeatures", numTrees=100, maxDepth=8, maxBins=32, featureSubsetStrategy="auto")                                                      

                            

# Créer un pipeline qui inclut les étapes de transformation et le classificateur

pipeline = Pipeline(stages=[assembler, scaler, pca, rf])

 

# Entraîner le modèle en utilisant le pipeline sur l'ensemble d'entraînement équilibré

rf_model = pipeline.fit(df_balanced_sans_correlation)

 

# Prédire sur l'ensemble de test

predictions = rf_model.transform(test_data)

 

# Évaluation du modèle

evaluator = MulticlassClassificationEvaluator(labelCol="flag_v0", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})

weighted_precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})

weighted_recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

 

# Affichage des métriques

print("Rapport de classification :")

print("Précision pondérée : {:.2f}".format(weighted_precision))

print("Rappel pondéré : {:.2f}".format(weighted_recall))

print("Score F1 pondéré : {:.2f}".format(f1_score))

print("Précision globale : {:.2f}".format(accuracy))

 

 
