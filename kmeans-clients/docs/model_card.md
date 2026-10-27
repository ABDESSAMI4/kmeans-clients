# Model card - segmentation RFM V2

## Identification

- **Tâche:** segmentation non supervisée exploratoire de clients.
- **Algorithme de démonstration:** K-Means, K=3, `n_init=20`, graine 42.
- **Artefact:** `models/kmeans_rfm_pipeline.pkl`.
- **Métadonnées:** `models/kmeans_rfm_meta.json`.
- **Données de référence:** 2 240 lignes et 29 colonnes dans le CSV fourni.
- **Versionnement des entrées:** empreinte SHA-256 du CSV consignée dans les métadonnées.

## Population et limites des données

Le fichier reçu est associé au jeu public *Customer Personality Analysis*, dont l'attribution Kaggle reste à confirmer à partir d'une source primaire. La licence, la période exacte de collecte, les unités des montants et la représentativité ne sont pas certifiées localement. Le CSV comprend 24 revenus manquants, aucun doublon complet observé et aucun ID dupliqué. La date d'inscription est interprétée avec `Dt_Customer`; l'année de référence 2014 de la tenure est une hypothèse imposée par l'étude de cas.

## Variables RFM et transformations

Les cinq dimensions entrées dans le modèle sont:

| Variable | Définition |
|---|---|
| Recency | Valeur de la colonne `Recency`, en jours selon le jeu |
| Frequency | Somme de `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases` et `NumStorePurchases` |
| Monetary | Somme des six colonnes `Mnt*` |
| Tenure | `2014 - year(Dt_Customer)` selon l'année de référence supposée |
| Income | Revenu fourni, imputé par médiane si manquant |

`Frequency`, `Monetary` et `Income` sont transformées par `log1p`, puis les cinq variables sont standardisées. Le plafonnement au 99e percentile et l'imputation par médiane sont appris au `fit`, puis réutilisés à l'inférence. La tenure par année entière est grossière: elle prend peu de valeurs sur ce jeu et doit être remplacée par une référence de date vérifiée dans une application réelle.

## Prévention de fuite de cible

`Response` et `AcceptedCmp*` ne sont pas des colonnes d'entrée requises et ne sont jamais utilisées par `fit` ni par `predict`. `Response` est éventuellement agrégée après le clustering pour une description statistique, avec une interprétation non causale.

## Validation robuste

Pour chaque K de 2 à 9 et les graines `[42, 1, 123, 2024, 999]`, le protocole ajuste un pipeline de référence sur toutes les lignes et effectue 10 bootstrap avec remise, de taille 80 %. Chaque bootstrap réajuste le constructeur RFM, l'imputation, les plafonds, les transformations, le scaler et K-Means. Les deux modèles prédisent ensuite les mêmes lignes du fichier complet et l'ARI mesure l'accord des partitions. `outputs/stability_v2.csv` conserve moyenne, écart-type et les 10 valeurs par graine et K.

## Benchmark et résultats

Les candidats comparés à K=3 sont K-Means sans log1p, K-Means avec log1p, GaussianMixture, Agglomerative Ward et HDBSCAN. Silhouette (échantillon reproductible jusqu'à 1 200 clients), Davies-Bouldin et Calinski-Harabasz sont reportés dans `outputs/comparaison_modeles.csv`. Pour HDBSCAN, les métriques sont calculées sur les points non classés comme bruit; le taux de bruit est enregistré à part. Ces scores décrivent seulement la structure géométrique de cet échantillon.

Sur l'exécution de référence, la stabilité ARI moyenne (moyennée sur les cinq graines) est environ 0,983 pour K=2 et 0,950 pour K=3. L'évaluation K-Means RFM donne une silhouette d'environ 0,384 à K=2 et 0,279 à K=3. Le modèle de démonstration est cependant fixé à K=3 à des fins de segmentation business plus détaillée; ce choix n'est pas présenté comme l'optimum géométrique.

## Interprétation business

Les noms observés pour le run sont Champions (668 clients), À réactiver (707; dépense élevée, achats plus anciens) et Petits acheteurs (865; faible fréquence et dépense). Ils sont assignés par rang relatif sur Recency, Frequency et Monetary, pas par étiquettes métier de référence. Les actions, canaux et offres sont des hypothèses à tester. Les KPI proposés servent de mesures; aucune valeur cible commerciale ne peut être justifiée par ce jeu seul. `outputs/profils_rfm_k3.csv`, `outputs/boxplots_rfm_k3.png` et `outputs/radar_rfm_k3.png` présentent les distributions et profils en unités d'origine ou normalisés selon la figure.

Les écarts de réponse aux campagnes sont descriptifs et post-clustering. Ils ne démontrent ni que l'appartenance à un cluster entraîne une réponse, ni qu'une campagne améliore les conversions. La simulation d'addition de revenu du dashboard utilise un écart Monetary hypothétique entre le segment À réactiver et les Champions; ce n'est pas une prévision, une estimation d'uplift ni un chiffre causal.

## Usages prévus et exclus

**Prévus:** portfolio, démonstration de pipeline, exploration initiale de données marketing et illustration d'un flux de prédiction.

**Exclus sans validation supplémentaire:** déploiement direct pour le ciblage réel, décisions défavorables envers des personnes, communication d'un uplift causal, promesses de revenus, ou généralisation à une clientèle réelle.

## Étapes avant tout usage opérationnel

1. Confirmer provenance, licence, dictionnaire, unités, date de référence et autorisation des données.
2. Remplacer le CSV ancien par une population cible récente, pertinente et représentative.
3. Valider les définitions RFM, notamment la tenure en jours ou mois avec une date d'observation vérifiée.
4. Tester la stabilité temporelle et par sous-populations; examiner l'effet des variables et du plafonnement.
5. Faire valider l'actionnabilité par les équipes métier.
6. Mesurer l'effet marketing via une expérimentation randomisée et suivre marge, coût et effets indésirables.
7. Ajouter monitoring des distributions, des affectations et des versions avant industrialisation.
