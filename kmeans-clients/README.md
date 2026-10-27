# Segmentation clients RFM - Customer Personality Analysis

Projet de segmentation non supervisée avec pipeline scikit-learn réutilisable, validation par bootstrap avec refit complet, benchmark multi-algorithmes, dashboard Streamlit et API FastAPI.

> Le CSV de 2 240 clients et 29 colonnes est associé au jeu public *Customer Personality Analysis*. Sa provenance primaire, sa licence, sa période exacte et sa représentativité ne sont pas vérifiables depuis le seul fichier. Le projet reste une étude de cas, pas une preuve sur une entreprise réelle.

## 1. Objectif et résultat

Le but est de construire des groupes compréhensibles à partir des comportements d'achat, de comparer des méthodes et de fournir une démonstration reproductible de prédiction.

Le modèle utilisé pour la démonstration business est un K-Means à **K=3** sur cinq dimensions RFM. Il donne plus de granularité au profilage. Ce choix n'est pas présenté comme l'optimum géométrique: sur le fichier fourni, K=2 a la silhouette la plus élevée (environ 0,384) et une stabilité bootstrap supérieure. À K=3, silhouette environ 0,279 et ARI moyen environ 0,950 sur la moyenne des cinq graines. Le benchmarking confirme que les algorithmes ne partitionnent pas les données de la même façon.

## 2. Variables et justification RFM

Le pipeline transforme le schéma brut en cinq variables métier:

| Variable | Définition | Intérêt analytique |
|---|---|---|
| Recency | `Recency` en jours | Mesurer la proximité du dernier achat; une valeur basse signifie un achat plus récent |
| Frequency | Somme de `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` | Approximer l'intensité et la fréquence d'achat tous canaux confondus |
| Monetary | Somme des six colonnes `Mnt*` | Résumer la valeur d'achat observée dans les catégories produits |
| Tenure | `2014 - year(Dt_Customer)` | Approximer l'ancienneté selon l'année de référence demandée |
| Income | `Income` | Apporter un indicateur socio-économique complémentaire au comportement observé |

Le plafonnement au 99e percentile et l'imputation médiane sont appris par `fit` dans chaque pipeline. `log1p` est appliqué à Frequency, Monetary et Income, puis les cinq variables sont standardisées. Les variables de campagne `Response` et `AcceptedCmp*` sont exclues de l'entraînement et de la prédiction; `Response` ne sert qu'à un descriptif post-clustering.

La tenure par année entière est une approximation grossière: les dates s'étendent sur peu d'années et l'année 2014 est supposée. En production, confirmer la date de collecte et calculer une durée depuis une date d'observation métier valide. Les quatre composantes de Frequency sont additionnées même si achats promotionnels et achats par canal peuvent ne pas représenter des événements parfaitement disjoints; cette définition doit être validée avec le dictionnaire des données.

## 3. Validation et benchmark

### Stabilité par bootstrap

Pour chaque K=2..9, les graines `[42, 1, 123, 2024, 999]` et 10 bootstraps de 80 % avec remise sont exécutés. Dans chaque réplication, le constructeur RFM, l'imputation, le plafonnement, les transformations, le scaler et K-Means sont entièrement réajustés. Les modèles de référence et bootstrap prédisent le même fichier complet; l'ARI compare les partitions alignées sur les mêmes clients. Les moyennes, écarts-types et dix scores individuels sont écrits dans `outputs/stability_v2.csv`.

### Comparaison des algorithmes

Le benchmark à trois groupes compare K-Means sans log1p, K-Means avec log1p, GaussianMixture, Agglomerative Ward et HDBSCAN. Il calcule silhouette sur jusqu'à 1 200 observations, Davies-Bouldin, Calinski-Harabasz, nombre de clusters et fraction de bruit. Pour HDBSCAN, les métriques excluent les observations étiquetées bruit; ce traitement est indiqué dans les métadonnées. Les résultats ne sont pas une validation business.

## 4. Profils et recommandations

Les noms business sont attribués à partir des profils observés: **Champions** (valeur élevée et achats plus récents), **À réactiver** (valeur élevée mais achats plus anciens, donc Recency plus élevée) et **Petits acheteurs** (faible fréquence et montant). Dans ce run, les effectifs sont respectivement 668, 707 et 865 clients. Les moyennes Monetary sont environ 943, 943 et 70. Ces noms résument les profils du fichier et restent à faire valider par les équipes métier. Les profils comprennent moyennes et médianes en unités du fichier, distributions boxplot et radar normalisé. Le canal suggéré est le canal ayant le volume moyen d'achat le plus fort dans le segment.

Les offres et actions sont des hypothèses, avec KPI à suivre (réachat, marge nette, réactivation, panier, coût et désabonnements). Aucune valeur cible ne peut être déduite sans baseline métier. Les différences des taux de réponse sont descriptives. La simulation du dashboard calcule un scénario de revenu à partir de l'écart Monetary moyen entre clients à réactiver et Champions; elle ne mesure pas un uplift causal, ne tient pas compte des coûts et ne constitue pas une prévision.

## 5. Installation et exécution

Depuis la racine du dépôt, avec Python 3.13:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install -e .
```

Lancer le réentraînement et tous les diagnostics (dont 400 fits bootstrap):

```powershell
$env:PYTHONPATH = "src"
python -m customer_segmentation.pipeline
```

Pour produire les artefacts plus rapidement en sautant la stabilité:

```powershell
python -m customer_segmentation.pipeline --skip-stability
```

Exécuter les tests:

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

## 6. Dashboard Streamlit V2

Après génération des artefacts:

```powershell
streamlit run app/dashboard.py
```

Le dashboard comprend quatre vues:

1. KPIs, courbes elbow/silhouette interactives et benchmark sélectionnable par métrique.
2. Radar, profilage RFM en unités originales, statistiques moyennes/médianes, PCA 3D et boxplots.
3. Import CSV, affectation par le pipeline sauvegardé et recommandation par segment.
4. Simulation paramétrable de revenu additionnel, clairement marquée comme scénario non causal.

## 7. API FastAPI

L'API utilise le même pipeline et le même mapping de recommandations que le dashboard:

```powershell
uvicorn app.api:app --reload
```

Documentation interactive: `http://127.0.0.1:8000/docs`.

Exemple `POST /predict`:

```json
{
  "Income": 58138,
  "Recency": 58,
  "NumDealsPurchases": 3,
  "NumWebPurchases": 8,
  "NumCatalogPurchases": 10,
  "NumStorePurchases": 4,
  "MntWines": 635,
  "MntFruits": 88,
  "MntMeatProducts": 546,
  "MntFishProducts": 172,
  "MntSweetProducts": 88,
  "MntGoldProds": 88,
  "Dt_Customer": "04-09-2012"
}
```

`Income` peut être nul et sera imputé; les autres variables nécessaires doivent être fournies. Les champs de campagne supplémentaires sont ignorés.

## 8. Artefacts générés

```text
models/kmeans_rfm_pipeline.pkl       # Feature builder + imputeur/capper + scaler + K-Means
models/kmeans_rfm_meta.json          # empreinte, métriques, mapping métier, versions
outputs/stability_v2.csv             # ARI par K/graine, moyenne, std et 10 bootstraps
outputs/comparaison_modeles.csv      # métriques benchmark
outputs/comparaison_modeles.png      # graphique multi-métriques
outputs/comparaison_k_rfm.csv        # inertie et scores K-Means pour K=2..9
outputs/clients_segmentes_rfm_k3.csv # données analysées, variables RFM et affectation
outputs/profils_rfm_k3.csv           # moyennes et médianes des variables et composantes
outputs/boxplots_rfm_k3.png          # distributions des cinq variables RFM
outputs/radar_rfm_k3.png             # radar des moyennes normalisées
app/dashboard.py                    # dashboard Streamlit
app/api.py                          # endpoint FastAPI
tests/test_pipeline.py               # tests de transformation, fit/predict et absence de fuite
```

## 9. Architecture

- `src/customer_segmentation/pipeline.py`: feature engineering RFM, transformations, bootstrap, benchmark, profiling et sérialisation.
- `models/`: pipeline et métadonnées versionnées.
- `app/dashboard.py`: quatre pages Streamlit.
- `app/api.py`: endpoints `/health` et `/predict`.
- `tests/`: tests pytest.
- `docs/model_card.md`: usage prévu, limites, méthodologie et validations avant production.

## 10. Plan de mise en production

Remplacer le jeu de cas par des données récentes, autorisées et représentatives; vérifier les définitions des achats et la date de référence; faire une validation temporelle; faire approuver les profils par les équipes métier; et mesurer l'uplift par expérimentation randomisée. Ajouter ensuite monitoring de dérive, journaux, tests CI, versionnage et contrôles de confidentialité. Le présent dépôt est une démonstration professionnelle, pas un système marketing validé.
