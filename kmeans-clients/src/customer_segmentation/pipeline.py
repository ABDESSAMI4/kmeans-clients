"""Reproducible RFM customer segmentation, validation and benchmark utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "marketing_campaign.csv"
MODEL_PATH = ROOT / "models" / "kmeans_rfm_pipeline.pkl"
META_PATH = ROOT / "models" / "kmeans_rfm_meta.json"
SEEDS = (42, 1, 123, 2024, 999)
K_RANGE = range(2, 10)
N_CLUSTERS_BUSINESS = 3
BOOTSTRAP_RUNS = 10
BOOTSTRAP_FRACTION = 0.80
SILHOUETTE_SAMPLE_SIZE = 1200

# These raw variables are inputs to RFMFeatureBuilder. Campaign outcomes are
# intentionally absent: Response and AcceptedCmp* are never model inputs.
RAW_NUMERIC_COLUMNS = [
    "Income",
    "Recency",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
RAW_COLUMNS = RAW_NUMERIC_COLUMNS + ["Dt_Customer"]
RFM_FEATURES = ["Recency", "Frequency", "Monetary", "Tenure", "Income"]
LOG_FEATURES = ["Frequency", "Monetary", "Income"]
FREQUENCY_COLUMNS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]
MONETARY_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
CHANNEL_COLUMNS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
REFERENCE_YEAR = 2014
logger = logging.getLogger("customer_segmentation")


class RFMFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build Recency, Frequency, Monetary, Tenure and Income from raw columns."""

    def __init__(self, reference_year: int = REFERENCE_YEAR):
        self.reference_year = reference_year

    def _raw_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=RAW_COLUMNS)
        missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Colonnes d'entrée absentes : {', '.join(missing)}")
        numeric = frame.loc[:, RAW_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        invalid = frame.loc[:, RAW_NUMERIC_COLUMNS].notna() & numeric.isna()
        if invalid.any().any():
            bad = invalid.columns[invalid.any()].tolist()
            raise ValueError(f"Valeurs non numériques dans : {', '.join(bad)}")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("Les colonnes numériques ne doivent pas contenir de valeurs infinies.")
        non_negative = [c for c in RAW_NUMERIC_COLUMNS if c != "Income"]
        if (numeric[non_negative] < 0).any().any() or (numeric["Income"].dropna() < 0).any():
            raise ValueError("Les variables RFM et Income doivent être positives ou nulles.")
        dates = pd.to_datetime(frame["Dt_Customer"], errors="coerce", dayfirst=True, format="mixed")
        invalid_dates = frame["Dt_Customer"].notna() & dates.isna()
        if invalid_dates.any() or dates.isna().any():
            raise ValueError("Dt_Customer contient une date absente ou illisible.")
        clean = numeric.copy()
        clean["Dt_Customer"] = dates
        return clean

    def fit(self, X: pd.DataFrame, y: Any = None) -> RFMFeatureBuilder:
        self._raw_frame(X)
        self.feature_names_in_ = np.asarray(RAW_COLUMNS, dtype=object)
        self.n_features_in_ = len(RAW_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = self._raw_frame(X)
        result = pd.DataFrame(index=raw.index)
        result["Recency"] = raw["Recency"]
        result["Frequency"] = raw[FREQUENCY_COLUMNS].sum(axis=1, min_count=len(FREQUENCY_COLUMNS))
        result["Monetary"] = raw[MONETARY_COLUMNS].sum(axis=1, min_count=len(MONETARY_COLUMNS))
        # Required convention from the dataset brief; the collection reference year 2014
        # remains an assumption and must be replaced when a verified reference date exists.
        result["Tenure"] = self.reference_year - raw["Dt_Customer"].dt.year
        result["Income"] = raw["Income"]
        if (result["Tenure"] < 0).any():
            raise ValueError("Dt_Customer contient une année postérieure à l'année de référence.")
        return result.loc[:, RFM_FEATURES]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(RFM_FEATURES, dtype=object)


class CustomerPreprocessor(BaseEstimator, TransformerMixin):
    """Learn median imputation, 99th percentile clipping and optional log1p."""

    def __init__(
        self,
        features: tuple[str, ...] | list[str] = tuple(RFM_FEATURES),
        log_features: tuple[str, ...] | list[str] = tuple(LOG_FEATURES),
        cap_quantile: float = 0.99,
    ):
        self.features = features
        self.log_features = log_features
        self.cap_quantile = cap_quantile

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.features)
        missing = sorted(set(self.features) - set(frame.columns))
        if missing:
            raise ValueError(f"Variables dérivées absentes : {', '.join(missing)}")
        raw = frame.loc[:, self.features]
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        invalid = raw.notna() & numeric.isna()
        if invalid.any().any():
            bad = invalid.columns[invalid.any()].tolist()
            raise ValueError(f"Valeurs non numériques dans : {', '.join(bad)}")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("Les variables RFM ne doivent pas contenir de valeurs infinies.")
        if (numeric.drop(columns=["Income"], errors="ignore") < 0).any().any():
            raise ValueError("Les variables RFM doivent être positives ou nulles.")
        if (numeric["Income"].dropna() < 0).any():
            raise ValueError("Income doit être positif ou nul.")
        return numeric.astype(float)

    def fit(self, X: pd.DataFrame, y: Any = None) -> CustomerPreprocessor:
        if not 0 < self.cap_quantile <= 1:
            raise ValueError("cap_quantile doit être dans ]0, 1].")
        frame = self._frame(X)
        self.medians_ = frame.median().fillna(0.0)
        filled = frame.fillna(self.medians_)
        self.upper_bounds_ = filled.quantile(self.cap_quantile).fillna(0.0)
        self.n_features_in_ = len(self.features)
        self.feature_names_in_ = np.asarray(self.features, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "medians_"):
            raise RuntimeError("Le préprocesseur doit être ajusté avant transform.")
        frame = self._frame(X).fillna(self.medians_)
        frame = frame.clip(upper=self.upper_bounds_, axis="columns")
        for column in self.log_features:
            frame[column] = np.log1p(frame[column].clip(lower=0))
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Le prétraitement a produit une valeur non finie.")
        return values

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.features, dtype=object)


def make_pipeline(n_clusters: int = 3, seed: int = 42, use_log: bool = True) -> Pipeline:
    """Return the full raw-input-to-cluster pipeline."""
    logs = tuple(LOG_FEATURES) if use_log else tuple()
    return Pipeline(
        [
            ("rfm", RFMFeatureBuilder()),
            ("features", CustomerPreprocessor(log_features=logs)),
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)),
        ]
    )


def charger_source(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {path}")
    data = pd.read_csv(path)
    required = set(RAW_COLUMNS + ["ID", "Year_Birth"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes : {', '.join(missing)}")
    if data.empty:
        raise ValueError("Le fichier source ne contient aucune ligne.")
    return data


def audit_source(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "lignes_initiales": int(len(df)),
        "colonnes_initiales": int(df.shape[1]),
        "doublons_exacts": int(df.duplicated().sum()),
        "ids_dupliques": int(df["ID"].duplicated().sum()),
        "valeurs_manquantes_income": int(df["Income"].isna().sum()),
        "dates_client_invalides": int(pd.to_datetime(
            df["Dt_Customer"], errors="coerce", dayfirst=True, format="mixed"
        ).isna().sum()),
    }


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(X: np.ndarray, labels: np.ndarray, seed: int, sample_size: int) -> dict[str, float]:
    valid = np.asarray(labels) != -1
    x_valid = X[valid]
    y_valid = np.asarray(labels)[valid]
    unique = np.unique(y_valid)
    if len(unique) < 2 or len(unique) >= len(y_valid):
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(
            x_valid, y_valid, sample_size=min(sample_size, len(y_valid)), random_state=seed
        )),
        "davies_bouldin": float(davies_bouldin_score(x_valid, y_valid)),
        "calinski_harabasz": float(calinski_harabasz_score(x_valid, y_valid)),
    }


def _engineered_scaled(raw: pd.DataFrame, use_log: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    builder = RFMFeatureBuilder().fit(raw)
    engineered = builder.transform(raw)
    processor = CustomerPreprocessor(log_features=tuple(LOG_FEATURES) if use_log else tuple()).fit(engineered)
    scaled = StandardScaler().fit_transform(processor.transform(engineered))
    return engineered, scaled


def evaluate_kmeans(raw: pd.DataFrame, k_min: int = 2, k_max: int = 9, seed: int = 42) -> pd.DataFrame:
    engineered, scaled = _engineered_scaled(raw, use_log=True)
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(scaled)
        rows.append({
            "k": k,
            "inertie": float(model.inertia_),
            **_metrics(scaled, model.labels_, seed, SILHOUETTE_SAMPLE_SIZE),
        })
    return pd.DataFrame(rows)


def stability_validation(
    raw: pd.DataFrame,
    k_values: range | list[int] = K_RANGE,
    seeds: tuple[int, ...] = SEEDS,
    bootstrap_runs: int = BOOTSTRAP_RUNS,
    bootstrap_fraction: float = BOOTSTRAP_FRACTION,
) -> pd.DataFrame:
    """Refit feature engineering, imputer/cap, scaler and KMeans in each bootstrap.

    Each bootstrap is sampled with replacement at 80% of n. ARI compares predictions
    from the full-data reference pipeline and the bootstrap pipeline on the full original
    dataset, so both partitions cover the same observations.
    """
    if bootstrap_runs < 2 or not 0 < bootstrap_fraction <= 1:
        raise ValueError("Utiliser au moins 2 réplications et une fraction dans ]0, 1].")
    raw = raw.reset_index(drop=True)
    n = len(raw)
    size = max(2, int(round(bootstrap_fraction * n)))
    rows = []
    for seed in seeds:
        for k in k_values:
            reference = make_pipeline(n_clusters=k, seed=seed, use_log=True).fit(raw)
            reference_labels = reference.predict(raw)
            rng = np.random.default_rng(seed * 1_000_003 + k)
            values = []
            for repetition in range(bootstrap_runs):
                indices = rng.integers(0, n, size=size)
                replica = make_pipeline(
                    n_clusters=k, seed=seed + repetition + 1, use_log=True
                ).fit(raw.iloc[indices].reset_index(drop=True))
                score = float(adjusted_rand_score(reference_labels, replica.predict(raw)))
                values.append(score)
            row: dict[str, Any] = {
                "k": int(k), "seed": int(seed), "bootstrap_runs": int(bootstrap_runs),
                "bootstrap_fraction": float(bootstrap_fraction),
                "ari_mean": float(np.mean(values)), "ari_std": float(np.std(values, ddof=1)),
            }
            row.update({f"ari_bootstrap_{i + 1:02d}": value for i, value in enumerate(values)})
            rows.append(row)
            logger.info("stabilité: seed=%s K=%s ARI=%.4f +/- %.4f", seed, k, row["ari_mean"], row["ari_std"])
    return _add_stability_aggregates(pd.DataFrame(rows))


def _add_stability_aggregates(stability: pd.DataFrame) -> pd.DataFrame:
    """Add pooled per-K mean/std over all seed x bootstrap ARI values."""
    if stability.empty:
        return stability
    bootstrap_cols = sorted(c for c in stability.columns if c.startswith("ari_bootstrap_"))
    means: dict[int, float] = {}
    stds: dict[int, float] = {}
    for k, group in stability.groupby("k"):
        values = group[bootstrap_cols].to_numpy(dtype=float).ravel()
        means[int(k)] = float(np.mean(values))
        stds[int(k)] = float(np.std(values, ddof=1))
    stability = stability.copy()
    stability["ari_overall_mean"] = stability["k"].map(means)
    stability["ari_overall_std"] = stability["k"].map(stds)
    return stability


def benchmark_models(raw: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the requested five RFM clusterers using a common K=3 benchmark."""
    _, x_log = _engineered_scaled(raw, use_log=True)
    _, x_no_log = _engineered_scaled(raw, use_log=False)
    models: list[tuple[str, str, np.ndarray, Any]] = [
        ("KMeans sans log1p", "RFM standardisé sans log1p", x_no_log,
         KMeans(n_clusters=3, random_state=seed, n_init=20)),
        ("KMeans avec log1p", "RFM avec log1p + standardisation", x_log,
         KMeans(n_clusters=3, random_state=seed, n_init=20)),
        ("GaussianMixture", "RFM avec log1p + standardisation", x_log,
         GaussianMixture(n_components=3, random_state=seed, n_init=5, reg_covar=1e-5)),
        ("Agglomerative Ward", "RFM avec log1p + standardisation", x_log,
         AgglomerativeClustering(n_clusters=3, linkage="ward")),
        ("HDBSCAN", "RFM avec log1p + standardisation", x_log,
         HDBSCAN(min_cluster_size=max(20, int(0.02 * len(raw))), min_samples=10, copy=False)),
    ]
    rows = []
    for name, preprocessing, matrix, estimator in models:
        try:
            labels = estimator.fit_predict(matrix)
            counts = pd.Series(labels).value_counts()
            n_clusters = int((counts.index != -1).sum())
            noise = float(np.mean(np.asarray(labels) == -1))
            metrics = _metrics(matrix, labels, seed, SILHOUETTE_SAMPLE_SIZE)
            status = "ok" if np.isfinite(metrics["silhouette"]) else "non calculable: moins de deux clusters non-bruit"
            rows.append({
                "modele": name, "pretraitement": preprocessing, "n_clusters": n_clusters,
                "noise_fraction": noise, **metrics, "status": status,
            })
        except Exception as exc:  # Keep the benchmark complete if one method cannot fit.
            logger.exception("Échec du benchmark %s", name)
            rows.append({
                "modele": name, "pretraitement": preprocessing, "n_clusters": np.nan,
                "noise_fraction": np.nan, "silhouette": np.nan,
                "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                "status": f"échec: {type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows), pd.DataFrame({"seed": [seed], "sample_size": [min(SILHOUETTE_SAMPLE_SIZE, len(raw))]})


def name_business_clusters(rfm: pd.DataFrame, clusters: pd.Series) -> dict[int, str]:
    profile = rfm.assign(cluster=clusters.to_numpy()).groupby("cluster")[RFM_FEATURES].mean()
    # Relative ranks make the label assignment robust to scale differences.
    score = (
        0.25 * profile["Recency"].rank(pct=True, ascending=False)
        + 0.35 * profile["Frequency"].rank(pct=True)
        + 0.40 * profile["Monetary"].rank(pct=True)
    )
    ids = score.sort_values(ascending=False).index.tolist()
    if len(ids) == 3:
        champion = ids[0]
        other_ids = [cluster for cluster in ids if cluster != champion]
        # The stale-purchase, high-value group is more actionable as a win-back
        # segment than calling every middle-ranked group "Occasionnel".
        at_risk = profile.loc[other_ids, "Recency"].idxmax()
        small_buyers = next(cluster for cluster in other_ids if cluster != at_risk)
        return {
            int(champion): "Champions",
            int(at_risk): "À réactiver",
            int(small_buyers): "Petits acheteurs",
        }
    return {int(cluster): f"Segment {cluster}" for cluster in profile.index}


def build_recommendations(
    labels: dict[int, str], profiles: pd.DataFrame
) -> dict[str, dict[str, str]]:
    """Return concrete suggestions; KPIs are measures to monitor, not causal promises."""
    output = {}
    for cluster, name in labels.items():
        profile = profiles.loc[cluster]
        channel_scores = profile[CHANNEL_COLUMNS].astype(float)
        channel = {
            "NumWebPurchases": "web / email",
            "NumCatalogPurchases": "catalogue / courrier",
            "NumStorePurchases": "magasin",
        }[channel_scores.idxmax()]
        if name == "Champions":
            action, offer, kpi = (
                "Fidéliser et développer la valeur à long terme sans sur-solliciter.",
                "Accès anticipé, service premium ou bundle complémentaire à marge maîtrisée.",
                "Taux de réachat, marge par client contacté et désabonnements.",
            )
            expected = "Faire progresser le réachat et la marge nette versus un groupe témoin, sans hausse des désabonnements."
        elif name in {"Dormants", "À réactiver"}:
            action, offer, kpi = (
                "Lancer une relance win-back avec fréquence limitée et groupe témoin.",
                "Offre de réactivation à durée courte, avec seuil de panier et coût plafonné.",
                "Réactivation incrémentale, marge nette, coût par client réactivé et plaintes.",
            )
            expected = "Obtenir une réactivation incrémentale et une marge nette positive versus le groupe témoin."
        else:
            action, offer, kpi = (
                "Développer la fréquence par recommandations de produits complémentaires.",
                "Bundle personnalisé ou livraison offerte à partir d’un seuil pertinent.",
                "Conversion, fréquence à 90 jours, panier moyen et marge nette.",
            )
            expected = "Augmenter fréquence et panier net à 90 jours versus le baseline du segment."
        output[str(cluster)] = {
            "segment": name, "action": action, "canal": channel,
            "offre": offer, "kpi_attendu": expected, "kpi_a_suivre": kpi,
            "precaution": "Hypothèse d'action à tester; une association descriptive ne prouve pas un uplift causal.",
        }
    return output


def _save_profile_artifacts(data: pd.DataFrame, rfm: pd.DataFrame, cluster_labels: pd.Series,
                            labels: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    profile_source = data.copy()
    profile_source[RFM_FEATURES] = rfm[RFM_FEATURES].to_numpy()
    profile_source["cluster"] = cluster_labels.to_numpy()
    profile_vars = list(dict.fromkeys(RFM_FEATURES + FREQUENCY_COLUMNS + MONETARY_COLUMNS))
    profile_rows = []
    for cluster, group in profile_source.groupby("cluster"):
        for feature in profile_vars:
            profile_rows.append({
                "cluster": int(cluster), "segment": labels[int(cluster)], "variable": feature,
                "mean": float(group[feature].mean()), "median": float(group[feature].median()),
            })
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(outputs / "profils_rfm_k3.csv", index=False)

    # Response is used only after fitting, to describe observed response rates.
    response = data.assign(cluster=cluster_labels.to_numpy()).groupby("cluster")["Response"].mean()
    rfm_profile = rfm.assign(cluster=cluster_labels.to_numpy()).groupby("cluster")[RFM_FEATURES].mean()
    rfm_profile["clients"] = cluster_labels.value_counts().sort_index()
    rfm_profile["response_observee_descriptive"] = response
    rfm_profile["segment"] = [labels[int(i)] for i in rfm_profile.index]
    return profiles, rfm_profile


def plot_benchmark(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specs = [
        ("silhouette", "Silhouette (plus élevé préférable)", "#168C8C"),
        ("davies_bouldin", "Davies-Bouldin (plus bas préférable)", "#F0B44D"),
        ("calinski_harabasz", "Calinski-Harabasz (plus élevé préférable)", "#547AA5"),
    ]
    for ax, (column, title, color) in zip(axes, specs):
        valid = metrics.dropna(subset=[column])
        ax.barh(valid["modele"], valid[column], color=color)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Benchmark des algorithmes sur les variables RFM (K=3 sauf HDBSCAN)", color="#17324D")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rfm_profiles(rfm: pd.DataFrame, labels: dict[int, str], data: pd.DataFrame,
                      cluster_labels: pd.Series) -> None:
    outputs = ROOT / "outputs"
    # Boxplots in original feature units, with robust axes and outliers hidden for readability.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, RFM_FEATURES):
        ordered = sorted(labels)
        values = [rfm.loc[cluster_labels.to_numpy() == cid, feature].dropna().to_numpy() for cid in ordered]
        ax.boxplot(values, tick_labels=[labels[cid] for cid in ordered], showfliers=False)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=12, labelsize=8)
        ax.grid(axis="y", alpha=.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution des variables RFM - unités d'origine (valeurs extrêmes masquées)")
    fig.tight_layout()
    fig.savefig(outputs / "boxplots_rfm_k3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Radar is normalized across cluster means for visualization only.
    means = rfm.groupby(cluster_labels.to_numpy())[RFM_FEATURES].mean()
    scaled = (means - means.min()) / (means.max() - means.min()).replace(0, 1)
    scaled["Recency"] = 1 - scaled["Recency"]  # a recent purchase is the favorable direction
    categories = RFM_FEATURES
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)
    for cluster, row in scaled.iterrows():
        values = row[categories].tolist() + [row[categories[0]]]
        ax.plot(angles, values, linewidth=2, label=labels[int(cluster)])
        ax.fill(angles, values, alpha=.09)
    ax.set_xticks(angles[:-1], categories)
    ax.set_ylim(0, 1)
    ax.set_title("Radar des moyennes RFM normalisées par variable\n(Recency inversée pour lecture favorable)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
    fig.tight_layout()
    fig.savefig(outputs / "radar_rfm_k3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_rfm_project(path: str | Path = DATA_PATH, run_stability: bool = True) -> dict[str, Any]:
    """Build v2 outputs, diagnostics, benchmark and the K=3 business model."""
    path = Path(path)
    source = charger_source(path)
    audit = audit_source(source)
    data = source.drop_duplicates(subset="ID", keep="first").reset_index(drop=True)
    if len(data) < 20:
        raise ValueError("Il faut au moins 20 clients pour entraîner le pipeline RFM.")
    outputs, models = ROOT / "outputs", ROOT / "models"
    outputs.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    logger.info("Évaluation K-Means RFM sur K=%s..%s", min(K_RANGE), max(K_RANGE))
    k_scores = evaluate_kmeans(data)
    k_scores.to_csv(outputs / "comparaison_k_rfm.csv", index=False)

    stability_path = outputs / "stability_v2.csv"
    if run_stability:
        stability = stability_validation(data)
        stability.to_csv(stability_path, index=False)
    elif stability_path.exists():
        stability = pd.read_csv(stability_path)
    else:
        stability = pd.DataFrame()
    if not stability.empty:
        stability = _add_stability_aggregates(stability)
        stability.to_csv(stability_path, index=False)

    benchmark, _ = benchmark_models(data)
    benchmark.to_csv(outputs / "comparaison_modeles.csv", index=False)
    plot_benchmark(benchmark, outputs / "comparaison_modeles.png")

    pipeline = make_pipeline(N_CLUSTERS_BUSINESS, seed=42, use_log=True).fit(data)
    cluster_ids = pipeline.predict(data)
    rfm = pipeline.named_steps["rfm"].transform(data)
    labels = name_business_clusters(rfm, pd.Series(cluster_ids))
    prof_long, prof_wide = _save_profile_artifacts(data, rfm, pd.Series(cluster_ids), labels)
    plot_rfm_profiles(rfm, labels, data, pd.Series(cluster_ids))

    scored = data.copy()
    scored[RFM_FEATURES] = rfm[RFM_FEATURES].to_numpy()
    scored["cluster"] = cluster_ids
    scored["segment"] = [labels[int(cluster)] for cluster in cluster_ids]
    scored.to_csv(outputs / "clients_segmentes_rfm_k3.csv", index=False)

    # The simulation uses the observed Monetary gap as a scenario assumption only.
    recommendations = build_recommendations(
        labels, data.assign(cluster=cluster_ids).groupby("cluster").mean(numeric_only=True)
    )
    meta = {
        "model_name": "RFM KMeans v2",
        "source": "Customer Personality Analysis (attribution à confirmer)",
        "source_url": "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis",
        "source_sha256": _hash_file(path),
        "n_clients": int(len(data)), "n_features": len(RFM_FEATURES),
        "features": RFM_FEATURES, "raw_columns": RAW_COLUMNS,
        "frequency_definition": "NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases",
        "monetary_definition": "sum of six Mnt* purchase amount columns",
        "tenure_definition": "2014 - year(Dt_Customer); reference year assumed",
        "log_features": LOG_FEATURES, "cap_quantile": 0.99,
        "k": N_CLUSTERS_BUSINESS, "seed": 42,
        "business_labels": {str(k): v for k, v in labels.items()},
        "recommendations": recommendations,
        "k_metrics": k_scores.to_dict(orient="records"),
        "benchmark": benchmark.to_dict(orient="records"),
        "stability_seeds": list(SEEDS), "bootstrap_runs": BOOTSTRAP_RUNS,
        "bootstrap_fraction": BOOTSTRAP_FRACTION,
        "stability_summary": stability.groupby("k")["ari_overall_mean"].first().to_dict() if not stability.empty else {},
        "stability_summary_std": stability.groupby("k")["ari_overall_std"].first().to_dict() if not stability.empty else {},
        "diagnostics": audit,
        "versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
        "response_usage": "Descriptive post-clustering analysis only; never used in model fit.",
        "uplift_warning": "Scenario simulation only; not a causal uplift estimate.",
    }
    joblib.dump(pipeline, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Pipeline RFM K=3 exporté: %s", MODEL_PATH)
    return {"data": scored, "pipeline": pipeline, "metadata": meta,
            "profiles": prof_long, "profile_summary": prof_wide,
            "k_scores": k_scores, "stability": stability, "benchmark": benchmark}


def predict_with_recommendation(raw_client: pd.DataFrame, pipeline: Pipeline | None = None,
                                metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    pipeline = pipeline or joblib.load(MODEL_PATH)
    metadata = metadata or json.loads(META_PATH.read_text(encoding="utf-8"))
    cluster_ids = pipeline.predict(raw_client)
    recommendations = metadata["recommendations"]
    return [
        {"cluster": int(cluster), **recommendations[str(int(cluster))]}
        for cluster in cluster_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîner le pipeline RFM K=3 et ses diagnostics")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--skip-stability", action="store_true", help="Sauter les 400 ajustements bootstrap")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    result = train_rfm_project(args.data, run_stability=not args.skip_stability)
    print("\nBenchmark des modèles:")
    print(result["benchmark"].round(4).to_string(index=False))
    print("\nProfils RFM K=3:")
    print(result["profile_summary"].round(2).to_string())
    if not result["stability"].empty:
        print("\nStabilité bootstrap (ARI moyen/std par K et seed):")
        print(result["stability"][["k", "seed", "ari_mean", "ari_std"]].round(4).to_string(index=False))


if __name__ == "__main__":
    # joblib serializes estimator classes by module path. Importing the stable
    # package entry point prevents artifacts from referring to __main__.
    from customer_segmentation.pipeline import main as package_main

    package_main()
