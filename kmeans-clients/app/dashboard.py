"""Four-page interactive dashboard for the RFM segmentation v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_segmentation.pipeline import (  # noqa: E402
    CHANNEL_COLUMNS,
    MODEL_PATH,
    META_PATH,
    RAW_COLUMNS,
    RFM_FEATURES,
    predict_with_recommendation,
)

st.set_page_config(page_title="Segmentation RFM clients", page_icon=":material/groups:", layout="wide")


@st.cache_data(ttl="15m", max_entries=4)
def load_scored_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "clients_segmentes_rfm_k3.csv")


@st.cache_data(ttl="15m", max_entries=4)
def load_k_metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "comparaison_k_rfm.csv")


@st.cache_data(ttl="15m", max_entries=4)
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "comparaison_modeles.csv")


@st.cache_resource
def load_pipeline():
    import joblib
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="15m", max_entries=4)
def load_metadata() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


try:
    df = load_scored_data()
    k_scores = load_k_metrics()
    benchmark = load_benchmark()
    pipeline = load_pipeline()
    metadata = load_metadata()
except FileNotFoundError:
    st.error(
        "Les artefacts RFM V2 ne sont pas encore générés. Exécute "
        "`python -m customer_segmentation.pipeline` depuis la racine du projet.",
        title="Artefacts manquants",
    )
    st.stop()

labels = {int(key): value for key, value in metadata["business_labels"].items()}
recommendations = {int(key): value for key, value in metadata["recommendations"].items()}
cluster_order = sorted(labels)
color_map = {labels[cid]: px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
             for i, cid in enumerate(cluster_order)}
page = st.sidebar.segmented_control(
    "Navigation", ["KPIs & choix de K", "Profils", "Prédiction", "Simulation"],
    default="KPIs & choix de K", key="page_v2",
)
st.sidebar.divider()
st.sidebar.caption("Modèle RFM K-Means · K=3 · graine 42")
st.sidebar.caption(f"Python {metadata['versions']['python']} · scikit-learn {metadata['versions']['scikit_learn']}")
st.title("Segmentation clients | RFM V2")
st.caption("Exploration descriptive du jeu Customer Personality Analysis. Les décisions marketing doivent être validées par expérimentation.")

if page == "KPIs & choix de K":
    total_spend = df["Monetary"].sum()
    response = df.groupby("segment")["Response"].mean().reindex([labels[i] for i in cluster_order])
    with st.container(horizontal=True):
        st.metric("Clients analysés", f"{len(df):,}".replace(",", " "), border=True)
        st.metric("Segments métier", len(labels), border=True)
        st.metric("Dépense cumulée observée", f"{total_spend:,.0f}".replace(",", " "), border=True)
        stability_k3 = metadata["stability_summary"].get("3")
        st.metric("ARI stabilité K=3", f"{stability_k3:.3f}" if stability_k3 is not None else "N/D", border=True)
    selected_k = st.select_slider("Examiner la valeur de K", options=k_scores["k"].astype(int).tolist(), value=3)
    selected = k_scores.loc[k_scores.k == selected_k].iloc[0]
    inertia_fig = go.Figure(go.Scatter(x=k_scores.k, y=k_scores.inertie, mode="lines+markers", name="Inertie"))
    inertia_fig.add_trace(go.Scatter(x=[selected_k], y=[selected.inertie], mode="markers", name=f"K sélectionné: {selected_k}", marker=dict(size=14, color="#E45756")))
    inertia_fig.update_layout(xaxis_title="Nombre de clusters K", yaxis_title="Inertie", margin=dict(t=25, b=25), showlegend=False)
    silhouette_fig = go.Figure(go.Scatter(x=k_scores.k, y=k_scores.silhouette, mode="lines+markers", name="Silhouette"))
    silhouette_fig.add_trace(go.Scatter(x=[selected_k], y=[selected.silhouette], mode="markers", name=f"K sélectionné: {selected_k}", marker=dict(size=14, color="#E45756")))
    silhouette_fig.update_layout(xaxis_title="Nombre de clusters K", yaxis_title="Silhouette", margin=dict(t=25, b=25), showlegend=False)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Elbow: inertie selon K")
        st.plotly_chart(inertia_fig, width="stretch", alt="Courbe interactive d'inertie avec valeur K sélectionnée")
    with right, st.container(border=True):
        st.subheader("Silhouette selon K")
        st.plotly_chart(silhouette_fig, width="stretch", alt="Courbe interactive de silhouette avec valeur K sélectionnée")
    st.info(f"K={selected_k}: silhouette {selected.silhouette:.3f}; inertie {selected.inertie:,.1f}. Le modèle de démonstration reste entraîné avec K=3 pour tester des profils métiers plus fins.")
    st.subheader("Benchmark des algorithmes")
    available_metric = st.selectbox("Métrique à comparer", ["silhouette", "davies_bouldin", "calinski_harabasz"], format_func=lambda x: {
        "silhouette": "Silhouette (plus haut préférable)",
        "davies_bouldin": "Davies-Bouldin (plus bas préférable)",
        "calinski_harabasz": "Calinski-Harabasz (plus haut préférable)",
    }[x])
    chart = px.bar(benchmark, x="modele", y=available_metric, color="modele", hover_data=["n_clusters", "noise_fraction", "status"], title="Comparaison sur les variables RFM")
    st.plotly_chart(chart, width="stretch", alt="Comparaison des métriques entre algorithmes")
    st.dataframe(benchmark, hide_index=True, width="stretch", alt="Résultats du benchmark des algorithmes")
    st.caption("Les métriques décrivent la géométrie sur cet échantillon; elles ne démontrent ni la validité commerciale ni une performance future.")

elif page == "Profils":
    st.subheader("Radar RFM par cluster")
    st.caption("Les rayons sont normalisés entre les moyennes des segments pour comparer les formes; Recency est inversée. Les valeurs brutes restent dans le tableau.")
    means = df.groupby("cluster")[RFM_FEATURES].mean().reindex(cluster_order)
    normalized = (means - means.min()) / (means.max() - means.min()).replace(0, 1)
    normalized["Recency"] = 1 - normalized["Recency"]
    radar = go.Figure()
    for cid in cluster_order:
        vals = normalized.loc[cid, RFM_FEATURES].tolist()
        radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=RFM_FEATURES + [RFM_FEATURES[0]], fill="toself", name=labels[cid], line_color=color_map[labels[cid]], opacity=.8))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=480, margin=dict(t=35, b=25))
    st.plotly_chart(radar, width="stretch", alt="Profil RFM normalisé comparant les trois segments")

    st.subheader("Profilage en unités d'origine")
    profile_vars = RFM_FEATURES + CHANNEL_COLUMNS
    means_table = df.groupby(["cluster", "segment"])[profile_vars].mean().reset_index()
    medians_table = df.groupby(["cluster", "segment"])[profile_vars].median().reset_index()
    stat = st.segmented_control("Statistique", ["Moyennes", "Médianes"], default="Moyennes", key="profile_stat")
    table = means_table if stat == "Moyennes" else medians_table
    st.dataframe(table.drop(columns="cluster").round(2), hide_index=True, width="stretch", alt="Statistiques RFM et de canaux par segment")
    st.download_button("Télécharger le profilage CSV", table.to_csv(index=False).encode("utf-8-sig"), "profilage_rfm.csv", "text/csv")
    st.subheader("Projection PCA 3D")
    transformed = pipeline[:-1].transform(df[RAW_COLUMNS])
    coords = PCA(n_components=3, random_state=42).fit_transform(transformed)
    pca_frame = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    pca_frame["segment"] = df["segment"].to_numpy()
    fig3d = px.scatter_3d(pca_frame, x="PC1", y="PC2", z="PC3", color="segment", color_discrete_map=color_map, opacity=.65, title="Projection en trois composantes PCA")
    st.plotly_chart(fig3d, width="stretch", alt="Projection PCA 3D des clients colorée par segment")
    st.caption(f"Variance expliquée par les trois composantes: {PCA(n_components=3).fit(transformed).explained_variance_ratio_.sum():.1%}. PCA est une vue de projection et ne remplace pas l'espace RFM du modèle.")
    st.subheader("Boxplots")
    for metric in RFM_FEATURES:
        box = px.box(df, x="segment", y=metric, color="segment", color_discrete_map=color_map, points=False, title=f"{metric} par segment")
        st.plotly_chart(box, width="stretch", alt=f"Distribution de {metric} par segment, valeurs extrêmes masquées")

elif page == "Prédiction":
    st.subheader("Prédire un nouveau client")
    st.write("Charge un CSV avec les colonnes brutes RFM. Les champs de campagne ne sont pas utilisés dans l'affectation.")
    st.caption("Schéma requis: " + ", ".join(RAW_COLUMNS))
    upload = st.file_uploader("CSV clients", type=["csv"], key="rfm_upload")
    if upload is not None:
        try:
            new_data = pd.read_csv(upload)
            missing = sorted(set(RAW_COLUMNS) - set(new_data.columns))
            if missing:
                st.error("Colonnes requises manquantes: " + ", ".join(missing), title="Schéma invalide")
            elif new_data.empty:
                st.warning("Le fichier ne contient aucune ligne.")
            else:
                results = predict_with_recommendation(new_data[RAW_COLUMNS], pipeline, metadata)
                result_frame = pd.DataFrame(results)
                safe_output = pd.DataFrame({"ligne": np.arange(1, len(new_data) + 1), "cluster": result_frame.cluster, "segment": result_frame.segment, "action": result_frame.action, "canal": result_frame.canal, "offre": result_frame.offre, "kpi_attendu": result_frame.kpi_attendu, "kpi_a_suivre": result_frame.kpi_a_suivre})
                st.success(f"{len(safe_output):,} clients affectés.".replace(",", " "), title="Prédiction terminée")
                st.dataframe(safe_output.head(100), hide_index=True, width="stretch", alt="Prédictions et recommandations pour les cent premières lignes")
                st.download_button("Télécharger les prédictions et recommandations", safe_output.to_csv(index=False).encode("utf-8-sig"), "predictions_recommandations.csv", "text/csv")
                st.info("Les recommandations sont des hypothèses métier à tester. Les différences observées entre clusters ne sont pas causales.", title="Interprétation")
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.error(str(exc), title="Fichier impossible à traiter")
    st.subheader("Recommandations par segment")
    for cid in cluster_order:
        rec = recommendations[cid]
        with st.container(border=True):
            st.markdown(f"**{rec['segment']}** · canal suggéré: {rec['canal']}")
            st.write(rec["action"])
            st.write(f"Offre: {rec['offre']}")
            st.write("Résultat attendu à tester: " + rec["kpi_attendu"])
            st.caption("KPI à suivre: " + rec["kpi_a_suivre"])

else:
    st.subheader("Scénario de réactivation des clients à risque")
    st.warning("Simulation de scénario, pas mesure d'uplift causal. Pour estimer un effet causal, utiliser un groupe témoin randomisé.", title="Hypothèse de simulation")
    dormant = df.loc[df.segment == "À réactiver", "Monetary"]
    champion = df.loc[df.segment == "Champions", "Monetary"]
    dormant_count = len(dormant)
    dormant_mean = float(dormant.mean()) if dormant_count else 0.0
    champion_mean = float(champion.mean()) if len(champion) else dormant_mean
    rate = st.slider("Part du segment À réactiver dans le scénario (%)", min_value=0, max_value=100, value=10, step=1)
    converted = dormant_count * rate / 100
    per_client_gap = max(0.0, champion_mean - dormant_mean)
    incremental = converted * per_client_gap
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients à réactiver", f"{dormant_count:,}".replace(",", " "))
    c2.metric("Clients convertis simulés", f"{converted:,.0f}".replace(",", " "))
    c3.metric("Revenu additionnel théorique", f"{incremental:,.0f}".replace(",", " "))
    st.write(f"Hypothèses: chaque client converti atteint le niveau moyen Monetary des Champions ({champion_mean:,.1f}) au lieu de la moyenne du segment À réactiver ({dormant_mean:,.1f}); nombre converti = {rate}% du segment.")
    st.caption("Monetary correspond à la somme des six colonnes Mnt* sur la période du jeu. Le calcul ne tient pas compte des marges, coûts de campagne, délais, attrition ni incertitude. Il ne doit pas être présenté comme un revenu attendu ou un uplift prouvé.")
    scenario = pd.DataFrame({"Taux simulé (%)": range(0, 101, 5)})
    scenario["Clients réactivés hypothétiques"] = dormant_count * scenario["Taux simulé (%)"] / 100
    scenario["Revenu incrémental de scénario"] = scenario["Clients réactivés hypothétiques"] * per_client_gap
    curve = px.line(scenario, x="Taux simulé (%)", y="Revenu incrémental de scénario", title="Sensibilité du scénario à la part réactivée")
    st.plotly_chart(curve, width="stretch", alt="Revenu additionnel de scénario selon différents taux de réactivation")
