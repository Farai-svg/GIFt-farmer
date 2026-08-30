"""
GIFt Farmer Crop Recommender -- Streamlit app (v3: sidebar layout)
====================================================================
Interactive front-end for the NACAL k-means clustering pipeline
(gift_nacal_clustering.py). A user enters a farmer's characteristics in
the SIDEBAR, the app assigns them to the nearest household cluster, and
the main panel shows which crop that cluster performs best at.

Run with:  streamlit run app.py

Required files in the same folder (all produced by gift_nacal_clustering.py):
  - kmeans_model.joblib
  - encoding.joblib          (numeric scaler + categorical category lists/weights)
  - cluster_profile_summary.csv
  - cluster_crop_yield_full_table.csv

Theme: colors are set in .streamlit/config.toml (not in this file) --
edit that file to change the color palette without touching any code.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GIFt Farmer Crop Recommender", page_icon="🌾", layout="wide")

# Scale down font sizes in the main content panel only (sidebar untouched),
# so the results view reads as "zoomed out" / more compact. Tweak the
# rem values below to taste -- everything here is scoped to
# [data-testid="stMain"], which is the main panel Streamlit renders into;
# [data-testid="stSidebar"] is deliberately left alone.
st.markdown(
    """
    <style>
    [data-testid="stMain"] {
        font-size: 0.85rem;
    }
    [data-testid="stMain"] h1 { font-size: 1.6rem; }
    [data-testid="stMain"] h2 { font-size: 1.25rem; }
    [data-testid="stMain"] h3 { font-size: 1.05rem; }
    [data-testid="stMain"] p, [data-testid="stMain"] li { font-size: 0.85rem; }
    [data-testid="stMain"] [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMain"] [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    [data-testid="stMain"] [data-testid="stDataFrame"] * { font-size: 0.8rem; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    kmeans_model = joblib.load("kmeans_model.joblib")
    encoding = joblib.load("encoding.joblib")
    cluster_profile = pd.read_csv("cluster_profile_summary.csv").set_index("km_cluster")
    crop_stats = pd.read_csv("cluster_crop_yield_full_table.csv")
    return kmeans_model, encoding, cluster_profile, crop_stats


kmeans_model, encoding, cluster_profile, crop_stats = load_artifacts()
NUMERIC_COLS = encoding["numeric_cols"]
CAT_META = encoding["categorical_meta"]  # {"marital": {...}, "region": {...}, "tenure": {...}, "district": {...}}

FEATURE_LABELS = {
    "total_land_ha": "Total land size (ha)",
    "n_plots": "Number of plots/gardens",
    "hh_has_irrigated_plot": "Has an irrigated plot",
    "has_farm_equipment": "Owns farm equipment",
    "has_extension_access": "Received extension/advisory services",
    "head_age": "Farmer's age",
    "head_is_male": "Farmer is male",
    "hh_size": "Household size",
    "is_urban": "Lives in an urban area",
}


def transform_new_farmer(new_farmer: dict) -> np.ndarray:
    """Mirrors gift_nacal_clustering.transform_new_farmer exactly, so a
    farmer entered here lands in the same feature space the model was
    trained on."""
    X_num_df = pd.DataFrame([new_farmer])[NUMERIC_COLS]
    X_numeric = encoding["scaler"].transform(X_num_df)

    cat_blocks = []
    for name, meta in CAT_META.items():
        cats = meta["categories"]
        val = new_farmer.get(name, "Unknown")
        if val not in cats:
            val = "Unknown" if "Unknown" in cats else cats[0]
        row = np.array([[1.0 if c == val else 0.0 for c in cats]]) * meta["weight"]
        cat_blocks.append(row)

    return np.hstack([X_numeric] + cat_blocks)


st.title("🌾 GIFt Farmer Crop Recommender")
st.caption(
    "Enter a new farmer's characteristics in the sidebar to find their closest "
    "match among household clusters derived from the NACAL agricultural survey, "
    "and see which crop that cluster performs best at."
)

# ---------------------------------------------------------------------
# SIDEBAR: all farmer inputs live here now
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("🧑‍🌾 Farmer characteristics")

    min_obs = st.slider(
        "Minimum plots to trust a crop estimate",
        min_value=5, max_value=100, value=30, step=5,
        help="Crops with fewer observed plots than this in the matched cluster are excluded from the recommendation.",
    )

    st.subheader("Farm")
    total_land_ha = st.number_input(
        FEATURE_LABELS["total_land_ha"], min_value=0.01, max_value=20.0, value=1.0, step=0.1
    )
    n_plots = st.number_input(
        FEATURE_LABELS["n_plots"], min_value=1, max_value=40, value=3, step=1
    )
    hh_has_irrigated_plot = st.radio(
        FEATURE_LABELS["hh_has_irrigated_plot"], ["No", "Yes"], horizontal=True
    ) == "Yes"
    has_farm_equipment = st.radio(
        FEATURE_LABELS["has_farm_equipment"], ["No", "Yes"], horizontal=True, index=1
    ) == "Yes"
    has_extension_access = st.radio(
        FEATURE_LABELS["has_extension_access"], ["No", "Yes"], horizontal=True
    ) == "Yes"
    land_tenure = st.selectbox(
        "Land tenure", [c for c in CAT_META["tenure"]["categories"] if c != "Unknown"]
    )

    st.subheader("Farmer & household")
    head_age = st.slider(FEATURE_LABELS["head_age"], min_value=15, max_value=100, value=40)
    head_is_male = st.radio("Farmer's sex", ["Female", "Male"], horizontal=True) == "Male"
    hh_size = st.number_input(
        FEATURE_LABELS["hh_size"], min_value=1, max_value=25, value=5, step=1
    )
    marital_status = st.selectbox(
        "Marital status", [c for c in CAT_META["marital"]["categories"] if c != "Unknown"]
    )
    is_urban = st.radio(
        FEATURE_LABELS["is_urban"], ["No (rural)", "Yes (urban)"], horizontal=True
    ) == "Yes (urban)"
    region = st.selectbox("Region", CAT_META["region"]["categories"])
    district_options = [c for c in CAT_META["district"]["categories"] if c != "Unknown"]
    district = st.selectbox("District", sorted(district_options))

    submitted = st.button("🔎 Find cluster & recommend crop", type="primary", use_container_width=True)

# ---------------------------------------------------------------------
# MAIN PANEL: tabs for results / explore
# ---------------------------------------------------------------------

tab_recommend, tab_explore = st.tabs(["🌱 Recommendation", "🔍 Explore clusters & crops"])

with tab_recommend:
    with st.expander("ℹ️ How this works / data caveats", expanded=False):
        n_district = len(CAT_META["district"]["categories"])
        st.markdown(
            f"""
- **Clustering**: households are grouped by k-means on farm size, plot count,
  irrigation, equipment, extension access, farmer demographics (age, sex,
  household size), urban/rural residence, marital status, region, district
  (all {n_district} of them), and land tenure.
- **Mixed data types**: numeric/binary features are standardized (z-scores);
  categorical variables (marital status, region, district, tenure) are
  one-hot encoded, with each variable's block scaled by 1/√(number of
  categories) so that District — with {n_district} categories — doesn't
  mechanically dominate the distance calculation just by having more
  columns than everything else combined.
- **Crop performance** is measured as a **relative yield index**: a cluster's
  median yield for a crop, divided by that crop's median yield across *all*
  households. An index of 2.0 means this cluster's farmers get roughly double
  the typical yield for that crop; below 1.0 means below average.
- **Data quality caveat**: a few crops (tobacco in particular) show
  implausibly high absolute yields versus published agronomic benchmarks,
  suggesting a units issue in the underlying survey field for at least some
  records. Absolute yield numbers should be treated as directional, not
  exact, until checked against the original questionnaire.
- Recommendations are restricted to crops with a minimum number of observed
  plots in a cluster (adjustable in the sidebar), so recommendations aren't
  driven by a handful of unusual records.
            """
        )

    if not submitted:
        st.info("👈 Fill in the farmer's characteristics in the sidebar, then click "
                 "**Find cluster & recommend crop**.")
    else:
        new_farmer = {
            "total_land_ha": total_land_ha,
            "n_plots": n_plots,
            "hh_has_irrigated_plot": float(hh_has_irrigated_plot),
            "has_farm_equipment": float(has_farm_equipment),
            "has_extension_access": float(has_extension_access),
            "head_age": head_age,
            "head_is_male": float(head_is_male),
            "hh_size": hh_size,
            "is_urban": float(is_urban),
            "marital": marital_status,
            "region": region,
            "tenure": land_tenure,
            "district": district,
        }
        X_new = transform_new_farmer(new_farmer)
        assigned_cluster = int(kmeans_model.predict(X_new)[0])

        st.success(f"This farmer matches **Cluster {assigned_cluster}**")

        candidates = crop_stats[
            (crop_stats["km_cluster"] == assigned_cluster) & (crop_stats["n_obs"] >= min_obs)
        ].sort_values("relative_yield_index", ascending=False)

        if candidates.empty:
            st.warning(
                "No crop in this cluster meets the minimum-observations threshold. "
                "Try lowering the threshold in the sidebar."
            )
        else:
            top = candidates.iloc[0]
            st.markdown(f"**Recommended crop:** {top['crop_code']}")
            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Relative yield index", f"{top['relative_yield_index']:.2f}x")
            rcol2.metric("Median yield (kg/ha)", f"{top['median_yield_kg_per_ha']:,.0f}")

            st.markdown("**Other strong candidate crops for this cluster:**")
            display_df = candidates.head(6)[
                ["crop_code", "relative_yield_index", "median_yield_kg_per_ha", "n_obs"]
            ].rename(columns={
                "crop_code": "Crop",
                "relative_yield_index": "Relative yield index",
                "median_yield_kg_per_ha": "Median yield (kg/ha)",
                "n_obs": "Plots observed",
            })
            st.dataframe(display_df.set_index("Crop").style.format({
                "Relative yield index": "{:.2f}x",
                "Median yield (kg/ha)": "{:,.0f}",
            }), use_container_width=True)

            st.bar_chart(display_df.set_index("Crop")["Relative yield index"])

        st.markdown("**How this farmer compares to the cluster's typical profile:**")
        profile_row = cluster_profile.loc[assigned_cluster]
        compare_df = pd.DataFrame({
            "This farmer": pd.Series({k: v for k, v in new_farmer.items() if k in NUMERIC_COLS}),
            "Cluster average": profile_row[NUMERIC_COLS],
        })
        compare_df.index = [FEATURE_LABELS.get(i, i) for i in compare_df.index]
        st.dataframe(compare_df.round(2), use_container_width=True)

        st.markdown("**Cluster's most common categories** (for context):")
        cat_cols = st.columns(4)
        cat_cols[0].metric("Marital status", profile_row.get("top_marital_status", "—"))
        cat_cols[1].metric("Region", profile_row.get("top_region", "—"))
        cat_cols[2].metric("District", profile_row.get("top_district", "—"))
        cat_cols[3].metric("Land tenure", profile_row.get("top_land_tenure", "—"))

# ---------------------------------------------------------------------
# TAB 2: Explore clusters & crops
# ---------------------------------------------------------------------

with tab_explore:
    st.subheader("Browse all clusters")
    display_profile = cluster_profile.rename(columns=FEATURE_LABELS)
    st.dataframe(display_profile.round(2), use_container_width=True)

    st.subheader("Crop performance by cluster")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        cluster_filter = st.multiselect(
            "Filter by cluster", sorted(crop_stats["km_cluster"].unique()),
            default=sorted(crop_stats["km_cluster"].unique()),
        )
    with fcol2:
        crop_filter = st.multiselect(
            "Filter by crop", sorted(crop_stats["crop_code"].unique())
        )
    with fcol3:
        min_obs_explore = st.slider("Min plots observed", 1, 200, 30, key="explore_min_obs")

    filtered = crop_stats[
        crop_stats["km_cluster"].isin(cluster_filter) & (crop_stats["n_obs"] >= min_obs_explore)
    ]
    if crop_filter:
        filtered = filtered[filtered["crop_code"].isin(crop_filter)]

    filtered = filtered.sort_values(["km_cluster", "relative_yield_index"], ascending=[True, False])
    st.dataframe(
        filtered[["km_cluster", "crop_code", "relative_yield_index",
                  "median_yield_kg_per_ha", "avg_yield_kg_per_ha", "n_obs"]].rename(columns={
            "km_cluster": "Cluster", "crop_code": "Crop",
            "relative_yield_index": "Relative yield index",
            "median_yield_kg_per_ha": "Median yield (kg/ha)",
            "avg_yield_kg_per_ha": "Mean yield (kg/ha)",
            "n_obs": "Plots observed",
        }),
        use_container_width=True,
        height=450,
    )
