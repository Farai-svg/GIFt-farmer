"""
GIFt - Farmer-Level Module: Self-Discovery Approach (k-means clustering)
=========================================================================
v2: adds Marital status, urban/rural residence, region, district, and
land tenure to the clustering feature set (on top of the v1 features:
land, plots, irrigation, equipment, extension access, age, sex, hh size).

Applied to the REAL Malawi NACAL agricultural survey data
(final_datafile__1_.dta, 82,237 plot-level rows / 22,493 households).

MIXED-TYPE FEATURE HANDLING
----------------------------
K-means needs numeric input, so categorical variables are one-hot encoded.
But naively one-hot encoding a high-cardinality variable like District (32
categories) would give it 32 columns versus ~9 for everything else combined
-- in Euclidean distance, more columns means more influence, so District
would end up dominating the clustering and the model would mostly just
rediscover district boundaries instead of meaningful farm/farmer profiles.

Fix: each categorical variable's one-hot block is scaled by 1/sqrt(k),
where k is its number of categories. This keeps every *variable* (not every
column) contributing a comparable, bounded amount to the distance metric,
regardless of how many categories it has. Numeric/binary features continue
to use standard z-score scaling (mean 0, sd 1) as before.

Pipeline:
1. Load data
2. Build one row per household (farmer) with clustering features
   (numeric/binary + weighted categorical dummies)
3. Cluster households (k-means, k chosen via silhouette score)
4. Compute REAL per-plot yield (kg/ha) from harvest_Qua_per_crop / plotareaHA
5. Identify best-performing crop per cluster (relative yield index)
6. Recommend a crop for a NEW farmer by nearest-cluster matching
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_OBS_PER_CROP_CLUSTER = 30  # min plots needed to trust a cluster x crop yield estimate


# ---------------------------------------------------------------------
# STEP 1: Load
# ---------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_stata(path)

    # Decode categorical/labelled columns to plain strings we can work with
    for col in ["Sex_farmer", "acess_irrigation", "Access_Extension_serv",
                "crop_code", "Marital_Status_farmer", "ag_a06", "ag_a01",
                "District", "land_tenure"]:
        df[col] = df[col].astype(str)
        df.loc[df[col].isin(["nan", "NaN", "None"]), col] = np.nan

    return df


# ---------------------------------------------------------------------
# STEP 2: Build one row per household (farmer) for clustering
# ---------------------------------------------------------------------

# Numeric + binary features -- standard z-score scaled together, as in v1
NUMERIC_FEATURE_COLS = [
    "total_land_ha",          # farm size (sum of measured plot areas, ha)
    "n_plots",                 # land fragmentation
    "hh_has_irrigated_plot",   # water access (binary)
    "has_farm_equipment",      # capital endowment (binary)
    "has_extension_access",    # advisory/support access (binary)
    "head_age",                 # farmer characteristics
    "head_is_male",             # farmer characteristics (binary)
    "hh_size",                  # labor endowment
    "is_urban",                 # residence type (binary) -- NEW
]

# Categorical features -- one-hot encoded, each block weighted by 1/sqrt(k)
# so no single high-cardinality variable (District) dominates the distance
# metric. source_col -> household-table column holding the raw category.
CATEGORICAL_SOURCE_COLS = {
    "marital": "head_marital_status",
    "region": "region",
    "tenure": "land_tenure_mode",
    "district": "District",
}


def build_household_table(df: pd.DataFrame) -> pd.DataFrame:
    # --- farm size, fragmentation & dominant tenure from plot-level records ---
    def mode_or_unknown(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else "Unknown"

    plot_agg = df.groupby("HHID").agg(
        total_land_ha=("plotareaHA", "sum"),
        n_plots=("plotareaHA", "size"),
        hh_has_irrigated_plot=("acess_irrigation", lambda s: (s == "YES").any()),
        land_tenure_mode=("land_tenure", mode_or_unknown),
    )

    # --- constant household/EA-level fields (take first) ---
    hh_const = df.groupby("HHID").agg(
        hh_size=("hhsize", "first"),
        has_farm_equipment=("equipment_owner", "first"),
        has_extension_access=("Access_Extension_serv", lambda s: (s == "YES").any()),
        District=("District", "first"),
        region=("ag_a01", "first"),
        is_urban=("ag_a06", lambda s: (s.iloc[0] == "URBAN")),
        hhweight=("hhweight", "first"),
    )

    # --- representative farmer (pid==1, else first available row) ---
    def pick_farmer(g):
        head_rows = g[g["pid"] == 1]
        row = head_rows.iloc[0] if len(head_rows) else g.iloc[0]
        return pd.Series({
            "head_age": row["Age_farmer"],
            "head_is_male": (row["Sex_farmer"] == "Male"),
            "head_marital_status": row["Marital_Status_farmer"] if pd.notna(row["Marital_Status_farmer"]) else "Unknown",
        })

    farmer = df.groupby("HHID").apply(pick_farmer, include_groups=False)

    hh = plot_agg.join(hh_const).join(farmer).reset_index()

    # cast booleans to 0/1
    for col in ["hh_has_irrigated_plot", "has_farm_equipment",
                "has_extension_access", "head_is_male", "is_urban"]:
        hh[col] = hh[col].astype(float)

    hh["head_age"] = pd.to_numeric(hh["head_age"], errors="coerce")

    # --- winsorize total_land_ha at 99th pct ---
    cap = hh["total_land_ha"].quantile(0.99)
    n_capped = (hh["total_land_ha"] > cap).sum()
    hh["total_land_ha"] = hh["total_land_ha"].clip(upper=cap)
    print(f"[clean] Capped {n_capped} households' total_land_ha at the "
          f"99th percentile ({cap:.2f} ha).")

    # --- impute remaining missing numeric values (median) ---
    for col in NUMERIC_FEATURE_COLS:
        if hh[col].isna().any():
            fill_val = hh[col].median()
            n_missing = hh[col].isna().sum()
            hh[col] = hh[col].fillna(fill_val)
            print(f"[impute] {col}: filled {n_missing} missing values with "
                  f"median ({fill_val:.2f})")

    # categorical missing -> explicit "Unknown" bucket (kept as a real
    # category, not silently dropped or guessed)
    for source_col in CATEGORICAL_SOURCE_COLS.values():
        n_missing = hh[source_col].isna().sum()
        if n_missing:
            hh[source_col] = hh[source_col].fillna("Unknown")
            print(f"[impute] {source_col}: {n_missing} missing values set to 'Unknown' category")

    return hh


# ---------------------------------------------------------------------
# STEP 2b: Build the weighted mixed-type design matrix
# ---------------------------------------------------------------------

def fit_transform_features(hh: pd.DataFrame):
    """Fit the numeric scaler + categorical encodings on the full household
    table, and return the combined design matrix X plus a `encoding` dict
    that fully describes how to reproduce this transform for a new farmer
    (saved via joblib so the Streamlit app can reuse it exactly)."""

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(hh[NUMERIC_FEATURE_COLS])

    cat_blocks = []
    cat_meta = {}
    for name, source_col in CATEGORICAL_SOURCE_COLS.items():
        categories = sorted(hh[source_col].unique().tolist())
        k = len(categories)
        weight = 1.0 / np.sqrt(k)
        dummies = pd.Categorical(hh[source_col], categories=categories)
        one_hot = pd.get_dummies(dummies).values.astype(float)  # (n, k)
        cat_blocks.append(one_hot * weight)
        cat_meta[name] = {"source_col": source_col, "categories": categories, "weight": weight}
        print(f"[encode] {name} ({source_col}): {k} categories, block weight = 1/sqrt({k}) = {weight:.3f}")

    X = np.hstack([X_numeric] + cat_blocks)

    encoding = {
        "numeric_cols": NUMERIC_FEATURE_COLS,
        "scaler": scaler,
        "categorical_meta": cat_meta,
    }
    return X, encoding


def transform_new_farmer(new_farmer: dict, encoding: dict) -> np.ndarray:
    """Apply a fitted `encoding` (from fit_transform_features) to a single
    new farmer's raw characteristics dict, returning a (1, n_features) array
    ready for kmeans.predict(). Used identically here and in the Streamlit app."""
    X_num_df = pd.DataFrame([new_farmer])[encoding["numeric_cols"]]
    X_numeric = encoding["scaler"].transform(X_num_df)

    cat_blocks = []
    for name, meta in encoding["categorical_meta"].items():
        cats = meta["categories"]
        val = new_farmer.get(name, "Unknown")
        if val not in cats:
            val = "Unknown" if "Unknown" in cats else cats[0]
        row = np.array([[1.0 if c == val else 0.0 for c in cats]]) * meta["weight"]
        cat_blocks.append(row)

    return np.hstack([X_numeric] + cat_blocks)


# ---------------------------------------------------------------------
# STEP 3: Cluster households
# ---------------------------------------------------------------------

def cluster_households(hh: pd.DataFrame, X: np.ndarray,
                        k_range=range(2, 21), min_cluster_size=100, force_k=None):
    scores = {}
    sizes = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        sizes[k] = pd.Series(labels).value_counts().sort_index().tolist()

    print("\n[cluster] k | silhouette | cluster sizes (smallest -> largest)")
    for k in k_range:
        s = sorted(sizes[k])
        print(f"  k={k:2d} | {scores[k]:.3f} | smallest={s[0]:5d}  largest={s[-1]:5d}")

    if force_k is not None:
        best_k = force_k
        print(f"\n[cluster] Using user-specified k = {best_k}")
    else:
        viable = {k: v for k, v in scores.items() if min(sizes[k]) >= min_cluster_size}
        best_k = max(viable, key=viable.get) if viable else max(scores, key=scores.get)
        print(f"\n[cluster] Selected k = {best_k} "
              f"(best silhouette among k's with min cluster size >= {min_cluster_size})")

    plt.figure(figsize=(7, 4))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.axvline(best_k, color="red", linestyle="--", alpha=0.5, label=f"selected k={best_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score by k (k=2 to 20) -- NACAL, v2 features")
    plt.legend()
    plt.tight_layout()
    plt.savefig("silhouette_scores_nacal.png", dpi=150)
    plt.close()

    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    hh = hh.copy()
    hh["km_cluster"] = final_model.fit_predict(X)

    return hh, final_model, best_k, scores, sizes


# ---------------------------------------------------------------------
# STEP 4: REAL plot-level yield
# ---------------------------------------------------------------------

def build_yield_table(df: pd.DataFrame, hh_clustered: pd.DataFrame) -> pd.DataFrame:
    plot = df[["HHID", "plotid", "gardenid", "crop_code", "plotareaHA",
               "harvest_Qua_per_crop"]].copy()

    plot = plot.merge(hh_clustered[["HHID", "km_cluster"]], on="HHID", how="left")

    plot["yield_kg_per_ha"] = plot["harvest_Qua_per_crop"] / plot["plotareaHA"].replace(0, np.nan)
    plot = plot.dropna(subset=["yield_kg_per_ha"])

    def winsorize(s):
        cap = s.quantile(0.99)
        return s.clip(upper=cap)

    plot["yield_kg_per_ha"] = plot.groupby("crop_code")["yield_kg_per_ha"].transform(winsorize)

    return plot


# ---------------------------------------------------------------------
# STEP 5: Best crop per cluster (REAL yields, relative index)
# ---------------------------------------------------------------------

def best_crop_per_cluster(plot: pd.DataFrame, min_obs=MIN_OBS_PER_CROP_CLUSTER,
                           rank_by="relative_yield_index"):
    pop_median = plot.groupby("crop_code")["yield_kg_per_ha"].median().rename("crop_pop_median")

    stats = (
        plot.groupby(["km_cluster", "crop_code"])["yield_kg_per_ha"]
        .agg(median_yield_kg_per_ha="median", avg_yield_kg_per_ha="mean", n_obs="count")
        .reset_index()
        .merge(pop_median, on="crop_code", how="left")
    )
    stats["relative_yield_index"] = stats["median_yield_kg_per_ha"] / stats["crop_pop_median"]

    reliable = stats[stats["n_obs"] >= min_obs]

    best_crop = (
        reliable.sort_values(rank_by, ascending=False)
        .groupby("km_cluster")
        .first()
        .reset_index()
        .rename(columns={"crop_code": "best_crop"})
    )
    return best_crop, stats


# ---------------------------------------------------------------------
# STEP 6: Recommend a crop for a NEW farmer
# ---------------------------------------------------------------------

def recommend_crop_for_new_farmer(new_farmer: dict, encoding: dict,
                                   model, best_crop_table) -> dict:
    X_new = transform_new_farmer(new_farmer, encoding)
    assigned_cluster = int(model.predict(X_new)[0])

    match = best_crop_table[best_crop_table["km_cluster"] == assigned_cluster]
    if match.empty:
        return {"cluster": assigned_cluster, "recommended_crop": None,
                "expected_yield_kg_per_ha": None}

    return {
        "cluster": assigned_cluster,
        "recommended_crop": match["best_crop"].values[0],
        "expected_yield_kg_per_ha (median)": round(float(match["median_yield_kg_per_ha"].values[0]), 2),
        "relative_yield_index": round(float(match["relative_yield_index"].values[0]), 2),
    }


# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = "/mnt/user-data/uploads/final_datafile__1_.dta"

    print("=" * 70)
    print("STEP 1: Load data")
    print("=" * 70)
    raw = load_data(DATA_PATH)
    print(f"Loaded {len(raw)} plot-level rows, {raw['HHID'].nunique()} households.")

    print("\n" + "=" * 70)
    print("STEP 2: Build household feature table (now with marital status, "
          "urban/rural, region, district, land tenure)")
    print("=" * 70)
    hh = build_household_table(raw)
    X, encoding = fit_transform_features(hh)
    print(f"\nFinal design matrix shape: {X.shape} "
          f"({len(NUMERIC_FEATURE_COLS)} numeric/binary + "
          f"{X.shape[1] - len(NUMERIC_FEATURE_COLS)} weighted categorical dummy columns)")

    print("\n" + "=" * 70)
    print("STEP 3: Cluster households")
    print("=" * 70)
    clustered, kmeans_model, best_k, scores, sizes = cluster_households(hh, X)
    print(f"\nFinal cluster sizes (k={best_k}):")
    print(clustered["km_cluster"].value_counts().sort_index())

    print("\n" + "=" * 70)
    print("STEP 4: Real plot-level yield (kg/ha)")
    print("=" * 70)
    plot_yield = build_yield_table(raw, clustered)
    print(f"{len(plot_yield)} plot-crop records with valid yield.")

    print("\n" + "=" * 70)
    print(f"STEP 5: Best crop per cluster (min {MIN_OBS_PER_CROP_CLUSTER} obs required)")
    print("=" * 70)
    best_crops, full_table = best_crop_per_cluster(plot_yield)
    print(best_crops.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 6: Recommend crop for a NEW farmer")
    print("=" * 70)
    new_farmer_example = {
        "total_land_ha": 1.2, "n_plots": 2, "hh_has_irrigated_plot": 0,
        "has_farm_equipment": 1, "has_extension_access": 0,
        "head_age": 41, "head_is_male": 1, "hh_size": 5, "is_urban": 0,
        "marital": "MONOGAMOUS MARRIED OR NON-FORMAL UNION",
        "region": "Central",
        "tenure": "CUSTOMARY",
        "district": "Lilongwe",
    }
    rec = recommend_crop_for_new_farmer(new_farmer_example, encoding, kmeans_model, best_crops)
    print(f"New farmer profile: {new_farmer_example}")
    print(f"-> Recommendation: {rec}")

    # --- Save outputs ---
    clustered.to_csv("household_clusters_nacal.csv", index=False)
    plot_yield.to_csv("plot_yield_with_cluster.csv", index=False)
    full_table.to_csv("cluster_crop_yield_full_table.csv", index=False)
    best_crops.to_csv("best_crop_per_cluster_nacal.csv", index=False)

    display_cols = NUMERIC_FEATURE_COLS + ["head_marital_status", "region", "District", "land_tenure_mode"]
    cluster_profile = clustered.groupby("km_cluster")[NUMERIC_FEATURE_COLS].mean().round(2)
    cluster_profile["n_households"] = clustered["km_cluster"].value_counts().sort_index()
    # most common category per cluster for each categorical variable (for display)
    for col, label in [("head_marital_status", "top_marital_status"),
                        ("region", "top_region"), ("District", "top_district"),
                        ("land_tenure_mode", "top_land_tenure")]:
        cluster_profile[label] = clustered.groupby("km_cluster")[col].agg(lambda s: s.mode().iloc[0])
    cluster_profile.to_csv("cluster_profile_summary.csv")

    joblib.dump(kmeans_model, "kmeans_model.joblib")
    joblib.dump(encoding, "encoding.joblib")

    print("\nSaved: household_clusters_nacal.csv, plot_yield_with_cluster.csv, "
          "cluster_crop_yield_full_table.csv, best_crop_per_cluster_nacal.csv, "
          "cluster_profile_summary.csv, silhouette_scores_nacal.png, "
          "kmeans_model.joblib, encoding.joblib")
