"""
GIFt - Farmer-Level Module: Self-Discovery Approach (k-means clustering)
=========================================================================
Applied to the REAL Malawi NACAL agricultural survey data
(final_datafile__1_.dta, 82,237 plot-level rows / 22,493 households),
which includes actual crop identities (crop_code) and plot-level harvest
quantities -- unlike the earlier draft, no synthetic crop/yield data is
used anywhere in this version.

Pipeline:
1. Load data; separate household-level vs plot/crop-level information
2. Build one row per household (farmer) with clustering features
3. Cluster households (k-means, k chosen via silhouette score)
4. Compute REAL per-plot yield (kg/ha) from harvest_Qua_per_crop / plotareaHA
5. Identify best-performing crop per cluster (based on real yields)
6. Recommend a crop for a NEW farmer by nearest-cluster matching

NOTE on scope: no crop price data is available in this file, so "best
crop" here is ranked by average realized yield (kg/ha), not economic
value. If price data becomes available it can be merged in the same way
value_per_ha was computed in the earlier draft.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_OBS_PER_CROP_CLUSTER = 30  # min plots needed to trust a cluster x crop yield estimate


# ---------------------------------------------------------------------
# STEP 1: Load
# ---------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_stata(path)

    # Decode categorical/labelled columns to plain values we can work with
    df["Sex_farmer"] = df["Sex_farmer"].astype(str)
    df["acess_irrigation"] = df["acess_irrigation"].astype(str)
    df["Access_Extension_serv"] = df["Access_Extension_serv"].astype(str)
    df["crop_code"] = df["crop_code"].astype(str)

    return df


# ---------------------------------------------------------------------
# STEP 2: Build one row per household (farmer) for clustering
# ---------------------------------------------------------------------

FEATURE_COLS = [
    "total_land_ha",          # farm size (sum of measured plot areas, ha)
    "n_plots",                 # land fragmentation
    "hh_has_irrigated_plot",   # water access (binary)
    "has_farm_equipment",      # capital endowment (binary)
    "has_extension_access",    # advisory/support access (binary)
    "head_age",                 # farmer characteristics
    "head_is_male",             # farmer characteristics (binary)
    "hh_size",                  # labor endowment
]
# Access_loan excluded: 88% missing at household level, too sparse to trust.
# land_tenure / Marital_Status_farmer excluded for this first pass -- both
# available and could be added as dummy variables in a future iteration.


def build_household_table(df: pd.DataFrame) -> pd.DataFrame:
    # --- farm size & fragmentation from plot-level records ---
    plot_agg = df.groupby("HHID").agg(
        total_land_ha=("plotareaHA", "sum"),
        n_plots=("plotareaHA", "size"),
        hh_has_irrigated_plot=("acess_irrigation", lambda s: (s == "YES").any()),
    )

    # --- constant household-level fields (take first) ---
    hh_const = df.groupby("HHID").agg(
        hh_size=("hhsize", "first"),
        has_farm_equipment=("equipment_owner", "first"),
        has_extension_access=("Access_Extension_serv", lambda s: (s == "YES").any()),
        District=("District", "first"),
        hhweight=("hhweight", "first"),
    )

    # --- representative farmer (pid==1, else first available row) ---
    def pick_farmer(g):
        head_rows = g[g["pid"] == 1]
        row = head_rows.iloc[0] if len(head_rows) else g.iloc[0]
        return pd.Series({
            "head_age": row["Age_farmer"],
            "head_is_male": (row["Sex_farmer"] == "Male"),
        })

    farmer = df.groupby("HHID").apply(pick_farmer, include_groups=False)

    hh = plot_agg.join(hh_const).join(farmer).reset_index()

    # cast booleans to 0/1
    for col in ["hh_has_irrigated_plot", "has_farm_equipment",
                "has_extension_access", "head_is_male"]:
        hh[col] = hh[col].astype(float)

    hh["head_age"] = pd.to_numeric(hh["head_age"], errors="coerce")

    # --- winsorize total_land_ha at 99th pct (mirrors earlier script's
    #     handling of extreme land-size outliers) ---
    cap = hh["total_land_ha"].quantile(0.99)
    n_capped = (hh["total_land_ha"] > cap).sum()
    hh["total_land_ha"] = hh["total_land_ha"].clip(upper=cap)
    print(f"[clean] Capped {n_capped} households' total_land_ha at the "
          f"99th percentile ({cap:.2f} ha).")

    # --- impute any remaining missing values (median for continuous cols) ---
    for col in FEATURE_COLS:
        if hh[col].isna().any():
            fill_val = hh[col].median()
            n_missing = hh[col].isna().sum()
            hh[col] = hh[col].fillna(fill_val)
            print(f"[impute] {col}: filled {n_missing} missing values with "
                  f"median ({fill_val:.2f})")

    return hh


# ---------------------------------------------------------------------
# STEP 3: Cluster households
# ---------------------------------------------------------------------

def cluster_households(hh: pd.DataFrame, feature_cols=FEATURE_COLS,
                        k_range=range(2, 21), min_cluster_size=100,
                        force_k=None):
    X = hh[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    sizes = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
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
    plt.title("Silhouette score by k (k=2 to 20) -- NACAL real data")
    plt.legend()
    plt.tight_layout()
    plt.savefig("silhouette_scores_nacal.png", dpi=150)
    plt.close()

    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    hh = hh.copy()
    hh["km_cluster"] = final_model.fit_predict(X_scaled)

    return hh, scaler, final_model, best_k, scores, sizes


# ---------------------------------------------------------------------
# STEP 4: REAL plot-level yield
# ---------------------------------------------------------------------

def build_yield_table(df: pd.DataFrame, hh_clustered: pd.DataFrame) -> pd.DataFrame:
    plot = df[["HHID", "plotid", "gardenid", "crop_code", "plotareaHA",
               "harvest_Qua_per_crop"]].copy()

    plot = plot.merge(hh_clustered[["HHID", "km_cluster"]], on="HHID", how="left")

    # yield in kg per hectare, computed per plot/crop record
    plot["yield_kg_per_ha"] = plot["harvest_Qua_per_crop"] / plot["plotareaHA"].replace(0, np.nan)
    plot = plot.dropna(subset=["yield_kg_per_ha"])

    # winsorize yield WITHIN each crop at the 99th percentile, so one crop's
    # extreme outliers don't distort comparisons with other crops
    def winsorize(s):
        cap = s.quantile(0.99)
        return s.clip(upper=cap)

    plot["yield_kg_per_ha"] = plot.groupby("crop_code")["yield_kg_per_ha"].transform(winsorize)

    return plot


# ---------------------------------------------------------------------
# STEP 5: Best crop per cluster (REAL yields)
# ---------------------------------------------------------------------

def best_crop_per_cluster(plot: pd.DataFrame, min_obs=MIN_OBS_PER_CROP_CLUSTER,
                           rank_by="relative_yield_index"):
    # Rank by MEDIAN, not mean: survey harvest quantities have a heavy right
    # tail from data-entry/unit errors (e.g. some maize plots implying
    # 700,000+ kg/ha, which is agronomically impossible). Per-crop 99th
    # percentile winsorization (done in build_yield_table) tames the very
    # top, but the mean is still pulled up by whatever survives the cap.
    # The median is robust to this and lines up with realistic benchmarks
    # (e.g. median maize yield here ~3,000 kg/ha, consistent with published
    # Malawi smallholder figures).
    #
    # But raw kg/ha is not comparable ACROSS crop types: root/tuber crops
    # (potato, cassava, sweet potato) are bulky and always show far higher
    # kg/ha than grains or legumes, regardless of how well-suited a cluster
    # actually is to growing them. With no price data available to convert
    # to value/ha (as the earlier synthetic draft did), we instead rank by
    # a RELATIVE performance index: this cluster's median yield for a crop,
    # divided by that crop's median yield across ALL households. An index
    # > 1 means the cluster grows that crop better than the typical farmer;
    # this is comparable across crop types and surfaces real differentiation
    # (e.g. "this cluster over-performs at maize") instead of tubers winning
    # by default everywhere.
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

def recommend_crop_for_new_farmer(new_farmer: dict, feature_cols,
                                   scaler, model, best_crop_table) -> dict:
    X_new = pd.DataFrame([new_farmer])[feature_cols]
    X_new_scaled = scaler.transform(X_new)
    assigned_cluster = model.predict(X_new_scaled)[0]

    match = best_crop_table[best_crop_table["km_cluster"] == assigned_cluster]
    if match.empty:
        return {"cluster": int(assigned_cluster), "recommended_crop": None,
                "expected_yield_kg_per_ha": None}

    return {
        "cluster": int(assigned_cluster),
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
    print("STEP 2: Build household feature table")
    print("=" * 70)
    hh = build_household_table(raw)
    print(hh[FEATURE_COLS].describe().round(2).to_string())

    print("\n" + "=" * 70)
    print("STEP 3: Cluster households")
    print("=" * 70)
    clustered, scaler, kmeans_model, best_k, scores, sizes = cluster_households(hh)
    print(f"\nFinal cluster sizes (k={best_k}):")
    print(clustered["km_cluster"].value_counts().sort_index())

    print("\n" + "=" * 70)
    print("STEP 4: Real plot-level yield (kg/ha)")
    print("=" * 70)
    plot_yield = build_yield_table(raw, clustered)
    print(f"{len(plot_yield)} plot-crop records with valid yield "
          f"(from {len(raw)} total plot rows).")

    print("\n" + "=" * 70)
    print("STEP 5: Best crop per cluster (REAL data, "
          f"min {MIN_OBS_PER_CROP_CLUSTER} obs required)")
    print("=" * 70)
    best_crops, full_table = best_crop_per_cluster(plot_yield)
    print(best_crops.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 6: Recommend crop for a NEW farmer")
    print("=" * 70)
    new_farmer_example = {
        "total_land_ha": 1.2,
        "n_plots": 2,
        "hh_has_irrigated_plot": 0,
        "has_farm_equipment": 1,
        "has_extension_access": 0,
        "head_age": 41,
        "head_is_male": 1,
        "hh_size": 5,
    }
    rec = recommend_crop_for_new_farmer(
        new_farmer_example, FEATURE_COLS, scaler, kmeans_model, best_crops
    )
    print(f"New farmer profile: {new_farmer_example}")
    print(f"-> Recommendation: {rec}")

    # Save outputs
    clustered.to_csv("household_clusters_nacal.csv", index=False)
    plot_yield.to_csv("plot_yield_with_cluster.csv", index=False)
    full_table.to_csv("cluster_crop_yield_full_table.csv", index=False)
    best_crops.to_csv("best_crop_per_cluster_nacal.csv", index=False)

    # Cluster profile summary (feature means per cluster) -- used by the
    # Streamlit app to describe each cluster in plain terms.
    profile_cols = FEATURE_COLS
    cluster_profile = clustered.groupby("km_cluster")[profile_cols].mean().round(2)
    cluster_profile["n_households"] = clustered["km_cluster"].value_counts().sort_index()
    cluster_profile.to_csv("cluster_profile_summary.csv")

    # Persist fitted model artifacts so the Streamlit app doesn't need to
    # re-run clustering (or even load the 45MB .dta) on every interaction.
    import joblib
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(kmeans_model, "kmeans_model.joblib")
    joblib.dump(FEATURE_COLS, "feature_cols.joblib")

    print("\nSaved: household_clusters_nacal.csv, plot_yield_with_cluster.csv, "
          "cluster_crop_yield_full_table.csv, best_crop_per_cluster_nacal.csv, "
          "cluster_profile_summary.csv, silhouette_scores_nacal.png, "
          "scaler.joblib, kmeans_model.joblib, feature_cols.joblib")
