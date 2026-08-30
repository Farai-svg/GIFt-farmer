# GIFt Farmer Crop Recommender (Streamlit)

Interactive app for the NACAL k-means clustering pipeline: enter a new
farmer's characteristics, get matched to the closest household cluster,
and see which crop performs best in that cluster.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `gift_nacal_clustering.py` | The full analysis pipeline (run this first / to regenerate artifacts) |
| `scaler.joblib` | Fitted `StandardScaler` from the pipeline |
| `kmeans_model.joblib` | Fitted `KMeans` model |
| `feature_cols.joblib` | List of feature columns, in the order the model expects |
| `cluster_profile_summary.csv` | Mean feature values per cluster (for display/comparison) |
| `cluster_crop_yield_full_table.csv` | Cluster x crop yield stats (median/mean yield, relative index, plot counts) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

If you regenerate the underlying data (new survey extract, different
feature set, etc.), re-run the pipeline first to refresh the `.joblib`
and `.csv` artifacts the app depends on:

```bash
python gift_nacal_clustering.py
```

## Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (default: http://localhost:8501).

## What the app does

- **Recommend for a new farmer** tab: enter land size, plot count,
  irrigation/equipment/extension access, farmer age & sex, and household
  size. The app scales these the same way the training data was scaled,
  predicts the nearest cluster, and shows the top candidate crops for that
  cluster ranked by relative yield index (a crop's yield in this cluster
  vs. its yield across all farmers) -- along with how this farmer's inputs
  compare to the cluster's typical profile.
- **Explore clusters & crops** tab: browse all cluster profiles, and filter
  the full cluster x crop performance table by cluster, crop, and minimum
  number of observed plots (to exclude thin/unreliable estimates).

## Known data caveat

A few crops (tobacco especially) show implausibly high absolute yields
compared to published agronomic benchmarks, most likely due to a units
issue in the underlying survey's harvest-quantity field for some records.
Treat absolute yield numbers as directional rather than exact until this
is checked against the original questionnaire documentation. The relative
yield index is somewhat more robust to this than raw kg/ha, since it
compares a cluster to the same crop's own population median, but it is not
immune to record-level unit errors.
