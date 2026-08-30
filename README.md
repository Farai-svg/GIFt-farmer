# GIFt Farmer Crop Recommender (Streamlit) -- v2

Interactive app for the NACAL k-means clustering pipeline: enter a new
farmer's characteristics, get matched to the closest household cluster,
and see which crop performs best in that cluster.

**v2 change:** added marital status, urban/rural residence, region,
district, and land tenure to the clustering features, on top of the
original farm/farmer traits (land size, plot count, irrigation, equipment,
extension access, farmer age/sex, household size). See "Mixed data types"
below for how these are combined with the numeric features.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `gift_nacal_clustering.py` | The full analysis pipeline (run this first / to regenerate artifacts) |
| `kmeans_model.joblib` | Fitted `KMeans` model |
| `encoding.joblib` | Numeric scaler + categorical category lists/weights, used to transform a new farmer's inputs into the same feature space the model was trained on |
| `cluster_profile_summary.csv` | Mean feature values per cluster, plus each cluster's most common marital status / region / district / land tenure |
| `cluster_crop_yield_full_table.csv` | Cluster x crop yield stats (median/mean yield, relative index, plot counts) |
| `household_clusters_nacal.csv` | Every household with its assigned cluster |
| `plot_yield_with_cluster.csv` | Every plot-level yield record with its household's cluster |
| `best_crop_per_cluster_nacal.csv` | The single best crop per cluster |
| `silhouette_scores_nacal.png` | Silhouette score by k, showing why k=7 was selected |
| `requirements.txt` | Python dependencies |
| `screenshots/` | Example screenshots of the app in use |

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
  irrigation/equipment/extension access, farmer age & sex, household size,
  marital status, urban/rural residence, region, district, and land tenure.
  The app encodes these the same way the training data was encoded,
  predicts the nearest cluster, and shows the top candidate crops for that
  cluster ranked by relative yield index -- along with how this farmer's
  inputs compare to the cluster's typical profile, and the cluster's most
  common categorical characteristics.
- **Explore clusters & crops** tab: browse all cluster profiles, and filter
  the full cluster x crop performance table by cluster, crop, and minimum
  number of observed plots (to exclude thin/unreliable estimates).

## Mixed data types: how numeric and categorical features are combined

K-means needs numeric input. Continuous/binary features (land size, plot
count, age, etc.) are standardized to mean 0 / sd 1, same as v1. The new
categorical features (marital status, region, district, land tenure) are
one-hot encoded -- but District alone has 32 categories, which would give
it far more columns (and therefore far more influence on distance
calculations) than every other feature combined. To prevent District from
mechanically dominating the clustering, each categorical variable's one-hot
block is scaled by `1/sqrt(number of categories)`, keeping every variable's
*total* contribution to the distance metric comparable regardless of its
cardinality. This logic lives in `fit_transform_features()` /
`transform_new_farmer()` in `gift_nacal_clustering.py`, and is mirrored
exactly in `app.py` (via the saved `encoding.joblib`) so a new farmer's
inputs land in the same feature space the model was trained on.

## Customizing the interface

- **Colors/theme**: edit `.streamlit/config.toml` — no code changes needed.
  Change `primaryColor` (buttons, active tab, slider), `backgroundColor`,
  `secondaryBackgroundColor` (sidebar), or `textColor`, using any hex code.
  Restart the app (or it'll pick it up on the next deploy) to see the change.
- **Font size**: the main panel (not the sidebar) uses a custom CSS block
  near the top of `app.py` (`st.markdown(..., unsafe_allow_html=True)`,
  right after `st.set_page_config`) to render everything more compactly.
  Adjust the `rem` values in that block to make text larger or smaller --
  e.g. change `font-size: 0.85rem;` to `1rem` for normal size, or `0.75rem`
  for even more compact. It only targets `[data-testid="stMain"]`, so the
  sidebar's font size is unaffected.
- **Layout**: as of this version, all farmer inputs live in the left
  sidebar (`with st.sidebar:` block in `app.py`), leaving the main panel
  full-width for results. To move fields back into the main area, or
  rearrange which fields appear where, edit the `with st.sidebar:` block
  and the two `with tab_recommend:` / `with tab_explore:` blocks directly.

## Known data caveat

A few crops (tobacco especially) show implausibly high absolute yields
compared to published agronomic benchmarks, most likely due to a units
issue in the underlying survey's harvest-quantity field for some records.
Treat absolute yield numbers as directional rather than exact until this
is checked against the original questionnaire documentation. The relative
yield index is somewhat more robust to this than raw kg/ha, since it
compares a cluster to the same crop's own population median, but it is not
immune to record-level unit errors.
