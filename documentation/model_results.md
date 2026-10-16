# Spotify genre model comparison

## Dataset

- Input file: `data/original/spotify-tracks-dataset-detailed.csv`.
- Original rows and columns: 114,000 and 20.
- Target: `track_genre`, with 114 genre labels.
- Exact duplicate rows removed: 450.
- Rows used after cleaning: 113,550.
- Numeric and boolean predictors used: 15 Spotify track features.
- Excluded identifiers and high-cardinality text: `track_id`, `artists`, `album_name`, and `track_name`.
- Repeated track IDs: 24,259 rows; 16,299 track IDs have multiple genre labels.
- Unique track groups: 89,741.
- Train/test: 90,839 / 22,711 rows using stratified group splitting (approximately 80/20).
- Shared track IDs between train and test: 0.
- Cross-validation: 5-fold stratified group CV on the training partition.
- Selection metric: mean validation macro F1.
- Random seed: 42.
- Logistic Regression: `max_iter=1500`, balanced class weights.
- Random Forest: 40 trees, `max_depth=12`, `min_samples_leaf=5`, `max_features=sqrt`, half-sample bootstrap, balanced-subsample class weights, two workers.
- K-Nearest Neighbors: 5 neighbors, distance weights, two workers.

Cleaning removes exact duplicate rows and rows with missing or blank target labels. This file has 450 exact duplicates and no missing or blank genre labels, leaving 113,550 records. Missing values are present only in the free-text `artists`, `album_name`, and `track_name` columns; those columns are excluded from the model. Missing model-feature values, if present in a changed dataset, are imputed within each training fold.

The 15 model inputs are `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `time_signature`. The fields `track_id`, `artists`, `album_name`, and `track_name` are excluded because they identify a record or are free-text descriptions, not consistent numeric/boolean classifier inputs. Popularity is retained as a dataset track feature; it is not a live Spotify score.

## Cross-validation results

| Model | Mean macro F1 | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.1730 | 0.0012 |
| Random Forest | 0.2737 | 0.0019 |
| K-Nearest Neighbors | 0.2357 | 0.0008 |

Random Forest was selected because it achieved the highest mean macro F1 across the five training folds.

## Held-out test comparison

All three classifiers are refit on the same training partition and scored on the same held-out test partition. The test comparison is reported for transparency; model selection remains based on cross-validation macro F1, not the test results.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.2049 | 0.1711 | 0.2054 | 0.1761 | 0.1757 |
| Random Forest | 0.3109 | 0.3175 | 0.3115 | 0.2793 | 0.2788 |
| K-Nearest Neighbors | 0.2495 | 0.2470 | 0.2500 | 0.2424 | 0.2419 |

The fitted selected pipeline is saved as `models/spotify_genre_pipeline.joblib`. It contains training-fold median imputation, boolean/numeric scaling, the Random Forest, and the feature schema.

## Interpretation and limitations

Performance is modest for a 114-class task. The Random Forest improves macro F1 over the other tested models, but the test result shows that Spotify audio features alone do not reliably separate all listed genres. The source repeats some tracks under multiple genre labels; the group split prevents direct track overlap between training and testing, while the conflicting labels still add uncertainty. The dataset source and license were not present with the CSV and must be documented by the group before submission. Results are not evidence of performance on new platform data outside this dataset.

The app explains the cleaning and input fields in plain language. Each model page shows its cross-validation score graph and held-out test metrics. The comparison page uses cross-validation macro F1 to identify the selected model, reports all three test scores separately, and shows the selected model's confusion matrix. The explore page includes feature summaries and data-quality information.
