"""Spotify genre model comparison and prediction dashboard."""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.spotify_classifier import build_estimators, load_saved_result

# Use one dark chart palette throughout the app.
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#121212", "axes.facecolor": "#181818",
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#ffffff",
    "xtick.color": "#b3b3b3", "ytick.color": "#b3b3b3",
    "text.color": "#ffffff", "grid.color": "#2a2a2a",
    "savefig.facecolor": "#121212",
})

# Keep project imports and saved paths relative to this folder.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Spotify Genre Lab", page_icon="♫", layout="wide")

from app.theme import HERO_HTML, THEME_CSS
from app.justify import render_justification
from app.picker import render_picker

st.markdown(THEME_CSS + HERO_HTML, unsafe_allow_html=True)

DATA_PATH = ROOT / "data" / "original" / "spotify-tracks-dataset-detailed.csv"
MODEL_PATH = ROOT / "models" / "spotify_genre_pipeline.joblib"
RESULTS_PATH = ROOT / "models" / "model_comparison.json"
TARGET = "track_genre"

MODEL_DETAILS = {
    "Logistic Regression": {
        "family": "Linear classifier",
        "description": "Learns weighted feature relationships and separates the 114 genre labels with linear decision boundaries.",
        "mechanism": "It combines the scaled audio and track features into class scores, then selects the genre with the strongest score.",
        "parameters": ["max_iter", "class_weight", "random_state"],
    },
    "Random Forest": {
        "family": "Ensemble of decision trees",
        "description": "Builds multiple decision trees from sampled rows and feature subsets, then combines their votes.",
        "mechanism": "The final genre is the strongest combined vote. Probability estimates come from the tree vote shares.",
        "parameters": ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples", "class_weight"],
    },
    "K-Nearest Neighbors": {
        "family": "Neighbor-based classifier",
        "description": "Finds the most similar tracks in the scaled feature space and uses their labels to classify the new track.",
        "mechanism": "It checks the five nearest training tracks and gives closer neighbors more voting weight.",
        "parameters": ["n_neighbors", "weights", "n_jobs"],
    },
}

FEATURE_INFO = {
    "popularity": ("Popularity", "Dataset score from 0 to 100. It is not a live Spotify score.", "score / 100"),
    "duration_ms": ("Track duration", "Track length. The form accepts minutes and converts to milliseconds for the model.", "minutes in form; milliseconds in dataset"),
    "explicit": ("Explicit content", "Dataset flag. False can also mean the source marked it as unknown.", "true / false"),
    "danceability": ("Danceability", "Audio-feature score describing how suitable the track is for dancing.", "score from 0 to 1"),
    "energy": ("Energy", "Audio-feature score for perceived intensity and activity.", "score from 0 to 1"),
    "key": ("Musical key", "Pitch-class code: 0=C, 1=C-sharp, through 11=B.", "pitch class 0 to 11"),
    "loudness": ("Loudness", "Average track loudness.", "decibels (dB)"),
    "mode": ("Musical mode", "0 means minor; 1 means major.", "0 / 1"),
    "speechiness": ("Speechiness", "Audio-feature score indicating how much spoken-word content is present.", "score from 0 to 1"),
    "acousticness": ("Acousticness", "Audio-feature score indicating how acoustic a track sounds.", "score from 0 to 1"),
    "instrumentalness": ("Instrumentalness", "Higher values indicate the track is more likely to contain no vocals.", "score from 0 to 1"),
    "liveness": ("Liveness", "Audio-feature score indicating signs of a live audience or performance.", "score from 0 to 1"),
    "valence": ("Valence", "Audio-feature score for the musical positiveness conveyed by a track.", "score from 0 to 1"),
    "tempo": ("Tempo", "Estimated overall tempo.", "beats per minute (BPM)"),
    "time_signature": ("Time signature", "Estimated beats per bar. A value of 0 represents unknown in this dataset.", "beats per bar"),
}

# Load the bundled data and verified artifacts once per app session.
if "_spotify_raw" not in st.session_state:
    if not DATA_PATH.is_file():
        st.error("The project dataset is missing from data/original.")
        st.stop()
    st.session_state["_spotify_raw"] = pd.read_csv(DATA_PATH, low_memory=False)
raw = st.session_state["_spotify_raw"]

if TARGET not in raw.columns:
    st.error(f"The required genre label column '{TARGET}' is missing from the project dataset.")
    st.stop()

artifact_signature = (
    DATA_PATH.stat().st_mtime_ns,
    MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else None,
    RESULTS_PATH.stat().st_mtime_ns if RESULTS_PATH.is_file() else None,
)
if st.session_state.get("_spotify_result_signature") != artifact_signature:
    try:
        with st.spinner("Loading the saved model and checking its results."):
            st.session_state["_spotify_result"] = load_saved_result(
                raw, TARGET, MODEL_PATH, RESULTS_PATH, DATA_PATH
            )
            st.session_state["_spotify_result_signature"] = artifact_signature
    except Exception as exc:
        st.error(f"The saved model or comparison results could not be loaded: {exc}")
        st.info("Restore the matching project model files or rerun the comparison workflow before opening the dashboard.")
        st.stop()
result = st.session_state["_spotify_result"]

required_models = list(MODEL_DETAILS)
if set(result["scores"]) != set(required_models):
    st.error("Saved comparison results must include Logistic Regression, Random Forest, and K-Nearest Neighbors.")
    st.stop()

pages = st.tabs([
    "Pick a track", "Compare", "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Manual entry", "Dataset",
])


def model_configuration(name: str) -> pd.DataFrame:
    """Return the settings used for one classifier."""
    estimator = build_estimators()[name]
    keys = MODEL_DETAILS[name]["parameters"]
    return pd.DataFrame({
        "Setting": keys,
        "Value": [str(estimator.get_params()[key]) for key in keys],
    })


def feature_label(name: str) -> str:
    """Return a clear display name for a known dataset feature."""
    return FEATURE_INFO.get(name, (name.replace("_", " ").title(), "Dataset feature.", "dataset value"))[0]


def feature_unit(name: str) -> str:
    """Return the unit shown in charts and input controls."""
    return FEATURE_INFO.get(name, (name, "Dataset feature.", "dataset value"))[2]


def render_model_page(name: str) -> None:
    """Explain one estimator and show its cross-validation result."""
    details = MODEL_DETAILS[name]
    score = result["scores"][name]
    cv_metrics = result["validation_metrics"].get(name, {})
    if "macro_f1" not in cv_metrics:
        # Gamitin ang naka-save na F1 summary; huwag mag-imbento ng ibang metrics.
        cv_metrics = {**cv_metrics, "macro_f1": score}
    actual_type = result["model_classes"][name]
    expected_type = type(build_estimators()[name]).__name__
    if actual_type != expected_type:
        st.error(f"Model identity mismatch: {name} is recorded as {actual_type}, expected {expected_type}.")
        return

    st.subheader(name)
    st.caption(f"In simple terms: {details['description']} {details['mechanism']}")
    st.markdown("#### Cross-validation results")
    st.write("Each score summarizes the same five training folds. A higher score is better; the bars show the average and whiskers show fold-to-fold variation.")
    metric_labels = [
        ("Accuracy", "accuracy"), ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]
    available_metrics = [(label, key) for label, key in metric_labels if key in cv_metrics]
    metric_columns = st.columns(len(available_metrics))
    for col, (label, key) in zip(metric_columns, available_metrics):
        values = cv_metrics[key]
        col.metric(f"{label} (mean ± SD)", f"{values['mean']:.4f} ± {values['std']:.4f}")
    if len(available_metrics) < len(metric_labels):
        st.caption("Some cross-validation metric summaries were not included in the saved comparison.")
    if available_metrics:
        chart_data = pd.DataFrame([
            {"Metric": label, "Mean": cv_metrics[key]["mean"],
             "SD": cv_metrics[key].get("std", 0.0)}
            for label, key in available_metrics
        ])
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.barh(chart_data["Metric"], chart_data["Mean"], xerr=chart_data["SD"],
                color="#1DB954", edgecolor="#b3b3b3", capsize=4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score (0 to 1)")
        ax.set_title(f"{name}: average cross-validation scores")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    model_test_metrics = result["test_metrics_by_model"].get(name)
    model_test_report = result["test_reports_by_model"].get(name)
    if model_test_metrics is not None:
        st.markdown("#### Same held-out test set")
        st.caption("All three models are scored on the same tracks. These test results are for comparison; the winner is chosen using cross-validation, not by repeatedly checking the test set.")
        test_columns = st.columns(5)
        for col, (label, key) in zip(test_columns, [
            ("Accuracy", "accuracy"), ("Macro precision", "macro_precision"),
            ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"),
            ("Weighted F1", "weighted_f1"),
        ]):
            col.metric(label, f"{model_test_metrics[key]:.4f}")
        if model_test_report is not None:
            with st.expander("View test results for each genre"):
                test_report = pd.DataFrame(model_test_report).T
                test_report = test_report.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
                st.dataframe(test_report[["precision", "recall", "f1-score", "support"]]
                             .sort_values("support", ascending=False).round(3),
                             width="stretch", height=450)
    with st.expander("Technical model settings"):
        st.write(f"Model family: {details['family']}. Scikit-learn classifier class: `{actual_type}`.")
        st.dataframe(model_configuration(name), hide_index=True, width="stretch")
    if name == result["best_name"]:
        st.success("Selected for deployment because it had the highest mean cross-validation macro F1.")
    else:
        best_score = result["scores"][result["best_name"]]["mean"]
        st.info(f"Its mean macro F1 was {score['mean']:.4f}; the selected model scored {best_score:.4f}.")



def render_prediction_input(name: str, series: pd.Series, label: str, description: str, unit: str):
    """Create one validated prediction input using observed data bounds."""
    observed_min = float(series.min())
    observed_max = float(series.max())
    observed_median = float(series.median()) if series.notna().any() else 0.0
    help_text = f"{description} Observed dataset range: {observed_min:g} to {observed_max:g} {unit}."
    if name == "duration_ms":
        return st.number_input(
            "Track duration (minutes)", min_value=observed_min / 60000,
            max_value=observed_max / 60000, value=observed_median / 60000,
            step=0.01, format="%.2f", help=help_text,
        )
    if name == "explicit":
        default = bool(series.mode(dropna=True).iloc[0]) if series.notna().any() else False
        return st.checkbox("Explicit content", value=default, help=description)
    if name == "key":
        options = sorted(int(value) for value in series.dropna().unique())
        key_names = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
        return st.selectbox(label, options, index=options.index(int(observed_median)) if int(observed_median) in options else 0,
                            format_func=lambda value: f"{value} - {key_names[value]}" if 0 <= value < len(key_names) else str(value),
                            help=help_text)
    if name == "mode":
        options = sorted(int(value) for value in series.dropna().unique())
        return st.selectbox(label, options, index=options.index(int(observed_median)) if int(observed_median) in options else 0,
                            format_func=lambda value: "Major" if value == 1 else "Minor" if value == 0 else str(value),
                            help=help_text)
    if name == "time_signature":
        options = sorted(int(value) for value in series.dropna().unique())
        return st.selectbox(label, options, index=options.index(int(observed_median)) if int(observed_median) in options else 0,
                            format_func=lambda value: "Unknown (0)" if value == 0 else f"{value} beats per bar",
                            help=help_text)
    if pd.api.types.is_integer_dtype(series.dtype):
        return st.number_input(label, min_value=int(observed_min), max_value=int(observed_max),
                               value=int(round(observed_median)), step=1, help=help_text)
    step = 0.01 if observed_max - observed_min <= 1.1 else 0.1
    precision = 2 if step == 0.01 else 1
    return st.number_input(label, min_value=observed_min, max_value=observed_max,
                           value=observed_median, step=step, format=f"%.{precision}f", help=help_text)

# Pick a track is the first tab. The other tabs show how the three classifiers were compared.
with pages[0]:
    render_picker(result, raw)

with pages[1]:
    st.subheader("Compare the three classifiers")
    st.write(f"All three use the same approximately 80/20 track-grouped split and the same {result['cv_folds']}-fold cross-validation on the training data. All are then scored on the same held-out tracks. Cross-validation macro F1 selects the winner; the test scores are reported separately.")
    st.caption(f"Dataset: {result['rows']:,} cleaned rows, {len(result['feature_names'])} model features, and {result['y'].nunique()} genre labels. Shared track IDs across train and test: {result['group_overlap']}.")

    cv_frame = pd.DataFrame([
        {"Model": name, "Mean CV macro F1": result["scores"][name]["mean"],
         "Fold standard deviation": result["scores"][name]["std"],
         "Classifier class": result["model_classes"][name]}
        for name in required_models
    ]).sort_values("Mean CV macro F1", ascending=False)
    selected_score = result["scores"][result["best_name"]]["mean"]
    st.markdown(f"<div class='section-card'><b>Selected model:</b> {result['best_name']} &nbsp; | &nbsp; <b>Classifier:</b> {result['model_classes'][result['best_name']]} &nbsp; | &nbsp; <b>CV macro F1:</b> {selected_score:.4f}</div>", unsafe_allow_html=True)

    st.markdown("#### Validation score by model")
    plot_data = cv_frame.sort_values("Mean CV macro F1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 3.7))
    bars = ax.barh(plot_data["Model"], plot_data["Mean CV macro F1"],
                   xerr=plot_data["Fold standard deviation"], color=["#1DB954" if model == result["best_name"] else "#6f6f6f" for model in plot_data["Model"]],
                   edgecolor="#b3b3b3", linewidth=.6, capsize=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean macro F1 across training folds (0 to 1)")
    ax.set_ylabel("Classifier")
    ax.set_title("Five-fold validation comparison")
    for bar, value in zip(bars, plot_data["Mean CV macro F1"]):
        ax.text(value + .012, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Longer bars mean higher average macro F1. Whiskers show one standard deviation across the same five training folds.")
    display_rows = []
    table_metrics = [
        ("Accuracy mean ± SD", "accuracy"), ("Macro precision mean ± SD", "macro_precision"),
        ("Macro recall mean ± SD", "macro_recall"), ("Macro F1 mean ± SD", "macro_f1"),
        ("Weighted F1 mean ± SD", "weighted_f1"),
    ]
    for name in cv_frame["Model"]:
        row = {"Model": name, "Classifier class": result["model_classes"][name]}
        for label, metric in table_metrics:
            values = result["validation_metrics"].get(name, {}).get(metric)
            if metric == "macro_f1" and values is None:
                values = result["scores"][name]
            row[label] = f"{values['mean']:.4f} ± {values['std']:.4f}" if values else "Not saved"
        display_rows.append(row)
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")
    with st.expander("What does macro F1 mean?"):
        st.write("Macro F1 calculates F1 for each genre, then gives every genre equal weight. It ranges from 0 to 1; higher is better. The standard deviation shows how much the score changed between folds.")

    st.markdown("#### Held-out test comparison")
    st.caption(f"Each model predicts the same {result['test_rows']:,} tracks. These results are not used to select the winner.")
    test_comparison = pd.DataFrame([
        {"Model": name, **result["test_metrics_by_model"][name]}
        for name in required_models if name in result["test_metrics_by_model"]
    ]).sort_values("macro_f1", ascending=False)
    st.dataframe(test_comparison.rename(columns={
        "accuracy": "Accuracy", "macro_precision": "Macro precision",
        "macro_recall": "Macro recall", "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
    }).round(4), hide_index=True, width="stretch")
    if not test_comparison.empty:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        bars = ax.barh(test_comparison["Model"], test_comparison["macro_f1"],
                       color=["#1DB954" if model == result["best_name"] else "#6f6f6f"
                              for model in test_comparison["Model"]])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Macro F1 on the held-out test set (0 to 1)")
        ax.set_title("Same test set: model performance comparison")
        for bar, value in zip(bars, test_comparison["macro_f1"]):
            ax.text(value + .012, bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}", va="center")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    support = np.asarray(result["confusion"]).sum(axis=1)
    top_indices = np.argsort(support)[::-1][:12]
    small_matrix = np.asarray(result["confusion"])[np.ix_(top_indices, top_indices)]
    small_labels = [result["labels"][index] for index in top_indices]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(small_matrix, annot=True, fmt="d", cmap="Greens", square=True,
                xticklabels=small_labels, yticklabels=small_labels,
                cbar_kws={"label": "Test tracks"}, ax=ax)
    ax.set(xlabel="Predicted genre", ylabel="True genre", title="Confusion matrix: 12 most frequent test genres")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("This readable view shows the 12 most frequent true genres and predictions within those genres. Each value is a track count; use the full matrix below for all 114 genres.")
    with st.expander("View full 114-genre confusion matrix"):
        full_matrix = pd.DataFrame(result["confusion"], index=result["labels"], columns=result["labels"])
        st.dataframe(full_matrix, width="stretch", height=520)
    with st.expander("View per-genre test scores"):
        report = pd.DataFrame(result["report"]).T
        report = report.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        st.dataframe(report[["precision", "recall", "f1-score", "support"]].sort_values("support", ascending=False).round(3), width="stretch", height=450)

    st.divider()
    # Evidence diagrams and verdict, computed from the saved comparison results.
    render_justification(result)

with pages[2]:
    render_model_page("Logistic Regression")

with pages[3]:
    render_model_page("Random Forest")

with pages[4]:
    render_model_page("K-Nearest Neighbors")

with pages[5]:
    st.subheader("Enter features manually")
    classifier_name = type(result["best_pipeline"].named_steps["classifier"]).__name__
    st.info(f"The deployed predictor uses the selected model: **{result['best_name']}** (`{classifier_name}`). The other model pages are for comparing how the three classifiers work.")
    st.write("Enter values for the track features. Defaults are the dataset medians or most common values. Numeric entries are limited to the observed dataset range.")

    feature_values = {}
    info_by_name = {name: FEATURE_INFO.get(name, (name.title(), "Dataset feature.", "dataset value")) for name in result["feature_names"]}
    with st.form("track_form"):
        st.markdown("#### Track details")
        metadata_columns = st.columns(3)
        metadata_features = [name for name in ["popularity", "duration_ms", "explicit", "key", "mode", "time_signature"] if name in result["feature_names"]]
        for index, name in enumerate(metadata_features):
            series = result["X"][name]
            label, description, unit = info_by_name[name]
            with metadata_columns[index % len(metadata_columns)]:
                feature_values[name] = render_prediction_input(name, series, label, description, unit)

        audio_features = [name for name in result["feature_names"] if name not in metadata_features]
        if audio_features:
            st.markdown("#### Audio features")
            audio_columns = st.columns(3)
            for index, name in enumerate(audio_features):
                series = result["X"][name]
                label, description, unit = info_by_name[name]
                with audio_columns[index % len(audio_columns)]:
                    feature_values[name] = render_prediction_input(name, series, label, description, unit)
        submitted = st.form_submit_button("Predict genre")

    if submitted:
        try:
            sample_values = {}
            for name in result["feature_names"]:
                value = feature_values[name]
                if name == "duration_ms":
                    value = float(value) * 60000
                if pd.isna(value):
                    raise ValueError(f"{feature_label(name)} is required.")
                if pd.api.types.is_numeric_dtype(result["X"][name]) and not isinstance(value, (bool, np.bool_)):
                    minimum = float(result["X"][name].min())
                    maximum = float(result["X"][name].max())
                    if not minimum <= float(value) <= maximum:
                        raise ValueError(f"{feature_label(name)} must be between {minimum:g} and {maximum:g} {feature_unit(name)}.")
                sample_values[name] = value
            sample = pd.DataFrame([sample_values], columns=result["feature_names"])
            pipeline = result["best_pipeline"]
            predicted = pipeline.predict(sample)[0]
            st.success(f"Predicted genre: **{predicted}**")
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(sample)[0]
                classifier = pipeline.named_steps["classifier"]
                top = pd.Series(probabilities, index=classifier.classes_).sort_values(ascending=False).head(5).sort_values()
                st.markdown("#### Top five model probability estimates")
                st.bar_chart(top, horizontal=True, x_label="Model probability", y_label="Genre", x_max=1.0, color="#1DB954")
                st.caption("These are model probability estimates, not guarantees that the genre is correct.")
        except Exception as exc:
            st.error(f"Could not predict this input. {exc}")

with pages[6]:
    st.subheader("Explore the dataset")
    st.write("Here is what happened to the dataset before training.")
    cleaning_columns = st.columns(4)
    cleaning_columns[0].metric("Source rows", f"{len(raw):,}")
    cleaning_columns[1].metric("Exact duplicate rows removed", f"{int(raw.duplicated().sum()):,}")
    cleaning_columns[2].metric("Rows used for modeling", f"{result['rows']:,}")
    cleaning_columns[3].metric("Genre labels", f"{result['y'].nunique():,}")
    missing_genres = int(raw[TARGET].isna().sum())
    blank_genres = int(raw[TARGET].astype("string").str.strip().eq("").sum())
    st.markdown(
        f"**Cleaning summary:** removed exact duplicate records; removed {missing_genres} rows with a missing genre "
        f"and {blank_genres} rows with a blank genre. The remaining labels were trimmed for extra spaces. "
        "Missing numeric values are filled using training-fold medians; missing categories use the most common "
        "training-fold value. This keeps test data out of preprocessing."
    )
    st.caption("In this source file, the only missing cells are in the excluded text columns (one each in artists, album_name, and track_name); the model inputs have no missing values.")
    st.write(f"Repeated track IDs are kept as separate labeled records, but all rows sharing an ID stay in the same split. There are {result['group_overlap']} track IDs shared by training and test data, so the same ID cannot leak across that boundary.")
    st.markdown("#### Fields used and excluded")
    st.write("The classifier uses the 15 numeric or yes/no track features listed below. Track IDs and long text descriptions are not used as predictors: they identify or describe a track, rather than provide consistent model inputs.")
    used_fields = pd.DataFrame([
        {"Model input": feature_label(name), "Dataset field": name,
         "What it means": FEATURE_INFO.get(name, (name, "Dataset feature.", "dataset value"))[1],
         "Scale or unit": feature_unit(name)}
        for name in result["feature_names"]
    ])
    with st.expander("See all fields the classifiers use or exclude"):
        st.dataframe(used_fields, hide_index=True, width="stretch")
        st.write("Excluded from prediction: " + ", ".join(result["dropped_ids"]) + ".")
        st.write("The excluded names and track ID help identify records but are not model features.")
        feature_missing = result["X"].isna().sum()
        st.write(f"Missing values among model inputs before fold-specific imputation: {int(feature_missing.sum()):,}.")
    st.markdown("#### Genre distribution")
    counts = result["class_counts"].sort_values(ascending=False)
    top_counts = counts.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top_counts.values, y=top_counts.index, color="#1DB954", ax=ax)
    ax.set(xlabel="Tracks after duplicate removal", ylabel="Genre", title="20 genres with the most tracks")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(f"Finding: the largest class is {counts.index[0]} ({int(counts.iloc[0]):,}); the smallest is {counts.index[-1]} ({int(counts.iloc[-1]):,}). Class counts range from {int(counts.min()):,} to {int(counts.max()):,}, so the largest class is {counts.max() / counts.min():.2f} times the smallest.")
    with st.expander("View counts for all 114 genres"):
        st.dataframe(counts.rename("Tracks").to_frame(), width="stretch", height=420)

    st.markdown("#### Example track-feature summary")
    numeric = result["X"].select_dtypes(include=["number", "bool"]).astype(float)
    feature_summary = numeric.describe().T
    feature_summary.insert(0, "Feature", [feature_label(name) for name in feature_summary.index])
    feature_summary.insert(1, "Scale or unit", [feature_unit(name) for name in feature_summary.index])
    st.dataframe(feature_summary[["Feature", "Scale or unit", "mean", "50%", "min", "max"]]
                 .rename(columns={"50%": "Median"}).round(3), hide_index=True, width="stretch")

    with st.expander("Data quality and feature notes"):
        quality = pd.DataFrame({
            "Data type": raw.dtypes.astype(str),
            "Missing values": raw.isna().sum(),
            "Unique values": raw.nunique(dropna=True),
        })
        st.markdown("**Source file summary**")
        st.write(f"{len(raw):,} rows, {len(raw.columns)} columns, {int(raw.duplicated().sum()):,} exact duplicate rows, and {int(raw[TARGET].isna().sum()):,} missing genre labels.")
        st.dataframe(quality, width="stretch", height=440)
        feature_rows = []
        for name in result["feature_names"]:
            label, meaning, unit = FEATURE_INFO.get(name, (feature_label(name), "Dataset feature.", "dataset value"))
            series = result["X"][name]
            feature_rows.append({"Feature": name, "Meaning": meaning, "Unit or scale": unit,
                                 "Observed range": f"{series.min()} to {series.max()}"})
        st.markdown("**Model feature guide**")
        st.dataframe(pd.DataFrame(feature_rows), hide_index=True, width="stretch", height=440)

        numeric_for_outliers = result["X"].select_dtypes(include=["number", "bool"]).astype(float)
        outlier_rows = []
        for name in numeric_for_outliers.columns:
            values = numeric_for_outliers[name].dropna()
            q1, q3 = values.quantile([.25, .75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flagged = int(((values < lower) | (values > upper)).sum())
            outlier_rows.append({"Feature": name, "IQR review count": flagged,
                                 "Review range": f"{lower:.3g} to {upper:.3g}"})
        st.markdown("**Potential outliers for review**")
        st.dataframe(pd.DataFrame(outlier_rows).set_index("Feature"), width="stretch", height=400)
        st.caption("IQR fences flag unusual values for review; they do not prove a value is wrong and do not remove values from training.")
        st.warning("The dataset source URL, license, and collection details were not supplied in the project folder. Add these to the source note before submission.")
