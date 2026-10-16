"""Reusable data preparation and three-model comparison for genre classification."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_CANDIDATES = ("track_genre", "genre", "music_genre", "label", "target")


def suggest_target(columns: list[str]) -> str:
    """Choose a likely label column, or the last column as a fallback."""
    by_normalized_name = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in by_normalized_name:
            return by_normalized_name[candidate]
    return columns[-1]


def prepare_dataset(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """Clean labels and remove empty, constant, and obvious identifier columns."""
    # Keep only rows with usable genre labels.
    frame = data.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if target not in frame.columns:
        raise ValueError("Choose a target column that exists in the uploaded CSV.")
    frame = frame.drop_duplicates()
    frame = frame.dropna(subset=[target])
    y = frame[target].astype(str).str.strip()
    valid_labels = y.ne("") & y.str.lower().ne("nan")
    frame, y = frame.loc[valid_labels], y.loc[valid_labels]
    X = frame.drop(columns=[target])
    group_column = next((c for c in X.columns if re.fullmatch(r"track.?id", c, re.I)), None)
    groups = frame[group_column].astype(str) if group_column else pd.Series(frame.index.astype(str), index=frame.index)
    id_columns = [c for c in X.columns if re.search(r"(^id$|_id$|^unnamed|track.?id)", c, re.I)]
    text_columns = X.select_dtypes(include=["object", "string"]).columns
    high_cardinality = [c for c in text_columns if X[c].nunique(dropna=True) > max(100, int(len(X) * 0.1))]
    excluded_columns = sorted(set(id_columns + high_cardinality))
    X = X.drop(columns=excluded_columns, errors="ignore")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.empty:
        raise ValueError("No usable feature columns remain after removing IDs and constants.")
    if y.nunique() < 2:
        raise ValueError("The selected target must contain at least two classes.")
    if y.value_counts().min() < 2:
        raise ValueError("Every target class needs at least two rows for a stratified split.")
    return X, y, excluded_columns, groups


def build_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    """Fit imputation, encoding, and scaling inside each training fold."""
    # Put every learned transform inside the model pipeline to prevent leakage.
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def build_estimators() -> dict:
    """Create the three project estimators with their documented settings."""
    # Keep estimator names and types explicit for the app and report.
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=5, max_features="sqrt", max_samples=0.5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=2),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=2),
    }

def compare_models(data: pd.DataFrame, target: str, folds: int = 5) -> dict:
    """Compare the three required estimators and test the selected one once."""
    # Use one stratified split and the same folds for all three models.
    X, y, dropped_ids, groups = prepare_dataset(data, target)
    counts = y.value_counts()
    cv_folds = min(folds, int(counts.min()))
    if cv_folds < 2:
        raise ValueError("Not enough training examples per class for cross-validation.")
    # Keep all rows for a track in the same split and validation fold.
    splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    split_indices = list(splitter.split(X, y, groups))
    train_indices, test_indices = split_indices[0]
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    groups_train = groups.iloc[train_indices]
    estimators = build_estimators()
    pipelines = {name: build_pipeline(X, estimator) for name, estimator in estimators.items()}
    metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
    scores = {}
    validation_metrics = {}
    validation_confusions = {}
    validation_reports = {}
    test_metrics_by_model = {}
    test_confusions_by_model = {}
    test_reports_by_model = {}
    fitted_pipelines = {}
    labels = sorted(y.unique())
    for name, pipeline in pipelines.items():
        fold_values = {metric: [] for metric in metric_names}
        oof_true = []
        oof_prediction = []
        for inner_train, inner_valid in StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        ).split(X_train, y_train, groups_train):
            fold_model = clone(pipeline).fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
            y_valid = y_train.iloc[inner_valid]
            fold_prediction = fold_model.predict(X_train.iloc[inner_valid])
            fold_scores = {
                "accuracy": accuracy_score(y_valid, fold_prediction),
                "macro_precision": precision_score(y_valid, fold_prediction, average="macro", zero_division=0),
                "macro_recall": recall_score(y_valid, fold_prediction, average="macro", zero_division=0),
                "macro_f1": f1_score(y_valid, fold_prediction, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_valid, fold_prediction, average="weighted", zero_division=0),
            }
            for metric, value in fold_scores.items():
                fold_values[metric].append(float(value))
            oof_true.extend(y_valid.astype(str).tolist())
            oof_prediction.extend(pd.Series(fold_prediction).astype(str).tolist())
        validation_metrics[name] = {}
        for metric, values in fold_values.items():
            values = np.asarray(values)
            validation_metrics[name][metric] = {"mean": float(values.mean()), "std": float(values.std())}
        scores[name] = validation_metrics[name]["macro_f1"]
        validation_confusions[name] = confusion_matrix(oof_true, oof_prediction, labels=labels).tolist()
        validation_reports[name] = classification_report(
            oof_true, oof_prediction, labels=labels, output_dict=True, zero_division=0
        )
    selected_name = max(scores, key=lambda name: scores[name]["mean"])
    for name, pipeline in pipelines.items():
        # I-score ang bawat modelo sa parehong holdout; CV pa rin ang batayan ng pagpili.
        fitted = clone(pipeline).fit(X_train, y_train)
        prediction = fitted.predict(X_test)
        fitted_pipelines[name] = fitted
        test_metrics_by_model[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
        }
        test_confusions_by_model[name] = confusion_matrix(y_test, prediction, labels=labels).tolist()
        test_reports_by_model[name] = classification_report(
            y_test, prediction, labels=labels, output_dict=True, zero_division=0
        )
    selected = fitted_pipelines[selected_name]
    metrics = test_metrics_by_model[selected_name]
    test_confusion = np.asarray(test_confusions_by_model[selected_name], dtype=int)
    test_report = test_reports_by_model[selected_name]
    return {
        "X": X, "y": y, "X_test": X_test, "y_test": y_test,
        "pipelines": pipelines, "model_classes": {name: type(estimator).__name__ for name, estimator in estimators.items()},
        "scores": scores, "validation_metrics": validation_metrics,
        "validation_confusions": validation_confusions, "validation_reports": validation_reports,
        "test_metrics_by_model": test_metrics_by_model,
        "test_confusions_by_model": test_confusions_by_model,
        "test_reports_by_model": test_reports_by_model,
        "best_pipeline": selected,
        "best_name": selected_name, "metrics": metrics,
        "confusion": test_confusion,
        "labels": labels, "report": test_report,
        "class_counts": counts, "dropped_ids": dropped_ids,
        "feature_names": X.columns.tolist(), "rows": len(X), "target": target,
        "cv_folds": cv_folds,
        "test_indices": test_indices.tolist(),
        "train_rows": len(train_indices), "test_rows": len(test_indices),
        "track_groups": int(groups.nunique()),
        "group_overlap": int(len(set(groups.iloc[train_indices]) & set(groups.iloc[test_indices]))),
    }


def save_pipeline(result: dict, path: str | Path = "models/spotify_genre_pipeline.joblib") -> Path:
    """Save the fitted pipeline and the feature schema used by the app."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result["best_pipeline"], "features": result["feature_names"],
                 "target": result["target"], "model": result["best_name"],
                 "model_class": result["model_classes"][result["best_name"]],
                 "metrics": result["metrics"]}, output)
    return output


def save_comparison_metadata(result: dict, metadata_path: str | Path,
                            data_path: str | Path) -> Path:
    """Save the measured CV and selected-model test results."""
    import hashlib
    import json

    output = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target": result["target"],
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "selected_model": result["best_name"],
        "selected_model_class": result["model_classes"][result["best_name"]],
        "cv_folds": result["cv_folds"],
        "primary_metric": "macro_f1",
        "cv_scores": result["scores"],
        "cv_metrics": result["validation_metrics"],
        "cv_labels": result["labels"],
        "cv_confusions": result["validation_confusions"],
        "cv_reports": result["validation_reports"],
        "test_metrics": result["metrics"],
        "test_metrics_by_model": result["test_metrics_by_model"],
        "test_confusions_by_model": result["test_confusions_by_model"],
        "test_reports_by_model": result["test_reports_by_model"],
        "test_labels": result["labels"],
        "test_confusion": np.asarray(result["confusion"]).tolist(),
        "test_report": result["report"],
        "test_indices": result["test_indices"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "track_groups": result["track_groups"],
        "group_overlap": result["group_overlap"],
    }
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output

def load_saved_result(data: pd.DataFrame, target: str, model_path: str | Path,
                      metadata_path: str | Path, data_path: str | Path) -> dict:
    """Load the saved comparison when its dataset fingerprint still matches."""
    import hashlib
    import json

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if digest != metadata["dataset_sha256"] or target != metadata["target"]:
        raise ValueError("The saved model does not match this dataset or target. Run a new model comparison.")
    saved = joblib.load(model_path)
    X, y, excluded, groups = prepare_dataset(data, target)
    if X.columns.tolist() != saved["features"]:
        raise ValueError("The saved model feature schema does not match this dataset.")
    if "test_indices" in metadata:
        test_indices = np.asarray(metadata["test_indices"], dtype=int)
        if (
            test_indices.ndim != 1
            or len(test_indices) != metadata["test_rows"]
            or np.any(test_indices < 0)
            or np.any(test_indices >= len(X))
            or len(np.unique(test_indices)) != len(test_indices)
        ):
            raise ValueError("The saved test split indices are invalid for this dataset.")
        train_indices = np.setdiff1d(np.arange(len(X)), test_indices, assume_unique=True)
    else:
        splitter = StratifiedGroupKFold(n_splits=metadata["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
        train_indices, test_indices = next(splitter.split(X, y, groups))
    X_test = X.iloc[test_indices][saved["features"]]
    y_test = y.iloc[test_indices]
    saved_class = type(saved["pipeline"].named_steps["classifier"]).__name__
    model_classes = {name: type(estimator).__name__ for name, estimator in build_estimators().items()}
    if saved_class != saved.get("model_class", model_classes.get(saved["model"])) or saved_class != model_classes.get(saved["model"]):
        raise ValueError("The saved model type does not match its selected model label.")
    if saved["model"] != metadata.get("selected_model", saved["model"]):
        raise ValueError("The saved pipeline does not match the model selected in the comparison.")
    labels = metadata.get("test_labels", sorted(y.unique()))
    if all(key in metadata for key in ("test_metrics_by_model", "test_confusions_by_model", "test_reports_by_model")):
        test_metrics_by_model = metadata["test_metrics_by_model"]
        test_confusions_by_model = metadata["test_confusions_by_model"]
        test_reports_by_model = metadata["test_reports_by_model"]
        metrics = test_metrics_by_model[saved["model"]]
        test_confusion = np.asarray(test_confusions_by_model[saved["model"]], dtype=int)
        test_report = test_reports_by_model[saved["model"]]
    else:
        prediction = saved["pipeline"].predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
        }
        test_confusion = confusion_matrix(y_test, prediction, labels=labels)
        test_report = classification_report(y_test, prediction, labels=labels, output_dict=True, zero_division=0)
        test_metrics_by_model = {saved["model"]: metrics}
        test_confusions_by_model = {saved["model"]: test_confusion.tolist()}
        test_reports_by_model = {saved["model"]: test_report}
    return {
        "X": X, "y": y, "X_test": X_test, "y_test": y_test,
        "scores": metadata["cv_scores"], "validation_metrics": metadata.get("cv_metrics", {}),
        "validation_confusions": metadata.get("cv_confusions", {}),
        "validation_reports": metadata.get("cv_reports", {}), "model_classes": model_classes,
        "test_metrics_by_model": test_metrics_by_model,
        "test_confusions_by_model": test_confusions_by_model,
        "test_reports_by_model": test_reports_by_model,
        "best_pipeline": saved["pipeline"],
        "best_name": saved["model"], "metrics": metrics,
        "confusion": test_confusion,
        "labels": labels, "report": test_report,
        "class_counts": y.value_counts(), "dropped_ids": excluded,
        "feature_names": X.columns.tolist(), "rows": len(X), "target": target,
        "cv_folds": metadata["cv_folds"], "train_rows": len(train_indices),
        "test_rows": len(test_indices), "track_groups": int(groups.nunique()),
        "group_overlap": int(len(set(groups.iloc[train_indices]) & set(groups.iloc[test_indices]))),
        "test_indices": test_indices.tolist(),
    }
