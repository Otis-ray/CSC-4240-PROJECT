from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

MODEL_FORMAT = 1
TREE_ITERATIONS = 300
LEARNING_RATE = 0.1

def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=TREE_ITERATIONS, learning_rate=LEARNING_RATE, early_stopping=False, random_state=seed
    )

def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=TREE_ITERATIONS, learning_rate=LEARNING_RATE, early_stopping=False, random_state=seed
    )


class _OnlyChoice:
    # fallback classifier when only a single label class appears in training data
    def __init__(self, value):
        self.classes = np.array([value])

    def predict_proba(self, features):
        return np.ones((len(features), 1))


def _with_slopes(features) -> np.ndarray:
    # augment spectral band features with first order differences
    X = np.atleast_2d(np.asarray(features, dtype=np.float64))
    return np.hstack([X, np.diff(X, axis=1)])

def cutoff_edges(kind: str, cutoff) -> list[float]:
    # normalize scalar or cutoff specs into a list of Hz freqs
    return [float(cutoff[0]), float(cutoff[1])] if kind == "bandpass" else [float(cutoff)]


def _edge_names(kind: str) -> list[str]:
    return ["bandpass_low", "bandpass_high"] if kind == "bandpass" else [kind]

def _pick_rows(items: list, rows) -> list:
    return [items[row] for row in rows]


class FilterIdModel:
    # Model filter type, order, and cutoff frequencies hierarchically.
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.type_model = None
        self.order_models = {}
        self.cutoff_models = {}
        self.cutoff_ranges = {}


    def fit(self, features, labels: list[dict]) -> FilterIdModel:
        X = _with_slopes(features)
        if len(X) != len(labels):
            raise ValueError(f"got {len(X)} feature rows but {len(labels)} labels")

        # reset learned sub models and bounds on each call
        self.order_models, self.cutoff_models, self.cutoff_ranges = {}, {}, {}

        types = np.array([label["type"] for label in labels])
        self.type_model = _classifier(self.seed).fit(X, types)

        # train per filter type order classifiers and cutoff regressors on matching subsets
        for kind in np.unique(types).tolist():
            rows = np.flatnonzero(types == kind)
            labels_of_kind = _pick_rows(labels, rows)
            self._fit_order(kind, X[rows], labels_of_kind)
            self._fit_cutoffs(kind, X[rows], labels_of_kind)
        
        return self

    def _fit_order(self, kind: str, X: np.ndarray, labels: list[dict]) -> None:
        # HistGradientBoostingClassifier requires two distinct classes
        orders = np.array([label["order"] for label in labels])
        if len(np.unique(orders)) == 1:
            self.order_models[kind] = _OnlyChoice(orders[0])
        else:
            self.order_models[kind] = _classifier(self.seed).fit(X, orders)


    def _fit_cutoffs(self, kind: str, X: np.ndarray, labels: list[dict]) -> None:
        edges_hz = [cutoff_edges(kind, label["cutoff"]) for label in labels]
        # Regress cutoffs in log2(Hz) so multiplicative errors are balanced.
        octaves = np.log2(edges_hz)

        regressors = [_regressor(self.seed).fit(X, octaves[:, edge]) for edge in range(octaves.shape[1])]
        self.cutoff_models[kind] = regressors
                # Store observed octave bounds to clamp out-of-range predictions.
        self.cutoff_ranges[kind] = (octaves.min(axis=0), octaves.max(axis=0))


    def _check_trained(self) -> None:
        if self.type_model is None:
            raise RuntimeError("the model isn't trained yet; call fit first or load a saved model")


    def _predict_cutoffs(self, X: np.ndarray, kind: str) -> np.ndarray:
        lowest, highest = self.cutoff_ranges[kind]
        columns = []
        for edge, regressor in enumerate(self.cutoff_models[kind]):
            octaves = np.clip(regressor.predict(X), lowest[edge], highest[edge])
            columns.append(2**octaves)
        return np.column_stack(columns)


    def _given_type(self, X: np.ndarray, kind: str) -> list[dict]:
        order_model = self.order_models[kind]
        order_proba = order_model.predict_proba(X)
        best_order = np.argmax(order_proba, axis=1)
        edges_hz = self._predict_cutoffs(X, kind)

        results = []
        for row in range(len(X)):
            cutoff = [float(edges_hz[row, 0]), float(edges_hz[row, 1])] if kind == "bandpass" else float(edges_hz[row, 0])
            order = int(order_model.classes_[best_order[row]])
            order_confidence = float(order_proba[row, best_order[row]])
            results.append({"spec": {"type": kind, "family": "butter", "order": order, "cutoff": cutoff}, "confidence": {"order": order_confidence}})

        return results

    def predict( self, features, kind: str | None = None) -> list[dict]:
        self._check_trained()
        X = _with_slopes(features)

        # force a given filter type instead of predicting it
        if kind is not None:
            results = self._given_type(X, kind)
            for result in results:
                result["confidence"]["type"] = 1.0
            return results
        # Check that the type classifier has been trained before prediction.
        type_model = self.type_model
        if type_model is None:
            raise RuntimeError("model is not trained")

        type_proba = type_model.predict_proba(X)
        best_type = np.argmax(type_proba, axis=1)
        predicted_types = type_model.classes_[best_type]

        results = [{}] * len(X)
        # batch sub predictions by predicted filter type
        for predicted in np.unique(predicted_types).tolist():
            rows = np.flatnonzero(predicted_types == predicted)
            results_for_type = self._given_type(X[rows], predicted)
            for row, result in zip(rows, results_for_type):
                result["confidence"]["type"] = float(type_proba[row, best_type[row]])
                results[row] = result

        return results


def split_by_recording(groups, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # GroupShuffleSplit keeps examples from one recording in a single split.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def _accuracy(hits) -> float:
   return float("nan") if len(hits) == 0 else float(np.mean(hits))


def _scores_given_true_type(model: FilterIdModel, X: np.ndarray, labels: list[dict]) -> tuple[dict, dict]:
    # Evaluate order and cutoff errors conditional on the true filter type.
    true_types = [label["type"] for label in labels]
    order_hits, cutoff_errors = {}, {}

    for kind in sorted(set(true_types)):
        rows = [i for i, t in enumerate(true_types) if t == kind]
        results = model.predict(X[rows], kind=kind)

        for i, result in zip(rows, results):
            order_hits.setdefault(kind, []).append(result["spec"]["order"] == labels[i]["order"])
            true_edges = cutoff_edges(kind, labels[i]["cutoff"])
            predicted_edges = cutoff_edges(kind, result["spec"]["cutoff"])

            for name, true_hz, predicted_hz in zip(_edge_names(kind), true_edges, predicted_edges):
                cutoff_errors.setdefault(name, []).append(abs(float(np.log2(predicted_hz / true_hz))))

    return order_hits, cutoff_errors


def _type_confusion(predicted: list[dict], labels: list[dict]) -> dict:
    # Build a nested type-confusion matrix.
    confusion = {}
    for result, label in zip(predicted, labels):
        true_type, guess = label["type"], result["spec"]["type"]
        confusion.setdefault(true_type, {})[guess] = confusion.setdefault(true_type, {}).get(guess, 0) + 1
    return confusion


def _scores_by_noise_stage(type_hits: np.ndarray, both_hits: np.ndarray, records: list[dict]) -> dict:
    # Group evaluation accuracy by the first recorded damage stage.
    stages = np.array([r["damage"][0]["stage"] if r.get("damage") and len(r["damage"]) > 0 and "stage" in r["damage"][0] else "none" for r in records])
    by_stage = {}
    for stage in sorted(set(stages.tolist())):
        in_stage = stages == stage
        by_stage[stage] = {
            "examples": int(np.sum(in_stage)),
            "type_accuracy": _accuracy(type_hits[in_stage]),
            "type_and_order_accuracy": _accuracy(both_hits[in_stage]),
        }
    return by_stage


def evaluate(model: FilterIdModel, features, labels: list[dict], records: list[dict]) -> dict:
    X = np.atleast_2d(np.asarray(features))
    if not len(X) == len(labels) == len(records):
        raise ValueError(f"got {len(X)} feature rows, {len(labels)} labels, and {len(records)} records")

    predicted = model.predict(X)
    type_hits = np.array([r["spec"]["type"] == l["type"] for r, l in zip(predicted, labels)])
    both_hits = np.array([(r["spec"]["type"] == l["type"] and r["spec"]["order"] == l["order"]) for r, l in zip(predicted, labels)])

    order_hits, cutoff_errors = _scores_given_true_type(model, X, labels)
    all_order_hits = [h for hits in order_hits.values() for h in hits]

    return {
        "examples": len(labels),
        "type_accuracy": _accuracy(type_hits),
        "type_and_order_accuracy": _accuracy(both_hits),
        "order_accuracy_given_true_type": _accuracy(all_order_hits),
        "order_accuracy_given_true_type_by_type": {k: _accuracy(v) for k, v in order_hits.items()},
        "cutoff_error_octaves": {name: {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90))} for name, errs in cutoff_errors.items()},
        "by_noise_stage": _scores_by_noise_stage(type_hits, both_hits, records),
        "type_confusion": _type_confusion(predicted, labels),
    }


def save_model(path, model: FilterIdModel, settings: dict) -> None:
    joblib.dump({"format": MODEL_FORMAT, "model": model, "settings": settings}, path)


def load_model(path) -> tuple[FilterIdModel, dict]:
    saved = joblib.load(path)
    if saved.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} has model format {saved.get('format')!r}, expected {MODEL_FORMAT}")
    return saved["model"], saved["settings"]
