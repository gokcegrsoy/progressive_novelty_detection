import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class RandomForestClassifierUnc(RandomForestClassifier):
    def predict_uncertainty(self, X: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        preds = []
        for e in self.estimators_:
            preds.append(e.predict_proba(X) + 1e-12)
        preds_array = np.stack(preds, axis=0)
        tu = entropy(np.mean(preds_array, axis=0), axis=1)
        du = np.mean(entropy(preds_array, axis=2), axis=0)
        return tu, tu - du


class XGBEnsemble:
    def __init__(self) -> None:
        self.ensemble: list[XGBClassifier] = []

    def fit(self, X: npt.NDArray, y: npt.NDArray, n_estimators: int = 10) -> "XGBEnsemble":
        self.ensemble = []
        for seed in range(n_estimators):
            self.ensemble.append(XGBClassifier(random_state=seed).fit(X, y))
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        preds = []
        for tree in self.ensemble:
            preds.append(tree.predict_proba(X))
        preds_array = np.stack(preds, axis=0)
        return np.argmax(np.mean(preds_array, axis=0), axis=1)

    def predict_uncertainty(self, X: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        preds = []
        for tree in self.ensemble:
            preds.append(tree.predict_proba(X))
        preds_array = np.stack(preds, axis=0)
        tu = entropy(np.mean(preds_array, axis=0), axis=-1)
        du = np.mean(entropy(preds_array, axis=-1), axis=0)
        return tu, tu - du


def entropy(x: npt.NDArray, axis: int) -> npt.NDArray:
    return np.sum(-x * np.log(x), axis=axis)


def train_gbdt_ensemble(X: npt.NDArray, y: npt.NDArray) -> list[XGBClassifier]:
    ensemble: list[XGBClassifier] = []
    for seed in range(10):
        ensemble.append(XGBClassifier(random_state=seed).fit(X, y))
    return ensemble


def predict_ensemble(ensemble: list[XGBClassifier], X: npt.NDArray):
    preds = []
    for tree in ensemble:
        preds.append(tree.predict_proba(X))
    preds_array = np.stack(preds, axis=0)
    tu = entropy(np.mean(preds_array, axis=0), axis=-1)
    du = np.mean(entropy(preds_array, axis=-1), axis=0)
    return np.argmax(np.mean(preds_array, axis=0), axis=1), tu, tu - du
