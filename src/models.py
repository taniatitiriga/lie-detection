from __future__ import annotations

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
except ModuleNotFoundError:
    RandomForestClassifier = None  # type: ignore[assignment]
    SVC = None  # type: ignore[assignment]
    MLPClassifier = None  # type: ignore[assignment]
    StratifiedKFold = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]

try:
    from scipy.ndimage import uniform_filter
except ModuleNotFoundError:
    uniform_filter = None  # type: ignore[assignment]


def rf_factory(run_seed: int):
    """Random Forest factory."""
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=run_seed,
    )

rf_factory.label = "RF"  # type: ignore[attr-defined]


def svm_factory(run_seed: int):
    """SVM factory with per-fold 4-fold CV grid search + 3×3 mean-filter smoothing."""

    class TunedSVM:
        def __init__(self):
            self._model = None

        def fit(self, X_train, y_train):
            C_values = [0.01, 0.1, 1, 10, 100]
            gamma_values = [0.001, 0.01, 0.1, 1, "scale"]

            n_C = len(C_values)
            n_gamma = len(gamma_values)
            loss_matrix = np.zeros((n_C, n_gamma))

            cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=run_seed)

            for i, C in enumerate(C_values):
                for j, gamma in enumerate(gamma_values):
                    fold_errors = []
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        Xtr, Xval = X_train[train_idx], X_train[val_idx]
                        ytr, yval = y_train[train_idx], y_train[val_idx]
                        clf = SVC(kernel="rbf", C=C, gamma=gamma, random_state=run_seed)
                        clf.fit(Xtr, ytr)
                        fold_errors.append(1.0 - accuracy_score(yval, clf.predict(Xval)))
                    loss_matrix[i, j] = np.mean(fold_errors)

            if uniform_filter is not None:
                loss_matrix = uniform_filter(loss_matrix, size=3, mode="nearest")

            best_idx = np.unravel_index(np.argmin(loss_matrix), loss_matrix.shape)
            best_C = C_values[best_idx[0]]
            best_gamma = gamma_values[best_idx[1]]

            self._model = SVC(
                kernel="rbf", C=best_C, gamma=best_gamma,
                probability=True, random_state=run_seed,
            )
            self._model.fit(X_train, y_train)

        def predict_proba(self, X_test):
            return self._model.predict_proba(X_test)

    return TunedSVM()

svm_factory.label = "SVM"  # type: ignore[attr-defined]


def nn_factory(run_seed: int):
    """Neural-network factory — compact MLP regularised for ~89 training samples.

    Architecture reduced from (100, 500) to (64, 32) to avoid extreme
    overparameterisation (~51 k → ~3.5 k parameters).  Stronger L2 (alpha=1e-3),
    a larger validation fraction for early stopping, and more iterations to
    compensate for slower convergence under stronger regularisation.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=1000,
        random_state=run_seed,
        early_stopping=True,
        validation_fraction=0.15,
    )

nn_factory.label = "NN"  # type: ignore[attr-defined]
