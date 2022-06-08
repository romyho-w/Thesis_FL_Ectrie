
import numpy as np
import matplotlib.pyplot as plt

from typing import Union

Number = Union[int, float]


def make_features(intervals: list[tuple[Number]], n_samples: int) -> np.ndarray:
    n_features = len(intervals)
    X = np.random.rand(n_samples, n_features)
    # X = np.random.multivariate_normal(mean, )
    for i, (low, high) in enumerate(intervals):
        diff = high - low
        X[:, i] = (X[:, i] * diff) + low
    return X


def make_labels(X: np.ndarray, thresholds: list[Number]) -> np.ndarray:
    thresholds = np.array(thresholds)[:, None]
    return (X @ thresholds) > 0


def make_logreg_data(intervals: list[tuple[Number]], thresholds: list[Number], n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    X = make_features(intervals, n_samples)
    y = make_labels(X, thresholds)
    return X, y


if __name__ == "__main__":
    intervals = [(-100, 100), (-100, 100)]
    thresholds = [-30, -40]
    n_samples = 1000
    X, y = make_logreg_data(intervals, thresholds, n_samples)
    # print(X)
    plt.figure()
    idx_red = y[:, 0] == 1
    plt.scatter(X[:, 0], X[:, 1], label='y = 0')
    plt.scatter(X[idx_red, 0], X[idx_red, 1], label='y = 1')
    plt.legend()
    plt.xlabel("feature")
    plt.ylabel("label")
    plt.title("make logistic regression data")
    plt.grid()
    plt.show()
