#%%
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn import metrics


#%%
def plot_data(x: np.ndarray, y: np.ndarray, y_noisy: np.ndarray, n_samples_plot: int = 100) -> tuple[Figure, Axes]:
    """
    Plots the data set.
    """
    noisy_alpha = 0.33
    noisy_color = (0.8, 0.8, 0.8)

    idx_sort = np.argsort(x)
    x, y, y_noisy = x[idx_sort], y[idx_sort], y_noisy[idx_sort]

    # Show only n_samples_plot samples of noisy data
    n_samples = x.shape[0]
    plot_interval = int(n_samples / n_samples_plot)
    idx_plot_data = np.arange(0, n_samples, plot_interval)

    fig, ax = plt.subplots()

    plt.scatter(x[idx_plot_data], y_noisy[idx_plot_data], alpha=noisy_alpha, color=noisy_color, label="noisy data")
    plt.plot(x, y, color="red", label="underlying data")

    plt.xlabel("input")
    plt.ylabel("output")
    plt.title("Data to be fitted.")

    plt.legend(bbox_to_anchor=(1.00, 1))
    plt.grid()

    plt.show()

    return fig, ax


def plot_loss_vs_iteration(
        **kwargs: dict[str, np.ndarray]
        ) -> tuple[Figure, Axes]:
    
    colors = plt.get_cmap("Set1").colors

    # Plot training results
    fig, ax = plt.subplots()

    plt.xlabel("outer training iterations")
    plt.ylabel("loss")
    
    plt.grid()

    for i, (name, loss_dict) in enumerate(kwargs.items()):
        for kind in ("train", "test"):
            loss_matrix = loss_dict[kind]
            n_rows, n_cols = loss_matrix.shape
            for col in range(n_cols):
                linestyle = "-" if kind == "train" else "--"
                label = f"{name} ({kind})" if n_cols == 1 else f"{name} ({kind}) #{col}"
                if n_rows == 1:
                    plt.axhline(y=loss_matrix[0,col], color=colors[i], linestyle=linestyle, label=label)
                else:
                    loss = loss_matrix[:,col]
                    iters = np.arange(1, len(loss)+1)
                    plt.loglog(iters, loss, color=colors[i], linestyle=linestyle, label=label)

    plt.legend(bbox_to_anchor=(1.00, 1))

    plt.show()

    return fig, ax


def plot_model_output(
    x: np.ndarray,
    y_noisy: np.ndarray,
    n_samples_plot: int = 100,
    **kwargs: Callable
    ) -> tuple[Figure, Axes]:
    """
    Plots underlying data, noisy data, and any model outputs.
    """
    data_color = (.8, .8, .8)
    data_alpha = 0.3
    colors = plt.get_cmap("Set1").colors

    # Sort data for cleaner line plots
    idx_sort = np.argsort(x)
    x, y_noisy = x[idx_sort], y_noisy[idx_sort]

    # Show only n_samples_plot samples of noisy data
    n_samples = x.shape[0]
    plot_interval = int(n_samples / n_samples_plot)
    idx_plot_data = np.arange(0, n_samples, plot_interval)
    
    fig, ax = plt.subplots()
    
    # Plot raw noisy data
    plt.scatter(x[idx_plot_data], y_noisy[idx_plot_data], 
                color=data_color, alpha=data_alpha, label="noisy data")

    for i, (model_name, predict_func) in enumerate(kwargs.items()):
        y = predict_func(x)
        plt.plot(x, y, color=colors[i], linestyle="-", label=f"{model_name}")

    plt.xlabel("input")
    plt.ylabel("output")
    plt.title("Comparison of model outputs")

    plt.legend(bbox_to_anchor=(1.00, 1))
    plt.grid()

    plt.show()

    return fig, ax


def plot_r2_scores(
    x: np.ndarray,
    y_true: np.ndarray,
    **kwargs: Callable
    ) -> tuple[Figure, Axes]:
    """
    Plots r-squared values of multiple models.
    """
    
    colors = plt.get_cmap("Set1").colors
    
    fig, ax = plt.subplots()

    for i, (model_name, predict_func) in enumerate(kwargs.items()):
        y_pred = predict_func(x)
        r2_score = metrics.r2_score(y_true, y_pred)
        plt.bar(i, r2_score, color=colors[i], label=f"{model_name}")

    ax.set_xticks(list(range(len(kwargs))))
    ax.set_xticklabels([model_name for model_name in kwargs])    

    plt.ylim(plt.ylim()[0], 1)

    plt.xlabel("model")
    plt.ylabel("R^2 score")
    plt.title("Comparison of model R^2 scores")

    plt.legend(bbox_to_anchor=(1.00, 1))
    plt.grid(axis="y")

    plt.show()

    return fig, ax


# %%
