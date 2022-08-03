#%%
from typing import Callable, Union

import numpy as np


#%%
def generate_polynomial_weights(v_min: float, v_max: float, degree: int) -> np.ndarray:
    """
    Generates a polynomial of certain degree, which has all of its roots lie between v_min and v_max.
    Coefficients of higher degrees are at the start of the array.
    """
    roots = np.random.rand(degree) * (v_max - v_min) + v_min
    C = np.ones(1)
    for root in roots:
        c = np.zeros(degree+1)
        c[-2:] = np.array([1., -root])
        C = np.convolve(C, c)[-(degree+1):]
    if np.random.rand() > .5:
        C *= -1
    return C[::-1]


def sum_underlying_components(*components: Callable) -> Callable:
    """
    Composes a set of functions by summing their individual outputs.
    """
    return lambda x: sum(component(x) for component in components)


def generate_data(
    input_range: tuple[Union[int, float], Union[int, float]],
    n_samples: int,
    underlying_model: Callable,
    noise_std: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates noisy data based on an independent variable x samples uniformly within some input range.
    The dependent variable is based on same underlying model plus gaussian noise.
    """
    x = np.random.rand(n_samples) * (input_range[1] - input_range[0]) + input_range[0]
    y_actual = underlying_model(x)
    y_noisy = y_actual + np.random.randn(*x.shape) * noise_std
    return x, y_actual, y_noisy


#%%
