#%%
from copy import deepcopy
from typing import Callable, Generator, Union

import numpy as np
from FL_toymodel.src.client import PolyLRClient
from FL_toymodel.src.utils.client import aggregate_client_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#%%
def uniform_overlapping_segments(
    x: np.ndarray,
    n_segments: int,
    overlap: float = 0.
    ) -> Callable:
    """
    Creates a function that returns a generator, which yields indices.
    These indices correspond to segments that span x entirely and
    which overlap a certain fraction at each side.
    """
    v_min, v_max = x.min(), x.max()
    D = v_max - v_min
    d = D / (n_segments - overlap * (n_segments-1))
    d_overlap = overlap * d
    def inner_generator() -> Generator[np.ndarray, None, None]:
        for i in range(n_segments):
            s_start = v_min if i == 0 else v_min + i * (d - d_overlap)
            s_end = v_max if i == n_segments-1 else s_start + d
            is_beyond_start = x >= s_start if i == 0 else x > s_start
            is_before_end = x <= s_end
            yield np.argwhere(is_beyond_start & is_before_end).flatten()
    return inner_generator


def uniform_overlapping_indices(
    x: np.ndarray,
    n_segments: int,
    overlap: float = 0.
    ) -> Callable:
    """
    Creates a function that returns a generator, which yields indices.
    The indices span x entirely, and have a certain degree of overlap.
    """
    n_samples = x.shape[0]
    x_replace = np.arange(n_samples)
    return uniform_overlapping_segments(x_replace, n_segments, overlap)


def random_overlapping_segments(
    x: np.ndarray,
    n_segments: int,
    segment_width_fraction: float
    ) -> Callable:
    """
    Creates a function that returns a generator, which yields indices.
    These indices correspond to segments of x,
    which have a width of segment_width_fraction times the total domain width
    and are uniformly randomly generated over the domain.
    """
    v_min, v_max = x.min(), x.max()
    segment_width = (v_max - v_min) * segment_width_fraction
    s_start_width = (v_max - segment_width) - v_min
    def inner_generator() -> Generator[np.ndarray, None, None]:
        for _ in range(n_segments):
            s_start = np.random.rand() * s_start_width + v_min
            s_end = s_start + segment_width
            is_beyond_start = x >= s_start
            is_before_end = x <= s_end
            yield np.argwhere(is_beyond_start & is_before_end).flatten()
    return inner_generator


def create_random_linear_regressor(n_inputs: int) -> LinearRegression:
    """
    Generate a random linear regression model.
    """
    linear_regression = LinearRegression(fit_intercept=False)
    linear_regression.coef_ = np.random.randn(n_inputs)
    linear_regression.intercept_ = 0
    return linear_regression


def initialize_clients(
    x: np.ndarray,
    y: np.ndarray,
    poly_degree: int,
    idx_generator: Callable,
    **client_kwargs: Union[None, int, float]
    ) -> list[PolyLRClient]:
    """
    Generates a list of clients with a certain polynomial features degree.
    Each client has the same, but randomized, linear model.
    Each client has access to a piece of the total data as determined by the
    idx_generator function.
    """
    idxs = idx_generator(x)
    poly_features = PolynomialFeatures(degree=poly_degree)
    random_linear_regressor = create_random_linear_regressor(poly_degree+1)
    clients = [PolyLRClient(
        poly_features, 
        random_linear_regressor, 
        x[idx], 
        y[idx], 
        **client_kwargs
        ) for idx in idxs]
    return clients


def initialize_centralized_client(
    clients: list[PolyLRClient],
    **client_kwargs: Union[None, int, float]
    ) -> PolyLRClient:
    """
    Aggregate a list of clients into a single client which has access to the same data (re-shuffled).
    """
    client_x, client_y = aggregate_client_data(clients)
    poly_features = clients[0]._polynomial_features
    linear_regressor = clients[0]._linear_regressor
    centralized_client = PolyLRClient(
        poly_features,
        linear_regressor,
        client_x,
        client_y,
        **client_kwargs
        )
    return centralized_client


#%%
