#%%
import collections.abc
from typing import Iterable


#%%
def weighted_average(xs: Iterable, ws: Iterable) -> float:
    xw_sum, w_sum = 0, sum(ws)
    for x, w in zip(xs, ws):
        xw_sum += x * w
    return xw_sum / w_sum


def update_nested_dict(
    dict_init: collections.abc.Mapping, 
    dict_new: collections.abc.Mapping
) -> collections.abc.Mapping:
    """
    Update a nested dictionary, overwriting the original.
    """

    for k, v in dict_new.items():
        if isinstance(v, collections.abc.Mapping):
            dict_init[k] = update_nested_dict(dict_init.get(k, {}), v)
        else:
            dict_init[k] = v
    return dict_init


# %%
