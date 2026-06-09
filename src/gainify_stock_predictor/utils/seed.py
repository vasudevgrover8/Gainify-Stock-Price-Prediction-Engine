import random

import numpy as np

from gainify_stock_predictor.utils.constants import DEFAULT_SEED


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return

    tf.random.set_seed(seed)
