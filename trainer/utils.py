# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import torch

from functools import wraps
from numpy.random import seed as np_seed
from numpy.random import get_state as np_get_state
from numpy.random import set_state as np_set_state
from pytorch3d.implicitron.tools import model_io
from random import seed as rand_seed
from random import getstate as rand_get_state
from random import setstate as rand_set_state
from torch import manual_seed as torch_seed
from torch import get_rng_state as torch_get_state
from torch import set_rng_state as torch_set_state
from pathlib import Path

def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def get_optimizer_discriminator_path(fl: str):
    opt_path = model_io.get_optimizer_path(fl)
    opt_disc_path = opt_path.replace("_opt.pth", "_opt_disc.pth")
    assert opt_disc_path != opt_path
    return opt_disc_path

def path_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class use_seed:
    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int) and seed >= 0
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.rand_state = rand_get_state()
            self.np_state = np_get_state()
            self.torch_state = torch_get_state()
            rand_seed(self.seed)
            np_seed(self.seed)
            torch_seed(self.seed)
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            rand_set_state(self.rand_state)
            np_set_state(self.np_state)
            torch_set_state(self.torch_state)

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop("seed", None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper
