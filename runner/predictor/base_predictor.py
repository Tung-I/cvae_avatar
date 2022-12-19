import torch
import logging
from tqdm import tqdm
import random
import copy
import numpy as np
from pathlib import Path
from typing import Callable, Sequence, Union, List, Dict

import callback


class BasePredictor:
    def __init__(
        self,
        device: torch.device,
        test_dataloader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
    ):
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        self.np_random_seeds = None

    def predict(self):
        raise NotImplementedError

    def _allocate_data(
        self,
        batch: dict
    ):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

