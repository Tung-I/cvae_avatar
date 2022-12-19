import torch
import logging
from tqdm import tqdm
import random
import copy
import numpy as np
from pathlib import Path
from typing import Callable, Sequence, Union, List, Dict

import callback


class BaseTrainer:
    def __init__(
        self,
        device: torch.device,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        optimizer: torch.optim,
        lr_scheduler: torch.optim,
        logger: callback.BaseLogger, 
        monitor: callback.Monitor,
        num_epochs: int
    ):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net.to(device)
        # self.loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
        # self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        # self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]
        self.optimizer = optimizer

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
            raise NotImplementedError('Do not support torch.optim.lr_scheduler.CyclicLR scheduler yet.')
        self.lr_scheduler = lr_scheduler

        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.epoch = 1
        self.np_random_seeds = None


    def _init_log(self):
        raise NotImplementedError


    def _update_log(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError


    def _run_epoch(
        self,
        mode: str
    ):
        raise NotImplementedError
    
    def _allocate_data(
        self,
        batch: dict
    ):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


    def load(self, path):
        raise NotImplementedError

