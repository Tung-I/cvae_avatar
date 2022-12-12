import torch
import random
import copy
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """
    """
    def __init__(
        self,
        log_dir: str,
        dummy_input: torch.Tensor
    ):
        self.writer = SummaryWriter(log_dir)

    def write(
        self,
        epoch: int,
        train_log: dict,
        train_output: dict,
        valid_log: dict,
        valid_output: dict,
    ):
        self._add_scalars(epoch, train_log, valid_log)
        self._add_images(epoch, train_output, valid_output)
    
    def close(self):
        self.writer.close()

    def _add_scalars(
        self,
        epoch: int,
        train_log: dict,
        valid_log: dict
    ):
        for key in train_log:
            self.writer.add_scalars(key, {'train': train_log[key], 'valid': valid_log[key]}, epoch)

    def _add_images(
        self,
        epoch: int,
        train_output: dict,
        valid_output: dict
    ):
        raise NotImplementedError


class DAVAELogger(BaseLogger):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def _add_images(
        self,
        epoch: int,
        train_output: dict,
        valid_output: dict
    ):

        train_gt_tex = make_grid(train_output['gt_tex'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_pred_tex = make_grid(train_output['pred_tex'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_gt_screen = make_grid(train_output['gt_screen'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_pred_screen = make_grid(train_output['pred_screen'], nrow=1, normalize=True, scale_each=True, pad_value=1)

        valid_gt_tex = make_grid(valid_output['gt_tex'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_pred_tex = make_grid(valid_output['pred_tex'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_gt_screen = make_grid(valid_output['gt_screen'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_pred_screen = make_grid(valid_output['pred_screen'], nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid_tex = torch.cat((train_gt_tex, train_pred_tex), dim=-1)
        valid_grid_tex = torch.cat((valid_gt_tex, valid_pred_tex), dim=-1)
        train_grid_screen = torch.cat((train_gt_screen, train_pred_screen), dim=-1)
        valid_grid_screen = torch.cat((valid_gt_screen, valid_pred_screen), dim=-1)

        self.writer.add_image('train_tex', train_grid_tex, epoch)
        self.writer.add_image('valid_tex', valid_grid_tex, epoch)
        self.writer.add_image('train_screen', train_grid_screen, epoch)
        self.writer.add_image('valid_screen', valid_grid_screen, epoch)



class ImageLogger(BaseLogger):
    """The 2D image logger
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def _add_images(
        self,
        epoch: int,
        train_batch: dict,
        train_output: torch.Tensor,
        valid_batch: dict,
        valid_output: torch.Tensor
    ):

        train_img = make_grid(train_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_label = make_grid(train_batch['label'].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_pred = make_grid(train_output.float(), nrow=1, normalize=True, scale_each=True, pad_value=1)

        valid_img = make_grid(valid_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_label = make_grid(valid_batch['label'].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_pred = make_grid(valid_output.float(), nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat((train_img, train_label, train_pred), dim=-1)
        valid_grid = torch.cat((valid_img, valid_label, valid_pred), dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)







