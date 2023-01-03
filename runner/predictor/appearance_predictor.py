import logging
import random
import copy
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Callable, Sequence, Union, List

from runner.predictor import BasePredictor
from runner.utils import Renderer


class DeepAppearancePredictor(BasePredictor):
    """
    Predictor of Deep Appearance Models
    Args:
        resolution: Size of screen rendering
        test_dataset: The pytorch dataset of testing
        net: The well-trained deep appearance model
    """ 
    def __init__(
        self,
        resolution,
        test_dataset,
        net, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.test_dataset = test_dataset
        self.net = net.to(device)
        self.renderer = Renderer(self.device)
        self.resolution = resolution
        self.avg_infer_time = None

    def predict(self):
        # Reset the numpy random seed.
        if self.np_random_seeds is None:
            self.np_random_seeds = 0
        np.random.seed(self.np_random_seeds)

        self.net.eval()
        dataset = self.test_dataset 
        dataloader = self.test_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc='test')

        gt_frames = []
        pred_frames = []
        upface_frames = []
        lowface_frames = []
        total_infer_time = 0
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            batch_size = batch["photo"].shape[0]
            gt_tex = batch["tex"]
            vertmean = torch.tensor(dataset.vertmean, dtype=torch.float32).view((1, -1, 3))
            vertmean = vertmean.to(self.device)
            vertstd = dataset.vertstd
            texmean = torch.tensor(dataset.texmean).permute((2, 0, 1))[None, ...]
            texmean = texmean.to(self.device)
            texstd = dataset.texstd

            time_flag = time.time()
            with torch.no_grad():
                pred_tex, pred_verts, kl  = self.net(batch['up_face'], batch['low_face'], batch["view"], cams=batch["cam"])
   
                pred_verts = pred_verts * vertstd + vertmean
                pred_tex = (pred_tex * texstd + texmean) / 255.0
                gt_tex = (gt_tex * texstd + texmean) / 255.0

                pred_screen, rast_out = self.renderer.render(
                    batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], pred_tex, self.resolution
                )

            if count > 0:
                total_infer_time += time.time() - time_flag

            # save the gt_screen & pred_screen
            pred_tex = torch.clamp(pred_tex*255, 0, 255)
            gt_screen = batch["photo"] * 255
            pred_screen = torch.clamp(pred_screen*255, 0, 255)
            gt_screen = gt_screen.squeeze().cpu().numpy().astype(np.uint8)
            pred_screen = pred_screen.squeeze().cpu().numpy().astype(np.uint8)
            gt_frames.append(gt_screen)
            pred_frames.append(pred_screen)
            
            # save the input
            up_face = batch['up_face'] * 255
            up_face = up_face.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            upface_frames.append(up_face)
            low_face = batch['low_face'] * 255
            low_face = low_face.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            lowface_frames.append(low_face)

            count += 1

        self.avg_infer_time = total_infer_time / (len(dataloader) - 1)
        return gt_frames, pred_frames, upface_frames, lowface_frames

    
    def _allocate_data(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch
