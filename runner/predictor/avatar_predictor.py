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


class DeepAvatarPredictor(BasePredictor):
    """
    Predictor of Deep-Avatar VAE
    Args:
        tex_size: Size of unwrapped textures
        resolution: Size of screen rendering
        test_dataset: The pytorch dataset of testing
        net: The well-trained deep avatar VAE
    """ 
    def __init__(
        self,
        tex_size,
        resolution,
        test_dataset, 
        net,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.net = net.to(device)
        self.tex_size = tex_size
        self.test_dataset = test_dataset
        self.renderer = Renderer(self.device)
        self.resolution = resolution
        self.avg_infer_time = None

    def predict(self):
        # Reset the numpy random seed.
        if self.np_random_seeds is None:
            self.np_random_seeds = 0
        np.random.seed(self.np_random_seeds)

        logging.info('Infer: {}/{}/{}'.format(
            self.test_dataset.target_individual,
            self.test_dataset.target_exp,
            self.test_dataset.target_cam
            ))

        self.net.eval()
        dataset = self.test_dataset 
        dataloader = self.test_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc='test')

        gt_frames = []
        pred_frames = []
        avgtex_frames = []
        total_infer_time = 0
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            batch_size, channel, height, width = batch["avg_tex"].shape
            gt_tex = batch["tex"]
            vertmean = torch.tensor(dataset.vertmean, dtype=torch.float32).view((1, -1, 3))
            vertmean = vertmean.to(self.device)
            vertstd = dataset.vertstd
            texmean = torch.tensor(dataset.texmean).permute((2, 0, 1))[None, ...]
            texmean = texmean.to(self.device)
            texstd = dataset.texstd

            time_flag = time.time()
            with torch.no_grad():
                pred_tex, pred_verts, kl = self.net(batch["avg_tex"], batch["aligned_verts"], batch["view"], cams=batch["cam"])
                pred_verts = pred_verts * vertstd + vertmean
                pred_tex = (pred_tex * texstd + texmean) / 255.0
                gt_tex = (gt_tex * texstd + texmean) / 255.0

                pred_screen, rast_out = self.renderer.render(
                    batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], pred_tex, self.resolution
                )

                # avg_tex = batch["avg_tex"]
                # avg_tex = (avg_tex * texstd + texmean) / 255.0
                # pred_screen, rast_out = self.renderer.render(
                #     batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], avg_tex, self.resolution
                # )


            if count > 0:
                total_infer_time += time.time() - time_flag

            # logging
            gt_tex *= 255
            pred_tex = torch.clamp(pred_tex*255, 0, 255)
            gt_screen = batch["photo"] * 255
            pred_screen = torch.clamp(pred_screen*255, 0, 255)
            gt_screen = gt_screen.squeeze().cpu().numpy().astype(np.uint8)
            pred_screen = pred_screen.squeeze().cpu().numpy().astype(np.uint8)
            gt_frames.append(gt_screen)
            pred_frames.append(pred_screen)
            
            avg_tex = batch["avg_tex"]
            # mask = avg_tex == 0
            avg_tex = (avg_tex * texstd + texmean)
            # avg_tex[mask] = 0.0
            avg_tex = avg_tex.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            avgtex_frames.append(np.flip(avg_tex, 0))

            count += 1

        self.avg_infer_time = total_infer_time / (len(dataloader) - 1)
        return gt_frames, pred_frames, avgtex_frames

    
    def _allocate_data(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch


    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        random.setstate(checkpoint['random_state'])
        self.np_random_seeds = checkpoint['np_random_seeds']

