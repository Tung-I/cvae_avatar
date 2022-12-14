import logging
import random
import copy
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Callable, Sequence, Union, List

from runner.trainer import BaseTrainer
from runner.utils import Renderer


class DAVAETrainer(BaseTrainer):
    def __init__(
        self,
        tex_size,
        resolution,
        train_dataset, 
        valid_dataset,
        lambda_screen=1.0,
        lambda_tex=1.0,
        lambda_verts=1.0,
        lambda_kl=1e-2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tex_size = tex_size
        self.mse = nn.MSELoss()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.renderer = Renderer(self.device)
        self.resolution = resolution
        self.lambda_verts = lambda_verts
        self.lambda_tex = lambda_tex
        self.lambda_screen = lambda_screen
        self.lambda_kl = lambda_kl 
        self.optimizer_cc = optim.Adam(self.net.get_cc_params(), 3e-4, (0.9, 0.999))
    

    def train(self):
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k=self.num_epochs)

        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(self.np_random_seeds[self.epoch - 1])

            # Do training and validation.
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_output = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')
            valid_log, valid_output = self._run_epoch('validation')
            logging.info(f'Valid log: {valid_log}.')

            # Adjust the learning rate.
            if self.lr_scheduler is None:
                pass
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                self.lr_scheduler.step(valid_log['Loss'])
            else:
                self.lr_scheduler.step()

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_output, valid_log, valid_output)

            # Save the regular checkpoint.
            saved_path = self.monitor.is_saved(self.epoch)
            if saved_path:
                logging.info(f'Save the checkpoint to {saved_path}.')
                self.save(saved_path)

            # Save the best checkpoint.
            saved_path = self.monitor.is_best(valid_log)
            if saved_path:
                logging.info(f'Save the best checkpoint to {saved_path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                self.save(saved_path)
            else:
                logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_improved_count}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

            # Early stop.
            if self.monitor.is_early_stopped():
                logging.info('Early stopped.')
                break

            self.epoch +=1

        self.logger.close()


    def _run_epoch(self, mode: str):
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataset = self.train_dataset if mode == 'training' else self.valid_dataset
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
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
            loss_weight_mask = torch.tensor(dataset.loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float()
            loss_weight_mask = loss_weight_mask.to(self.device)

            if mode == 'training':
                pred_tex, pred_verts, kl = self.net(batch["avg_tex"], batch["aligned_verts"], batch["view"], cams=batch["cam"])
                # compute loss
                vert_loss = self.mse(pred_verts, batch["aligned_verts"])
                pred_verts = pred_verts * vertstd + vertmean
                pred_tex = (pred_tex * texstd + texmean) / 255.0
                gt_tex = (gt_tex * texstd + texmean) / 255.0
                loss_mask = loss_weight_mask.repeat(batch_size, 1, 1, 1)
                tex_loss = self.mse(pred_tex * batch["mask"], gt_tex * batch["mask"]) * (255**2) / (texstd**2)
                screen_mask, rast_out = self.renderer.render(
                    batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], loss_mask, self.resolution
                )
                pred_screen, rast_out = self.renderer.render(
                    batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], pred_tex, self.resolution
                )
                screen_loss = (
                    torch.mean((pred_screen - batch["photo"]) ** 2 * screen_mask)
                    * (255**2)
                    / (texstd**2)
                )
                total_loss = (
                    self.lambda_verts * vert_loss +
                    self.lambda_tex * tex_loss +
                    self.lambda_screen * screen_loss +
                    self.lambda_kl * kl
                )
                # loss backward
                self.optimizer.zero_grad()
                self.optimizer_cc.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimizer.step()
                self.optimizer_cc.step()
            else:
                with torch.no_grad():
                    pred_tex, pred_verts, kl = self.net(batch["avg_tex"], batch["aligned_verts"], batch["view"], cams=batch["cam"])
                    # compute loss
                    vert_loss = self.mse(pred_verts, batch["aligned_verts"])
                    pred_verts = pred_verts * vertstd + vertmean
                    pred_tex = (pred_tex * texstd + texmean) / 255.0
                    gt_tex = (gt_tex * texstd + texmean) / 255.0
                    loss_mask = loss_weight_mask.repeat(batch_size, 1, 1, 1)
                    tex_loss = self.mse(pred_tex * batch["mask"], gt_tex * batch["mask"]) * (255**2) / (texstd**2)
                    screen_mask, rast_out = self.renderer.render(
                        batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], loss_mask, self.resolution
                    )
                    pred_screen, rast_out = self.renderer.render(
                        batch["M"], pred_verts, batch["vert_ids"], batch["uvs"], batch["uv_ids"], pred_tex, self.resolution
                    )
                    screen_loss = (
                        torch.mean((pred_screen - batch["photo"]) ** 2 * screen_mask)
                        * (255**2)
                        / (texstd**2)
                    )
                    total_loss = (
                        self.lambda_verts * vert_loss +
                        self.lambda_tex * tex_loss +
                        self.lambda_screen * screen_loss +
                        self.lambda_kl * kl
                    )

            self._update_log(log, batch_size, total_loss, [vert_loss, tex_loss, screen_loss, kl])
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))
        

        for key in log:
            log[key] /= count

        # for logger
        gt_tex *= 255
        # gt_tex *= batch["mask"]
        pred_tex = torch.clamp(pred_tex*255, 0, 255)
        # pred_tex *= batch["mask"]

        gt_screen = batch["photo"] * 255
        pred_screen = torch.clamp(pred_screen*255, 0, 255)

        
        output = {
            "gt_tex": gt_tex, 
            "pred_tex": pred_tex,
            "gt_screen": gt_screen.permute(0, 3, 1, 2),
            "pred_screen": pred_screen.permute(0, 3, 1, 2)
        }

        # print(gt_tex.size())
        # print(pred_tex.size())
        # print(gt_screen.size())
        # print(pred_screen.size())
        # raise Exception(' ')

        

        return log, output

    


    def _init_log(self):
        log = {}
        log['Loss'] = 0
        for loss_name in ['vert_loss', 'tex_loss', 'screen_loss', 'kl_loss']:
            log[loss_name] = 0
        return log


    def _update_log(
        self,
        log: dict,
        batch_size: int,
        total_loss: torch.Tensor,
        losses: Sequence[torch.Tensor],
    ):
        log['Loss'] += total_loss.item() * batch_size
        loss_names = ['vert_loss', 'tex_loss', 'screen_loss', 'kl_loss']
        for name, loss in zip(loss_names, losses):
            log[name] += loss.item() * batch_size

    
    def _allocate_data(
        self,
        batch: dict
    ):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch


    def save(self, path):
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_cc': self.optimizer_cc.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor': self.monitor,
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'np_random_seeds': self.np_random_seeds
        }, path)


    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer_cc.load_state_dict(checkpoint['optimizer_cc'])
        if checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.monitor = checkpoint['monitor']
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        self.np_random_seeds = checkpoint['np_random_seeds']
