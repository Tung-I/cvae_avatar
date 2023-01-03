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


class DomainAdaptiveTrainer(BaseTrainer):
    """
    Trainer of Domain-Adaptive VAE
    Args:
        pretrained_enc: The pretrained encoder from Deep Avatar VAE
        train_dataset: The pytorch dataset of training 
        valid_dataset: The pytorch dataset of validation
        lambda_retar: The weight of retargeting loss for latent code mapping
        lambda_rec: The weight of image reconstruction loss
        lambda_kl: The weight of kl-divergence loss
    """ 
    def __init__(
        self,
        pretrained_enc,
        train_dataset, 
        valid_dataset,
        lambda_retar=0.1,
        lambda_rec=1.0,
        lambda_kl=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mse = nn.MSELoss()
        self.pretrained_enc = pretrained_enc
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset    
        self.lambda_retar = lambda_retar
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl 
    

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

            # parameters for denormalization
            vertmean = torch.tensor(dataset.vertmean, dtype=torch.float32).view((1, -1, 3))
            vertmean = vertmean.to(self.device)
            vertstd = dataset.vertstd
            texmean = torch.tensor(dataset.texmean).permute((2, 0, 1))[None, ...]
            texmean = texmean.to(self.device)
            texstd = dataset.texstd

            # training
            if mode == 'training':
                up_face, low_face, kl, mapped_mean, mapped_logstd = self.net(batch['up_face'], batch['low_face'])
                mean, logstd = self.pretrained_enc.encode(batch["avg_tex"], batch["aligned_verts"])
                retar_loss = (self.mse(mean, mapped_mean) + self.mse(logstd, mapped_logstd)) * 0.5
                rec_loss = (
                    torch.mean((up_face - batch["up_face"]) ** 2) + torch.mean((low_face - batch["low_face"]) ** 2)
                ) * (255**2) * 0.5
                total_loss = (
                    self.lambda_rec * rec_loss +
                    self.lambda_retar * retar_loss +
                    self.lambda_kl * kl
                )
                # loss backward
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimizer.step()

            # validation
            else:
                with torch.no_grad():
                    up_face, low_face, kl, mapped_mean, mapped_logstd  = self.net(batch['up_face'], batch['low_face'])
                    mean, logstd = self.pretrained_enc.encode(batch["avg_tex"], batch["aligned_verts"])
                    retar_loss = (self.mse(mean, mapped_mean) + self.mse(logstd, mapped_logstd)) * 0.5
                    rec_loss = (
                        torch.mean((up_face - batch["up_face"]) ** 2) + torch.mean((low_face - batch["low_face"]) ** 2)
                    ) * (255**2) * 0.5
                    total_loss = (
                        self.lambda_rec * rec_loss +
                        self.lambda_retar * retar_loss +
                        self.lambda_kl * kl
                    )

            self._update_log(log, batch_size, total_loss, [retar_loss, rec_loss, kl])
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))
        
        # logging
        for key in log:
            log[key] /= count

        up_face = torch.clamp(up_face*255, 0, 255)
        low_face = torch.clamp(low_face*255, 0, 255)
        output = {
            "up_face": batch["up_face"] * 255., 
            "pred_up_face": up_face,
            "low_face": batch["low_face"] * 255.,
            "pred_low_face": low_face
        }

        return log, output

    


    def _init_log(self):
        log = {}
        log['Loss'] = 0
        for loss_name in ['retar_loss', 'rec_loss', 'kl_loss']:
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
        loss_names = ['retar_loss', 'rec_loss', 'kl_loss']
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
        if checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.monitor = checkpoint['monitor']
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        self.np_random_seeds = checkpoint['np_random_seeds']
