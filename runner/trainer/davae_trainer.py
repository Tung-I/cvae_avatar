import torch
import logging
from tqdm import tqdm
import random
import copy
import numpy as np
import sys

from runner.trainer import BaseTrainer


class DAVAETrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train(self):
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k=self.num_epochs)

        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(self.np_random_seeds[self.epoch - 1])

            # Do training and validation.
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')
            valid_log, valid_batch, valid_outputs = self._run_epoch('validation')
            logging.info(f'Valid log: {valid_log}.')

            # Adjust the learning rate.
            if self.lr_scheduler is None:
                pass
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                self.lr_scheduler.step(valid_log['Loss'])
            else:
                self.lr_scheduler.step()

            # Record the log information and visualization.
            self.logger.write(self.epoch, train_log, train_batch, train_outputs,
                              valid_log, valid_batch, valid_outputs)

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
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)

            verts, view, frontview, targetview = self._get_inputs_targets(batch)

            if mode == 'training':
                pred_tex, pred_verts, kl = self.net(frontview, verts, view)
                
                pred_tex = (pred_tex * batch['texstd'] + batch['texmean']) / 255.0
                targetview = (targetview * batch['texstd'] + batch['texmean']) / 255.0
                frontview = (frontview * batch['texstd'] + batch['texmean']) / 255.0

                # print(pred_tex.max())
                # print(targetview.max())
                # print(batch['texmean'].max())
                # print(batch['texstd'].max())

                # pred_tex *= batch['mask']
                # targetview *= batch['mask']
                # print(((pred_tex - targetview)**2).mean())

                losses = self._compute_losses(pred_verts, verts, pred_tex, targetview, batch['texstd'], kl)
                loss = (torch.stack(losses) * self.loss_weights).sum()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    pred_tex, pred_verts, kl = self.net(frontview, verts, view)
                    pred_tex = (pred_tex * batch['texstd'] + batch['texmean']) / 255.0
                    targetview = (targetview * batch['texstd'] + batch['texmean']) / 255.0
                    frontview = (frontview * batch['texstd'] + batch['texmean']) / 255.0
                    # pred_tex *= batch['mask']
                    # targetview *= batch['mask']


                    losses = self._compute_losses(pred_verts, verts, pred_tex, targetview, batch['texstd'], kl)
                    loss = (torch.stack(losses) * self.loss_weights).sum()

            metrics =  self._compute_metrics(pred_tex, targetview)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, {'image': frontview, 'label': targetview}, pred_tex

    
    def _compute_losses(
        self,
        pred_verts: torch.Tensor,
        verts: torch.Tensor,
        pred_tex: torch.Tensor,
        targetview: torch.Tensor,
        texstd: torch.Tensor,
        kl: torch.Tensor
    ):
        losses = []
        # print(((pred_tex - targetview)**2).mean())
        for loss_fn in self.loss_fns:
            if loss_fn.__class__.__name__== 'MeshLoss':
                losses.append(loss_fn(pred_verts, verts))
            elif loss_fn.__class__.__name__== 'ScreenLoss':
                losses.append(loss_fn(pred_tex, targetview, texstd))
            elif loss_fn.__class__.__name__== 'KLLoss':
                losses.append(kl)
            else:
                raise Exception('Unknown loss function: {}'.format(loss_fn.__class__.__name__))
        return losses


    def _compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ):
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics


    def _get_inputs_targets(
        self,
        batch: dict
    ):
        # return batch['verts'], batch['view'], batch['frontview'], batch['targetview'], batch['mask']
        return batch['verts'], batch['view'], batch['frontview'], batch['targetview']

    
    def _allocate_data(
        self,
        batch: dict
    ):
        batch['verts'] = batch['verts'].to(self.device)
        batch['view'] = batch['view'].to(self.device)
        batch['frontview'] = batch['frontview'].to(self.device)
        batch['targetview'] = batch['targetview'].to(self.device)
        batch['texmean'] = batch['texmean'].to(self.device)
        batch['texstd'] = batch['texstd'].to(self.device)
        # batch['mask'] = batch['mask'].to(self.device)
        return batch
