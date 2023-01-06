import argparse
import logging
import os
import sys
import torch
import random
import yaml
import imageio
from box import Box
from pathlib import Path

import data
import model
import runner
import callback


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)
    
    # Make experiment results deterministic.
    random.seed(config.main.random_seed)
    torch.manual_seed(random.getstate()[1][1])
    torch.cuda.manual_seed_all(random.getstate()[1][1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable.")
    device = torch.device(config.finetuner.kwargs.device)

    # datasets
    logging.info('Create the training and validation datasets.')
    base_dir = Path(config.dataset.kwargs.base_dir)
    config.dataset.kwargs.update(base_dir=base_dir, type='train')
    train_dataset = _get_instance(data.dataset, config.dataset)
    config.dataset.kwargs.update(base_dir=base_dir, type='valid')
    valid_dataset = _get_instance(data.dataset, config.dataset)

    # dataloaders
    logging.info('Create the training and validation dataloaders.')
    train_batch_size, valid_batch_size = config.dataloader.kwargs.pop('train_batch_size'), config.dataloader.kwargs.pop('valid_batch_size')
    config.dataloader.kwargs.update(collate_fn=None, batch_size=train_batch_size)
    train_dataloader = _get_instance(data.dataloader, config.dataloader, train_dataset)
    config.dataloader.kwargs.update(batch_size=valid_batch_size)
    valid_dataloader = _get_instance(data.dataloader, config.dataloader, valid_dataset)

    # model
    logging.info('Create the network architecture.')
    net = _get_instance(model.net, config.net)
    logging.info('Load the weights from {} and {}.'.format(config.main.encoder_path, config.main.decoder_path))
    encoder_dict = torch.load(config.main.encoder_path)['net']
    decoder_dict = torch.load(config.main.decoder_path)['net']
    model_state_dict = net.state_dict()

    for m in ['enc.', 'mean_map.', 'logstd_map.']:
        params = {k: v for k, v in encoder_dict.items() if m in k}
        model_state_dict.update(params)
    for m in ['dec.', 'cc.']:
        params = {k: v for k, v in decoder_dict.items() if m in k}
        model_state_dict.update(params)

    net.load_state_dict(model_state_dict)
    net = net.to(device)

    # optimizer
    logging.info('Create the optimizer.')
    optimizer = _get_instance(torch.optim, config.optimizer, net.get_model_params())
    logging.info('Create the learning rate scheduler.')
    lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer) if config.get('lr_scheduler') else None

    # logger & monitor
    logging.info('Create the logger.')
    config.logger.kwargs.update(log_dir=saved_dir / 'log', dummy_input=torch.randn(tuple(config.logger.kwargs.dummy_input)))
    logger = _get_instance(callback.loggers, config.logger)
    logging.info('Create the monitor.')
    config.monitor.kwargs.update(checkpoints_dir=saved_dir / 'checkpoints')
    monitor = _get_instance(callback.monitor, config.monitor)

    # finetuner
    logging.info('Create the finetuner.')
    kwargs = {'device': device,
                'train_dataloader': train_dataloader,
                'valid_dataloader': valid_dataloader,
                'train_dataset': train_dataset,
                'valid_dataset': valid_dataset,
                'net': net,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'logger': logger,
                'monitor': monitor}
    config.finetuner.kwargs.update(kwargs)
    finetuner = _get_instance(runner.finetuner, config.finetuner)

    # finetuning
    logging.info('Start finetuning.')
    finetuner.train()
    logging.info('End finetuning.')


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.
    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The script for training and testing.")
    parser.add_argument('-c', '--config_path', type=Path, help='The path of the config file.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    main(args)