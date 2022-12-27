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
    device = torch.device(config.predictor.kwargs.device)

    # datasets
    logging.info('Create the datasets.')
    base_dir = Path(config.dataset.kwargs.base_dir)
    config.dataset.kwargs.update(base_dir=base_dir, type='test')
    test_dataset = _get_instance(data.dataset, config.dataset)
    n_cameras = len(test_dataset.cameras)

    # dataloaders
    logging.info('Create the dataloader.')
    batch_size = config.dataloader.kwargs.pop('batch_size')
    config.dataloader.kwargs.update(collate_fn=None, batch_size=batch_size)
    test_dataloader = _get_instance(data.dataloader, config.dataloader, test_dataset)

    # model
    logging.info('Create the network architecture.')
    config.net.kwargs.update(n_cams=n_cameras)
    net = _get_instance(model.net, config.net)

    # predictor
    logging.info('Create the predictor.')
    kwargs = {'device': device,
                'test_dataloader': test_dataloader,
                'test_dataset': test_dataset,
                'net': net
            }
    config.predictor.kwargs.update(kwargs)
    predictor = _get_instance(runner.predictor, config.predictor)

    # load model
    loaded_path = config.main.get('loaded_path')
    logging.info(f'Load the previous checkpoint from "{loaded_path}".')
    predictor.load(Path(loaded_path))

    gt_frames, pred_frames, avgtex_frames = predictor.predict()
    print("Inference speed: {}".format(predictor.avg_infer_time))
    logging.info('End inference.')

    # save video
    save_path = "{}/{}".format(saved_dir, 'gt.mp4')
    imageio.mimwrite(save_path, gt_frames, fps=30, quality=8)
    save_path = "{}/{}".format(saved_dir, 'output.mp4')
    imageio.mimwrite(save_path, pred_frames, fps=30, quality=8)
    save_path = "{}/{}".format(saved_dir, 'input.mp4')
    imageio.mimwrite(save_path, avgtex_frames, fps=30, quality=8)


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