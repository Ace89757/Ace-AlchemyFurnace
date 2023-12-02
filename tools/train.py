# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import os
import time
import logging
import argparse

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.config import Config, DictAction

from alchemy.utils import format_datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--task', default='det2d', help='model task, Currently supports "det2d ", "mono3d"')
    parser.add_argument('--prefix', default='baseline', help='The prefix of the task.')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint.')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model.')
    parser.add_argument('--amp', action='store_true', help='enable automatic-mixed-precision training')
    parser.add_argument('--cfg-options', 
                        nargs='+', 
                        action=DictAction, 
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. '
                             'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                             'Note that the quotation marks are necessary and that no white space is allowed.')

    parser.add_argument('--distributed', action='store_true', help='distributed training.')
    parser.add_argument('--auto-scale-lr', action='store_true', default=True, help='enable automatically scaling LR.')

    # When using PyTorch version >= 2.0.0, the "torch.distributed.launch" will pass the "--local-rank" parameter to "tools/train.py" instead of "--local_rank".
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # 统计cpu时间, 单位s
    start_training = time.perf_counter()

    # args
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    
    # launcher
    if args.distributed:
        cfg.launcher = 'pytorch'
    else:
        cfg.launcher = 'none'
    
    # merge args
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work-dir
    if args.work_dir is not None:
        cfg.work_dir = os.path.join(args.work_dir, 'train', args.task, args.prefix)
    else:
        cfg.work_dir = os.path.join('./work_dirs', 'train', args.task, args.prefix)

  # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP training is already enabled in your config.', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (f'`--amp` is only supported when the optimizer wrapper type is `OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if ('auto_scale_lr' in cfg) and ('enable' in cfg.auto_scale_lr) and ('base_batch_size' in cfg.auto_scale_lr):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or "auto_scale_lr.enable" or "auto_scale_lr.base_batch_size" in your configuration file.')

    # resume & load from
    cfg.load_from = args.pretrained

    # build the runner from config
    runner = Runner.from_cfg(cfg)  # build the default runner

    # model info
    print_log(runner.model, logger='current', level=logging.INFO)

    # start training
    runner.train()
    
    # time
    print_log(
        f'The training took {format_datetime(time.perf_counter() - start_training)}', 
        logger='current', 
        level=logging.INFO
        )

if __name__ == '__main__':
    main()