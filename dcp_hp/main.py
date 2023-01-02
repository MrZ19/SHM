"""
@Time    :
@Author  : Zhiyuan Zhang & Jiadai Sun
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
from init_params import *
import os
import torch
import torch.nn as nn
from data import ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import HPNet
from framework import train, test


def fixed_random_seed(seed):
    # Fixed random seed to guarante reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = params_init()
    fixed_random_seed(args.seed)
    init_args(args)

    # Used to record the processing
    boardio = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, args.exp_name))
    textio = IOStream(os.path.join(args.checkpoint_dir, args.exp_name, 'run.log'))
    textio.cprint(str(args))

    # Load ModelNet40 dataset frome ./data/modelnet40_ply_hdf5_2048
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                             num_subsampled_points=args.n_subsampled_points,
                                             partition='train', gaussian_noise=args.gaussian_noise,
                                             unseen=args.unseen, factor=args.rot_factor),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                            num_subsampled_points=args.n_subsampled_points,
                                            partition='test', gaussian_noise=args.gaussian_noise,
                                            unseen=args.unseen, factor=args.rot_factor),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    else:
        raise Exception("not implemented, please check your args.dataset")

    # Load the network model to use
    if args.model == 'HPNet':
        net = HPNet(args).cuda()

        # To determine whether to load the model file
        # in evaluate | resume (continue training) | finetune etc.
        if args.eval or args.resume or args.finetune:
            model_path = os.path.join(args.checkpoint_dir, args.exp_name, 'models/model.best.t7') \
                if args.model_path is '' else args.model_path
            print("model path: {}".format(model_path))
            if not os.path.exists(model_path):
                print("can't find pretrained model file, please check your dir")
                return
            # load model params
            net.load_state_dict(torch.load(model_path), strict=False)

        # To determine whether to use multiple GPUs
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented, please check your args.model')

    # To determine whether to choose training, testing, or finetune
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        if args.finetune:
            print("Start finetune...")
        train(args, net, train_loader, test_loader, boardio, textio)

    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
