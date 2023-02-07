# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.bearings_datasets import *
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
from domainbed.datasets import DataGenerate
from visdom import Visdom


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', default=r'/data/yfy/FD-data/RUL/RUL_ts', type=str)
    parser.add_argument('--dataset', type=str, default="RUL")
    parser.add_argument('--data_name', type=int, default=['PHM1'],
                        help='each index corresponds to a str in [PHM1, PHM2, PHM3, XJTU1, XJTU2]')
    parser.add_argument('--algorithm', type=str, default="FC")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,   # 0
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,  # 0
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=['XJTU1'])  # PU [8,9,10,11])  # TODO: for cv, modify the default to [0]
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)   # /scripts/train_output
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # viz = Visdom()
    # viz.line([0.], [1.], win='train_loss', opts=dict(title='train_loss'))

    # print("Environment:")
    # print("\tPython: {}".format(sys.version.split(" ")[0]))
    # print("\tPyTorch: {}".format(torch.__version__))
    # print("\tTorchvision: {}".format(torchvision.__version__))
    # print("\tCUDA: {}".format(torch.version.cuda))
    # print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))

    for trails in range(1):
        print("trails:", trails)
        print('Args:')
        for k, v in sorted(vars(args).items()):
            print('\t{}: {}'.format(k, v))

        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)   # 根据算法、数据集 设定超参数
        else:
            hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                      misc.seed_hash(args.hparams_seed, args.trial_seed))
        if args.hparams:
            hparams.update(json.loads(args.hparams))

        print('HParams:')
        for k, v in sorted(hparams.items()):
            print('\t{}: {}'.format(k, v))

        random.seed(args.seed)   # 固定影响随机数的参数
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if args.dataset != 'RUL':
            pass
        else:
            dataset = vars(datasets)[args.dataset](args.data_dir, args.data_name,
                                                   args.test_envs,
                                                   # hparams
                                                   )
            # train_x, train_y, tr_num = dataset.construct_domain()
            # train_x, train_y, tr_num = dataset.cwru_domain()
            # train_x, train_y, tr_num = dataset.pu_domain()
            # train_x, train_y, tr_num = dataset.get_domains()
            train_x, train_y, tr_num = dataset.load_domains()

            rate = args.holdout_fraction
            in_splits, te_splits, mtedatalist, out_splits = [], [], [], []

            # loader_index = tr_num if tr_num != 1 else len(train_x['0'])
            # if args.data_name == args.test_envs:
            tr_index = {i: [0, 1, 2, 3, 4, 5, 6] for i in args.data_name}  # range(len(train_x[i]))
            te_index = {i: [0, 1] for i in args.test_envs}  # TODO: datasets.py还是需要传入对应数据名

            for k, v in te_index.items():
                for i in v:
                    te_splits.append(DataGenerate(args=args, domain_data=train_x[k][i], labels=train_y[k][i]))

            for k, v in tr_index.items():
                for i in v:
                    tmpdatay = DataGenerate(args=args, domain_data=train_x[k][i], labels=train_y[k][i]).labels
                    l = len(tmpdatay)

                    lslist = np.arange(l)
                    stsplit = ms.StratifiedShuffleSplit(2, test_size=rate, train_size=1 - rate,
                                                        random_state=args.seed)  # 划分训练/验证集
                    stsplit.get_n_splits(lslist, tmpdatay)  # 返回n_splits
                    indextr, indexval = next(stsplit.split(lslist, tmpdatay))  # (数字索引，对应标签)，划分2组，交叉验证
                    np.random.seed(args.seed)
                    indexmte = np.random.permutation(indextr)

                    in_splits.append(DataGenerate(args=args, domain_data=train_x[k][i], labels=train_y[k][i],
                                                  indices=indextr))
                    mtedatalist.append(
                        DataGenerate(args=args, domain_data=train_x[k][i], labels=train_y[k][i],
                                     indices=indexmte))
                    out_splits.append(DataGenerate(args=args, domain_data=train_x[k][i], labels=train_y[k][i],
                                                   indices=indexval))

            train_loaders = [InfiniteDataLoader(  # 这是一个自编无限数据loader
                dataset=env,
                weights=None,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS)
                for env in in_splits]

            meta_test_loaders = [InfiniteDataLoader(  # 这是一个自编无限数据loader
                dataset=env,
                weights=None,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS)
                for env in mtedatalist]

            eval_loaders = [DataLoader(
                dataset=env,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS,
                drop_last=False,
                shuffle=False
            )
                for env in (in_splits + out_splits + te_splits)]

            test_loaders = [DataLoader(
                dataset=env,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS,
                drop_last=False,
                shuffle=False
            )
                for env in te_splits]
            uda_loaders = []

            eval_weights = [None] * (len(in_splits) + len(out_splits) + len(te_splits))
            eval_loader_names = ['env{}_in'.format(i)
                                 for i in range(len(in_splits))]
            eval_loader_names += ['env{}_out'.format(i)
                                  for i in range(len(out_splits))]
            eval_loader_names += ['env{}_test'.format(i)
                                  for i in range(len(te_splits))]
            domain_num = len(in_splits)
            # eval_loader_names += ['env{}_test'.format(i)
            #                       for i in args.test_envs]
            # domain_num = tr_num - len(args.test_envs)

        algorithm_class = algorithms.get_algorithm_class(args.algorithm)    # 算法实例化
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, domain_num, hparams)  # 算法初始化：模型、优化器

        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)

        algorithm.to(device)

        train_minibatches_iterator = zip(*train_loaders)
        uda_minibatches_iterator = zip(*uda_loaders)
        checkpoint_vals = collections.defaultdict(lambda: [])  # defaultdict用法 https://blog.csdn.net/weixin_42160653/article/details/80297894

        if args.dataset != "RUL":
            steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])  # 每个迭代中，最少的step数(样本数/batch_size)
        else:
            steps_per_epoch = min([len(env) / hparams['batch_size'] for env in in_splits])

        n_steps = args.steps or dataset.N_STEPS
        checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

        def save_checkpoint(filename):
            if args.skip_model_save:
                return
            save_dict = {
                "args": vars(args),
                "model_input_shape": dataset.input_shape,
                "model_num_classes": dataset.num_classes,
                "model_num_domains": domain_num,
                "model_hparams": hparams,
                "model_dict": algorithm.state_dict()
            }
            torch.save(save_dict, os.path.join(args.output_dir, filename))

        last_results_keys = None
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            minibatches_device = [(x.to(device), y.to(device))
                                  for x, y, ind in next(train_minibatches_iterator)]
            if args.task == "domain_adaptation":
                uda_device = [x.to(device)
                              for x, _ in next(uda_minibatches_iterator)]
            else:
                uda_device = None
            step_vals = algorithm.update(minibatches_device, uda_device)   # 返回的是loss 字典
            checkpoint_vals['step_time'].append(time.time() - step_start_time)

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            # visdom可视化
            # viz.line(Y=[step_vals['loss']], X=[step],
            #          win='train_loss',
            #          update=None if step == 0 else 'append')

            if (step % checkpoint_freq == 0) or (step == n_steps - 1):
                results = {
                    'step': step,
                    'epoch': step / steps_per_epoch,
                }

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                results['average_acc'], count = 0, 0
                evals = zip(eval_loader_names, eval_loaders, eval_weights)
                for name, loader, weights in evals:
                    acc = misc.accuracy(algorithm, loader, weights, device)
                    # TODO: draw the confusion matrix cm2
                    results[name + '_acc'] = acc
                    if 'test' in name:
                        results['average_acc'] += acc
                        count += 1

                results['average_acc'] /= count

                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)   # 返回给定设备device的张量所占用的GPU内存的最大值, 单位转换

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys],
                               colwidth=12)

                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })

                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                algorithm_dict = algorithm.state_dict()

                start_step = step + 1
                checkpoint_vals = collections.defaultdict(lambda: [])

                if args.save_model_every_checkpoint:
                    save_checkpoint(f'model_step{step}.pkl')

        save_checkpoint('model.pkl')

        with open(os.path.join(args.output_dir, 'done'), 'w') as f:
            f.write('done')

        start_step = 0
        algorithm_dict = None



