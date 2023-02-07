# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)   # 设置随机数种子
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))   # choice返回一个 列表，元组或字符串的 随机项。
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', True,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        # _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))
        _hparam('meta_lr', 0.75, lambda r: r.choice([0.05, 0.1, 0.5]))   # for '2' -> CPUMP

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))  # 1e2
        _hparam('irm_penalty_anneal_iters', 500,  # 500
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        # _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))
        _hparam('mixup_alpha', 0.15, lambda r: 10 ** r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        # _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))
        _hparam('groupdro_eta', 6e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        # _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('mmd_gamma', 1.5, lambda r: 10 ** r.uniform(-1, 1))  # for '1/2'-> E
        # _hparam('mmd_gamma', 2, lambda r: 10 ** r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 0.2733254999483027, lambda r: 10**r.uniform(-1, 1))  # 1.

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        # _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))
        _hparam('penalty', 0.1, lambda r: 10 ** r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))   # 10

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        # _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('penalty_anneal_iters', 150, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        # _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('iters', 70, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm == "Transfer":
        # _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        # _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        # _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        # _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        # _hparam('gda', False, lambda r: True)
        # _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        # _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

        # for '2' -> MFPT
        # _hparam('t_lambda', 2.5, lambda r: 10 ** r.uniform(-2, 1))   # best 2.5// 500 50 update  2.75 // 400 50
        # _hparam('delta', 2, lambda r: r.uniform(0.1, 3.0))
        # _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        # _hparam('weight_decay_d', 8e-4, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('gda', False, lambda r: True)
        # _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        # _hparam('lr_d', 1e-4, lambda r: 10 ** r.uniform(-4.5, -2.5))
        # _hparam('lr_iter', 10000, lambda r: int(r.choice([700, 750, 800])))

        # for '1' -> MFPT
        # _hparam('t_lambda', 2.5, lambda r: 10 ** r.uniform(-2, 1))  # best 2.5// 500 50 update  2.75 // 400 50
        # _hparam('delta', 2, lambda r: r.uniform(0.1, 3.0))
        # _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        # _hparam('weight_decay_d', 8e-4, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('gda', False, lambda r: True)
        # _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        # _hparam('lr_d', 1e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
        # _hparam('lr_iter', 10000, lambda r: int(r.choice([700, 750, 800])))

        # for '2' -> CPUMP
        # _hparam('t_lambda', 0.8, lambda r: 10 ** r.uniform(-2, 1))  # best 2.5// 500 50 update  2.75 // 400 50
        # _hparam('delta', 2, lambda r: r.uniform(0.1, 3.0))
        # _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        # _hparam('weight_decay_d', 8e-4, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('gda', False, lambda r: True)
        # _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        # _hparam('lr_d', 5e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
        # _hparam('lr_iter', 675, lambda r: int(r.choice([700, 750, 800])))  # 651  675

        # '4'加强（A1 B1）-> MFPT
        # _hparam('t_lambda', 2, lambda r: 10 ** r.uniform(-2, 1))  # best 2.5// 500 50 update  2.75 // 400 50
        # _hparam('delta', 2, lambda r: r.uniform(0.1, 3.0))
        # _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        # _hparam('weight_decay_d', 8e-4, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('gda', False, lambda r: True)
        # _hparam('lr_d', 1e-5, lambda r: 10 ** r.uniform(-6.5, -2.5))
        # _hparam('lr_iter', 2001, lambda r: int(r.choice([1000, 1750, 1800])))

        _hparam('t_lambda', 0.8, lambda r: 10 ** r.uniform(-2, 1))  # best 2.5// 500 50 update  2.75 // 400 50
        _hparam('delta', 2, lambda r: r.uniform(0.1, 3.0))    # 2e-8
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 8e-4, lambda r: 10 ** r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('lr_d', 1e-5, lambda r: 10 ** r.uniform(-6.5, -2.5))
        _hparam('lr_iter', 2001, lambda r: int(r.choice([1000, 1750, 1800])))

    elif algorithm == 'DROMMD':
        _hparam('mmd_gamma', 1.5, lambda r: 10 ** r.uniform(-1, 1))
        _hparam('groupdro_eta', 6e-2, lambda r: 10 ** r.uniform(-3, -1))
        _hparam('sd_reg', 0.1, lambda r: 10 ** r.uniform(-5, -1))

    elif algorithm == 'FC':
        # maybe CWRU
        # _hparam('meta_step', 1, lambda r: int(r.choice([700, 750, 800])))
        # _hparam('beta', 0.5, lambda r: 10 ** r.uniform(-5, 1))
        # _hparam('lr_omega', 7e-3, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('heldout_p', 100, lambda r: 10 ** r.uniform(-3, 3))

        # PU data
        # _hparam('meta_step', 1, lambda r: int(r.choice([700, 750, 800])))
        # _hparam('beta', 1, lambda r: 10 ** r.uniform(-5, 1))
        # _hparam('lr_omega', 5e-3, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('heldout_p', 100, lambda r: 10 ** r.uniform(-3, 3))

       # RUL data
        _hparam('meta_step', 1, lambda r: int(r.choice([700, 750, 800])))
        _hparam('beta', 1, lambda r: 10 ** r.uniform(-5, 1))
        _hparam('lr_omega', 5e-3, lambda r: 10 ** r.uniform(-6, -2))
        _hparam('heldout_p', 100, lambda r: 10 ** r.uniform(-3, 3))

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        # _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        # _hparam('lr', 5e-4, lambda r: 10 ** r.uniform(-5.5, -3.5))   # 5e-4  C+P -> M good
        # _hparam('lr', 1e-5, lambda r: 10 ** r.uniform(-5.5, -3.5))  # C+P//'2' -> CP good GroupDRO相关算法
        _hparam('lr', 2.5e-5, lambda r: 10 ** r.uniform(-5.5, -3.5))  # 5e-5, 1e-5 0.80 '4'/all
        # _hparam('lr', 1e-4, lambda r: 10 ** r.uniform(-5.5, -3.5))  # FC pu data
        # _hparam('lr', 2.5e-4, lambda r: 10 ** r.uniform(-5.5, -3.5))  # FC cwru
        # _hparam('lr', 2.5e-5, lambda r: 10 ** r.uniform(-5.5, -3.5))
        # _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        # _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('weight_decay', 8e-4, lambda r: 10 ** r.uniform(-6, -2))
        # _hparam('weight_decay', 2.812885384798702e-06, lambda r: 10 ** r.uniform(-6, -2))   # for all'4'

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
