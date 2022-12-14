# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict, OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if len(input_shape) == 2:
            self.featurizer = networks.CNN(pretrained=False, in_channel=input_shape[0], out_channel=num_classes)
            self.classifier = networks.classifier_homo(self.featurizer.n_outputs, num_classes)
        else:
            self.featurizer = networks.Featurizer(input_shape, self.hparams)  # (224,224): ResNet50, output_size=2048
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'])  # 非线性分类器的指示参数为False，就是一层nn.Linear

        self.network = nn.Sequential(self.featurizer, self.classifier)  # 连接起来
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)   # ARM在predict处不同，输出也不同

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class DROMMD(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DROMMD, self).__init__(input_shape, num_classes, num_domains,
                                     hparams)
        self.register_buffer("q", torch.Tensor())   # weights

        self.kernel_type = "mean_cov"

        self.sd_reg = hparams["sd_reg"]

    def mmd(self, x, y):
        mean_x = x.mean(0, keepdim=True)  # 均值。类似求prototype
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x  # 样本-均值
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)  # 协方差, x y代表两个域的数据
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)  # 每列代表一个特征，表示一个随机变量。求列间特征的协方差

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update_coral(self, minibatches, unlabeled=None):
        penalty = 0
        nmb = len(minibatches)

        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        features = [self.featurizer(xi) for xi, _ in minibatches]   # 特征
        classifs = [self.classifier(fi) for fi in features]   # 预测类别
        targets = [yi for _, yi in minibatches]   # ground truth label

        for i in range(nmb):
            # objective += F.cross_entropy(classifs[i], targets[i])
            losses[i] = F.cross_entropy(classifs[i], targets[i])
            self.q[i] *= (self.hparams["groupdro_eta"] * losses[i].data).exp()
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        self.q /= self.q.sum()  # 权重归一化
        objective = torch.dot(losses, self.q)

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}

    def update_sd(self, minibatches, unlabeled=None):
        nmb = len(minibatches)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        features = [self.featurizer(xi) for xi, _ in minibatches]  # 特征
        classifs = [self.classifier(fi) for fi in features]  # 预测类别
        targets = [yi for _, yi in minibatches]  # ground truth label

        for i in range(nmb):
            # objective += F.cross_entropy(classifs[i], targets[i])
            losses[i] = F.cross_entropy(classifs[i], targets[i])
            self.q[i] *= (self.hparams["groupdro_eta"] * losses[i].data).exp()

        self.q /= self.q.sum()  # 权重归一化
        loss = torch.dot(losses, self.q)

        all_p = torch.cat(classifs)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):   # 返回num_domain个pairs，i、j为随机匹配的两域的batch
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class MLDG(DROMMD):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update_mixup(self, minibatches, unlabeled=None):
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj

            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            # inner_obj = F.cross_entropy(inner_net(xi), yi)
            inner_obj = F.cross_entropy(inner_net(x), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # objective += inner_obj.item()
            objective += lam * inner_obj.item()

            # loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            loss_inner_j = F.cross_entropy(inner_net(x), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                                         allow_unused=True)

            # objective += (self.hparams['mldg_beta'] * loss_inner_j).item()
            objective += (1 - lam) * (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    def update_mmd(self, minibatches, unlabeled=None):
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # lam = np.random.beta(self.hparams["mixup_alpha"],
            #                      self.hparams["mixup_alpha"])

            # x = lam * xi + (1 - lam) * xj

            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            features_i = inner_net(xi)
            inner_obj = F.cross_entropy(features_i, yi)
            # inner_obj = F.cross_entropy(inner_net(x), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            objective += inner_obj.item()
            # objective += lam * inner_obj.item()

            features_j = inner_net(xj)
            loss_inner_j = F.cross_entropy(features_j, yj)
            # loss_inner_j = F.cross_entropy(inner_net(x), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                                         allow_unused=True)

            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()
            # objective += (1 - lam) * (self.hparams['mldg_beta'] * loss_inner_j).item()

            objective += self.mmd(features_i, features_j)

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}
