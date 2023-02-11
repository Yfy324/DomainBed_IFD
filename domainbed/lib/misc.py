# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as F
import faiss
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, accuracy_score
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data  # 更新参数
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)  # 返回校正后的值
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()   # 保存未校正的值

        self._updates += 1
        return ema_dict_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)    # 哈希函数 hashlib:对二进制进行加密的(“编码” encode: 将str类型转换成bytes类型).获取加密后的16进制字符串 -- int十进制
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


class Critic:

    def evaluate(self, inlier_scores, outlier_scores):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError


class Fpr(Critic):
    def __init__(self, device, recall_level=0.95):
        super().__init__()
        self.recall_level = recall_level
        self.device = device

    def get_name(self):
        return 'FPR(' + str(self.recall_level * 100) + ')'

    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)  # 按行累加
        expected = np.sum(arr, dtype=np.float64)  # 所有元素求和
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                               'its last element does not correspond to sum')
        return out

    def fpr_and_fdr_at_recall(self, y_pred, y_true, y_score, recall_level, pos_label=None):
        # fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)
        locations = y_score
        aupr = average_precision_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred)
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                         np.array_equal(classes, [-1, 1]) or
                         np.array_equal(classes, [0]) or
                         np.array_equal(classes, [-1]) or
                         np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 0.   # 原pos为inlier/1，这里outlier样本为1 -- 需要修改为0.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]  # 返回由小到大排序后对应索引[::-1] --> 由大到小
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract  -- 对应第一行代码，提取不同的分数值对应索引 待定为阈值
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]  # diff(axis=-1): 后一列 - 前一列的值. where: 提取后小于前的索引
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]  # 保持维度，相当于把20000-1 加到distinct后面/加一行

        # accumulate the true positives with decreasing threshold： threshold是预测/变化的索引  tps是表示真正T/F变化的衡量
        tps = self.stable_cumsum(y_true)[threshold_idxs]  # 某个阈值下，正确识别inlier的个数
        fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing 某阈值下，错将out识别inlier的个数

        thresholds = y_score[threshold_idxs]  # 不同阈值

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1] 从15119索引开始倒序向前
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]  # 相同的softmax突变的地方 找fps、tps、对应的预测分

        cutoff = np.argmin(np.abs(recall - recall_level))  # 找到recall大于0.95对应的索引
        location = np.where(locations < thresholds[cutoff])[0]

        return location, aupr, auroc, fps[cutoff] / (np.sum(np.logical_not(y_true)))  # fps[cutoff]/(fps[cutoff] + tps[cutoff]) 计算recall=0.95是对应的fps

    def get_score(self, model_output):
        to_np = lambda x: x.data.cpu().numpy()
        score = to_np(F.softmax(model_output, dim=1))  # 按行softmax，保证行和为1 (第1维度进行归一化)
        score = np.max(score, axis=1)  # 保存的是每个预测最大的softmax score，而不是预测的标签
        return score

    def evaluate(self, network, loader):
        network.eval()
        with torch.no_grad():
            all_x = torch.cat([x for x, y, ind in loader]).to(self.device)
            all_y = torch.cat([y for x, y, ind in loader]).to(self.device)
            all_p = network.predict(all_x)
            y_true = all_y.cpu().numpy()
            y_pred = np.argmax(all_p.cpu().numpy(), axis=-1)
            # fault = np.where(y_true == 1)[0][0]
            # all_loc = range(all_x.shape[0])
            y_score = self.get_score(all_p)
        network.train()
        return self.fpr_and_fdr_at_recall(y_pred, y_true, y_score, self.recall_level)


class SSD_score():
    def __init__(self):
        pass

    def get_roc_sklearn(self, xin, xood):
        labels = [0] * len(xin) + [1] * len(xood)
        data = np.concatenate((xin, xood))
        auroc = skm.roc_auc_score(labels, data)
        return auroc

    def get_pr_sklearn(self, xin, xood):
        labels = [0] * len(xin) + [1] * len(xood)
        data = np.concatenate((xin, xood))
        aupr = skm.average_precision_score(labels, data)
        return aupr

    def get_fpr(self, xin, xood):
        return np.sum(xood < np.percentile(xin, 95)) / len(xood)  # percentile：0-100的几分位，50 == 中位数

    def get_scores_one_cluster(self, ftrain, ftest, food, shrunkcov=False):
        if shrunkcov:
            print("Using ledoit-wolf covariance estimator.")
            cov = lambda x: ledoit_wolf(x)[0]
        else:
            cov = lambda x: np.cov(x.T, bias=True)  # 协方差计算，转置：每一行表示一个特征，每列表示不同特征的取值

        # ToDO: Simplify these equations
        dtest = np.sum(
            (ftest - np.mean(ftrain, axis=0, keepdims=True))
            * (
                np.linalg.pinv(cov(ftrain)).dot(
                    (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )

        dood = np.sum(
            (food - np.mean(ftrain, axis=0, keepdims=True))
            * (
                np.linalg.pinv(cov(ftrain)).dot(
                    (food - np.mean(ftrain, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )

        return dtest, dood

    def get_scores(self, ftrain, ftest, food, labelstrain, args):
        if args:  #.clusters == 1:
            return self.get_scores_one_cluster(ftrain, ftest, food)
        else:
            if args.training_mode == "SupCE":
                print("Using data labels as cluster since model is cross-entropy")
                ypred = labelstrain
            else:
                ypred = self.get_clusters(ftrain, args.clusters)
            return self.get_scores_multi_cluster(ftrain, food, ypred)

    def get_clusters(self, ftrain, nclusters):
        kmeans = faiss.Kmeans(
            ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
        )
        kmeans.train(np.random.permutation(ftrain))
        _, ypred = kmeans.assign(ftrain)
        return ypred

    def get_scores_multi_cluster(self, ftrain, food, ypred):
        xc = [ftrain[ypred == i] for i in np.unique(ypred)]

        # din = [
        #     np.sum(
        #         (ftest - np.mean(x, axis=0, keepdims=True))
        #         * (
        #             np.linalg.pinv(np.cov(x.T, bias=True)).dot(
        #                 (ftest - np.mean(x, axis=0, keepdims=True)).T
        #             )  # linalg.pinv 求伪逆
        #         ).T,
        #         axis=-1,
        #     )
        #     for x in xc
        # ]
        dood = [
            np.sum(
                (food - np.mean(x, axis=0, keepdims=True))
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (food - np.mean(x, axis=0, keepdims=True)).T
                    )
                ).T,
                axis=-1,
            )
            for x in xc
        ]

        # din = np.min(din, axis=0)
        dood = np.min(dood, axis=0)

        return dood  # din, dood

    def get_eval_results(self, ftrain, food, labelstrain, args):
        """
        None.
        """
        # standardize data
        # ftrain = [i / np.linalg.norm(i, axis=-1, keepdims=True) + 1e-10 for i in ftrain]
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10  # 求多个行向量的范数，然后归一化
        # ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

        ftrain = (ftrain - m) / (s + 1e-10)  # M-距离
        # ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)

        dtest, dood = self.get_scores(ftrain, ftrain, food, labelstrain, args)

        fpr95 = self.get_fpr(dtest, dood)  # TPR＝95%时，对应的FPR，以in-test为标的
        auroc, aupr = self.get_roc_sklearn(dtest, dood), self.get_pr_sklearn(dtest, dood)  # AUROC面积，平均precision
        return fpr95, auroc, aupr

    def get_loc(self, ftrain, food, labelstrain, args):
        """
        None.
        """
        # standardize data
        # ftrain = [i / np.linalg.norm(i, axis=-1, keepdims=True) + 1e-10 for i in ftrain]
        score = 0
        location = 0
        for loc in range(1, len(ftrain), 1):
            ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10  # 求多个行向量的范数，然后归一化
            food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

            m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

            ftrain = (ftrain - m) / (s + 1e-10)  # M-距离
            food = (food - m) / (s + 1e-10)

            dtest, dood = self.get_scores(ftrain, ftrain, food, labelstrain, args)
            if score < np.mean(dood) - np.mean(dtest):
                location = loc
        # fpr95 = self.get_fpr(dtest, dood)  # TPR＝95%时，对应的FPR，以in-test为标的
        # auroc, aupr = self.get_roc_sklearn(dtest, dood), self.get_pr_sklearn(dtest, dood)  # AUROC面积，平均precision
        return loc  # fpr95, auroc, aupr


def get_features(network, loaders, device):
    network.eval()
    labels = []
    features = []
    for loader in loaders:
        all_x = torch.cat([x for x, y, ind in loader]).to(device)
        labels.append(torch.cat([y for x, y, ind in loader]).numpy())
        features.append(network.featurizer(all_x).detach().cpu().numpy())

    return features, labels


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def weight_accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y, ind in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)   # ARM在predict处不同
            pl = F.softmax(p, dim=1)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def accuracy(network, loader, weights, device):
    network.eval()
    with torch.no_grad():
        all_x = torch.cat([x for x, y, ind in loader]).to(device)
        all_y = torch.cat([y for x, y, ind in loader]).to(device)
        all_p = network.predict(all_x)
        y_true = all_y.cpu().numpy()
        y_pred = np.argmax(all_p.cpu().numpy(), axis=-1)
        accuracy = accuracy_score(y_true, y_pred)
        ploc = all_x.shape[0] if  np.where(y_pred == 1)[0].shape[0] == 0 else np.where(y_pred == 1)[0][0]
        location = np.where(y_true == 1)[0][0] - ploc + 0.

    network.train()
    return accuracy, location


def cm2(loader, network, device):
    network.eval()
    with torch.no_grad():
        all_x = torch.cat([x for x, y in loader]).to(device)
        all_y = torch.cat([y for x, y in loader]).to(device)
        all_p = network.predict(all_x)
        y_true = all_y.cpu().numpy()
        y_pred = np.argmax(all_p.cpu().numpy(), axis=-1)
        C = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
        # plt.colorbar()
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def fix_nn(model, theta, phi=False):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))
        else:
            for (k, v) in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if phi:
                    name = '1'
                    tmp_model._parameters[k] = theta[str(name + '.' + k)]
                else:
                    tmp_model._parameters[k] = theta[str(name + '.' + k)]


    k_param_fn(model)
    return model