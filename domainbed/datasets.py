# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from domainbed.lib.sequence_aug import *
from domainbed.bearings_datasets import *

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # IFD
    "Bearing",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
                                                         1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)   # 返回dataset

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class DataGenerate(object):
    num_classes = 3
    inputchannel = 1

    def __init__(self, args, domain_data, labels=None,
                 transform=True, target_transform=None,
                 indices=None):

        self.domain_num = 0
        self.labels = np.array(labels)
        self.x = domain_data
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(labels))
        else:
            self.indices = indices
        self.normlizetype = args.normlizetype
        self.transforms = Compose([
                # Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
            ])
        self.target_transforms = Compose([
                # Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transforms(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transforms(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img = self.input_trans(self.x[index])
        ctarget = self.target_trans(self.labels[index])
        return img, ctarget

    def __len__(self):
        return len(self.indices)


class Bearing(object):
    N_STEPS = 2701  #851 Default, subclasses may override
    CHECKPOINT_FREQ = 50  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = ['CWRU-DE', 'CWRU-FE', 'PU-A', 'PU-R', 'MFPT']  # Subclasses should override
    INPUT_SHAPE = (1, 512)  # Subclasses should override

    def __init__(self, data_dir):
        self.input_shape = (1, 512)
        self.num_classes = None
        self.dir = data_dir
        self.domain_num = None

    def construct_domain(self, all_data='4'):
        self.num_classes = 3
        print("Load Data.")
        if all_data == '1':  # easiest data
            print('data 1')
            Ax, Ay, numA = CWRU(self.dir, task='[cwru2, cwru3]', sub_root=0).get_files_agg()   # 2356, 3456
            Bx, By, numB = CWRU(self.dir, task='[cwru6, cwru7]', sub_root=1).get_files_agg()
            Cx, Cy, numC = PU(self.dir, task='[pu1]', data_num=2, test=True).get_files()
            Dx, Dy, numD = PU(self.dir, task='[pu2]', data_num=2, test=True).get_files()
            Ex, Ey, numE = MFPT(self.dir).get_files()
            # Hx, Hy, numH = CPUMP(self.dir).get_files()

            all_x = [Ax, Bx, Cx, Dx, Ex]
            all_y = [Ay, By, Cy, Dy, Ey]

            # all_x = [Ax, Bx, Cx, Dx, Hx]
            # all_y = [Ay, By, Cy, Dy, Hy]

        elif all_data == '2':  # same datasets, but more difficult than '1'
            print('data 2')
            Ax, Ay, numA = CWRU(self.dir, task='[cwru1, cwru2]', sub_root=0).get_files_agg()  # 2356, 3456
            Bx, By, numB = CWRU(self.dir, task='[cwru6, cwru5]', sub_root=1).get_files_agg()
            Cx, Cy, numC = PU(self.dir, task='[pu1]', data_num=2, test=True).get_files()
            Dx, Dy, numD = PU(self.dir, task='[pu2]', data_num=2, test=True).get_files()
            Ex, Ey, numE = MFPT(self.dir).get_files()
            # Hx, Hy, numH = CPUMP(self.dir).get_files()

            all_x = [Ax, Bx, Cx, Dx, Ex]
            all_y = [Ay, By, Cy, Dy, Ey]
            #
            # all_x = [Ax, Bx, Cx, Dx, Hx]
            # all_y = [Ay, By, Cy, Dy, Hy]

        elif all_data == '3':
            print('data 3')
            Ax, Ay, numA = CWRU(self.dir, task='[cwru1, cwru4]', sub_root=0).get_files_agg()  # 2356, 3456
            Bx, By, numB = CWRU(self.dir, task='[cwru5, cwru8]', sub_root=1).get_files_agg()
            Cx, Cy, numC = PU(self.dir, task='[pu3]', data_num=2, test=True).get_files()
            # Dx, Dy, numD = PU(self.dir, task='[pu2]', data_num=2, test=True).get_files()
            # Ex, Ey, numE = MFPT(self.dir).get_files()
            # Fx, Fy, numF = JNU(dir=self.dir, condition=None, num_class=3).get_files_agg()
            # Gx, Gy, numG = SEU(dir=self.dir, condition=None, num_class=3).get_files_agg()
            Hx, Hy, numH = CPUMP(self.dir).get_files()


            # all_x = [Ax, Bx, Cx, Dx, Ex, Fx, Gx, Hx]
            # all_y = [Ay, By, Cy, Dy, Ey, Fy, Gy, Hy]

            all_x = [Ax, Bx, Cx, Hx]
            all_y = [Ay, By, Cy, Hy]

        elif all_data == '4':
            print('data 4')
            Ax, Ay, numA = CWRU(self.dir, task='[cwru1, cwru4]', sub_root=0).get_files_agg()  # 2356, 3456
            Bx, By, numB = CWRU(self.dir, task='[cwru8, cwru5]', sub_root=1).get_files_agg()
            Cx, Cy, numC = PU(self.dir, task='[pu1]', data_num=2, test=False).get_files()
            Dx, Dy, numD = PU(self.dir, task='[pu3]', data_num=2, test=False).get_files()

            A1x, A1y, numA1 = CWRU(self.dir, task='[cwru2, cwru3]', sub_root=0).get_files_agg()  # 2356, 3456
            B1x, B1y, numB1 = CWRU(self.dir, task='[cwru6, cwru7]', sub_root=1).get_files_agg()
            C1x, C1y, numC1 = PU(self.dir, task='[pu2]', data_num=2, test=False).get_files()

            Ex, Ey, numE = MFPT(self.dir).get_files()
            # Fx, Fy, numF = JNU(dir=self.dir, condition=None, num_class=3).get_files_agg()
            # Gx, Gy, numG = SEU(dir=self.dir, condition=None, num_class=3).get_files_agg()
            # Hx, Hy, numH = CPUMP(self.dir).get_files()


            # all_x = [Ax, Bx, Cx, Dx, Ex, Fx, Gx, Hx]
            # all_y = [Ay, By, Cy, Dy, Ey, Fy, Gy, Hy]

            all_x = [Ax, Bx, Cx, Dx, Ex]
            all_y = [Ay, By, Cy, Dy, Ey]

            # all_x = [Ax, Bx, A1x, B1x, Cx, Dx, Ex]
            # all_y = [Ay, By, A1y, B1y, Cy, Dy, Ey]

        tr_x = {}
        tr_y = {}
        num = 0

        for x, y in zip(all_x, all_y):
            tr_x[str(num)] = x['0']
            tr_y[str(num)] = y['0']
            num += 1

        self.domain_num = tr_n = len(tr_x)

        print("Start Training.")

        return tr_x, tr_y, tr_n

    def cwru_domain(self, all_data='1'):
        self.num_classes = 3
        print("Load Data.")
        if all_data == '1':
            print('B->A')   # TODO: A->B lr=1e-4 lr-omg=5e-3
            Ax, Ay, numA = CWRU(self.dir, domain='B', balance=2).get_files()
            Bx, By, numB = CWRU(self.dir, domain='A', balance=2).get_files()

            # Ax, Ay, numA = CWRU(self.dir, domain='B', balance=2).get_files_agg()
            # Bx, By, numB = CWRU(self.dir, domain='A', balance=2).get_files()

            tr_x = Ax
            tr_y = Ay
            self.domain_num = tr_n = numA + 4  # numB
            for i in range(4):  # numB
                tr_x[str(numA + i)] = Bx[str(i)]
                tr_y[str(numA + i)] = By[str(i)]

        elif all_data == '2':
            Ax, Ay, envA, numA = CWRUEnv(self.dir, domain='A', balance=3).envelope_signal()
            Bx, By, envB, numB = CWRUEnv(self.dir, domain='B', balance=3).envelope_signal()

            # td = 1
            tr_x = envA
            tr_y = Ay
            self.domain_num = tr_n = numA + 1  # numB

            for i in range(1):  # numB
                tr_x[str(numA + i)] = envB[str(i)]
                tr_y[str(numA + i)] = By[str(i)]

        elif all_data == '3':
            print('A->B')
            Ax, Ay, numA = AugCWRU(self.dir, domain='A', balance=2, alpha=1.).get_files(5)
            Bx, By, numB = AugCWRU(self.dir, domain='B', balance=2, alpha=1.).get_files(5)

            tr_x = Ax
            tr_y = Ay
            self.domain_num = tr_n = numA + 1  # numB

            td = 0
            tr_x[str(numA)] = Bx[str(td)]
            tr_y[str(numA)] = By[str(td)]

        elif all_data == '4':
            print('B->A')
            Ax, Ay, numA = CWRU(self.dir, domain='A', balance=5).get_files()
            # Ax, Ay, numA = AugCWRU(self.dir, domain='A', balance=2, alpha=1.).get_files(5)
            # Bx, By, numB = AugCWRU(self.dir, domain='B', balance=2, alpha=1.).get_files(5)
            Bx, By, numB = CWRU(self.dir, domain='B', balance=2).get_files()

            tr_x = Bx
            tr_y = By
            self.domain_num = tr_n = numB + 1  # numB

            td = 2
            tr_x[str(numB)] = Ax[str(td)]
            tr_y[str(numB)] = Ay[str(td)]

        elif all_data == '5':
            print('A,C->B')   # TODO: A->B lr=1e-4 lr-omg=5e-3
            Ax, Ay, numA = CWRU(self.dir, domain='A', balance=2).get_files()
            Bx, By, numB = CWRU(self.dir, domain='B', balance=2).get_files()
            Cx, Cy, numC = CWRU(self.dir, domain='C', balance=2).get_files()

            tr_x = Ax
            tr_y = Ay
            for i in range(4):  # numB
                tr_x[str(numA + i)] = Cx[str(i)]
                tr_y[str(numA + i)] = Cy[str(i)]
            for i in range(4):  # numB
                tr_x[str(numA + numC + i)] = Bx[str(i)]
                tr_y[str(numA + numC + i)] = By[str(i)]

            self.domain_num = tr_n = len(tr_x)

        print('Start training')
        return tr_x, tr_y, tr_n

    def pu_domain(self, all_data='1'):
        self.num_classes = 3
        print("Load Data.")
        if all_data == '1':
            print('pu1,2->pu3')   # TODO 123456->456  lr:5e-4, lr-omg:7e-3  124->356  lr:1e-4, lr-omg:5e-3(600 epochs) 101步更新学习率
            Ax, Ay, numA = PU(self.dir, task='[pu1, pu2]').get_files()
            Bx, By, numB = PU(self.dir, task='[pu3]').get_files()

            # Ax, Ay, numA = PU(self.dir, task='[pu1, pu2]').get_files_agg()
            # Bx, By, numB = PU(self.dir, task='[pu3]').get_files()

            tr_x = Ax
            tr_y = Ay
            self.domain_num = tr_n = numA + 4  # numB
            for i in range(4):  # numB
                tr_x[str(numA + i)] = Bx[str(i)]
                tr_y[str(numA + i)] = By[str(i)]

        print('Start training')
        return tr_x, tr_y, tr_n
