import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import torch
import random
import codecs
from itertools import islice
from scipy.signal import hilbert

signal_size = 1024
axis = ["_DE_time", "_FE_time", "_BA_time"]
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data"]
# For 12k Drive End Bearing Fault Data

c1 = {'0': ["97.mat"],
      '1': [
          # "130.mat",
          "197.mat",
          # "234.mat"
      ],
      '2': [
          # "105.mat",
          "169.mat",
          # "209.mat"
      ],
      '3': [
          # "118.mat",
          "185.mat",
          # "222.mat"
      ]
      }  # 1797rpm
c2 = {'0': ["98.mat"],
      '1': [
          # "131.mat",
          "198.mat",
          # "235.mat"
      ],
      '2': [
          # "106.mat",
          "170.mat",
          # "210.mat"
      ],
      '3': [
          # "119.mat",
          "186.mat",
          # "223.mat"
      ]
      }  # 1772rpm
c3 = {'0': ["99.mat"],
      '1': [
          # "132.mat",
          "199.mat",
          # "236.mat"
      ],
      '2': [
          # "107.mat",
          "171.mat",
          # "211.mat"
      ],
      '3': [
          # "120.mat",
          "187.mat",
          # "224.mat"
      ]
      }  # 1750rpm
c4 = {'0': ["100.mat"],
      '1': [
          # "133.mat",
          "200.mat",
          # "237.mat"
      ],
      '2': [
          # "108.mat",
          "172.mat",
          # "212.mat"
      ],
      '3': [
          # "121.mat",
          "188.mat",
          # "225.mat"
      ]
      }  # 1730rpm
# For 12k Fan End Bearing Fault Data      1的后两列数据不够统一，有些事@3的OR
c5 = {'0': ["97.mat"],
      '1': [
          # "294.mat",
          "310.mat",
          # "315.mat"
      ],
      '2': [
          # "278.mat",
          "274.mat",
          # "270.mat"
      ],
      '3': [
          # "282.mat",
          "286.mat",
          # "290.mat"
      ]
      }  # 1797rpm
c6 = {'0': ["98.mat"],
      '1': [
          # "295.mat",
          "309.mat",
          # "316.mat"
      ],
      '2': [
          # "279.mat",
          "275.mat",
          # "271.mat"
      ],
      '3': [
          # "283.mat",
          "287.mat",
          # "291.mat"
      ]
      }  # 1772rpm
c7 = {'0': ["99.mat"],
      '1': [
          # "296.mat",
          "311.mat",
          # "317.mat"
      ],
      '2': [
          # "280.mat",
          "276.mat",
          # "272.mat"
      ],
      '3': [
          # "284.mat",
          "288.mat",
          # "292.mat"
      ]
      }  # 1750rpm
c8 = {'0': ["100.mat"],
      '1': [
          # "297.mat",
          "312.mat",
          # "318.mat"
      ],
      '2': [
          # "281.mat",
          "277.mat",
          # "273.mat"
      ],
      '3': [
          # "285.mat",
          "289.mat",
          # "293.mat"
      ]
      }  # 1730rpm

cwru1 = {'0': ["97.mat"],
         '1': [
             "130.mat",
             "197.mat",
             "234.mat"
         ],
         '2': [
             "105.mat",
             "169.mat",
             "209.mat"
         ],
         # '3': [
         #     "118.mat",
         #     "185.mat",
         #     "222.mat"
         # ]
         }  # 1797rpm
cwru2 = {'0': ["98.mat"],
         '1': [
             "131.mat",
             "198.mat",
             "235.mat"
         ],
         '2': [
             "106.mat",
             "170.mat",
             "210.mat"
         ],
         # '3': [
         #     "119.mat",
         #     "186.mat",
         #     "223.mat"
         # ]
         }  # 1772rpm
cwru3 = {'0': ["99.mat"],
         '1': [
             "132.mat",
             "199.mat",
             "236.mat"
         ],
         '2': [
             "107.mat",
             "171.mat",
             "211.mat"
         ],
         # '3': [
         #     "120.mat",
         #     "187.mat",
         #     "224.mat"
         # ]
         }  # 1750rpm
cwru4 = {'0': ["100.mat"],
         '1': [
             "133.mat",
             "200.mat",
             "237.mat"
         ],
         '2': [
             "108.mat",
             "172.mat",
             "212.mat"
         ],
         # '3': [
         #     "121.mat",
         #     "188.mat",
         #     "225.mat"
         # ]
         }  # 1730rpm
# For 12k Fan End Bearing Fault Data      1的后两列数据不够统一，有些事@3的OR
cwru5 = {'0': ["97.mat"],
         '1': [
             "294.mat",
             "310.mat",
             "315.mat"
         ],
         '2': [
             "278.mat",
             "274.mat",
             "270.mat"
         ],
         # '3': [
         #     "282.mat",
         #     "286.mat",
         #     "290.mat"
         # ]
         }  # 1797rpm
cwru6 = {'0': ["98.mat"],
         '1': [
             "295.mat",
             "309.mat",
             "316.mat"
         ],
         '2': [
             "279.mat",
             "275.mat",
             "271.mat"
         ],
         # '3': [
         #     "283.mat",
         #     "287.mat",
         #     "291.mat"
         # ]
         }  # 1772rpm
cwru7 = {'0': ["99.mat"],
         '1': [
             "296.mat",
             "311.mat",
             "317.mat"
         ],
         '2': [
             "280.mat",
             "276.mat",
             "272.mat"
         ],
         # '3': [
         #     "284.mat",
         #     "288.mat",
         #     "292.mat"
         # ]
         }  # 1750rpm
cwru8 = {'0': ["100.mat"],
         '1': [
             "297.mat",
             "312.mat",
             "318.mat"
         ],
         '2': [
             "281.mat",
             "277.mat",
             "273.mat"
         ],
         # '3': [
         #     "285.mat",
         #     "289.mat",
         #     "293.mat"
         # ]
         }  # 1730rpm
# For 48k Drive End Bearing Fault Data                                               wrong?    217wrong?
cwru9 = {'0': ["97.mat"], '1': ["135.mat", "201.mat", "238.mat"], '2': ["109.mat", "173.mat", "213.mat"],
         # '3': ["122.mat", "189.mat", "226.mat"]
         }  # 1797rpm
cwru10 = {'0': ["98.mat"], '1': ["136.mat", "202.mat", "239.mat"], '2': ["110.mat", "175.mat", "214.mat"],
          # '3': ["123.mat", "190.mat", "227.mat"]
          }  # 1772rpm
cwru11 = {'0': ["99.mat"], '1': ["137.mat", "203.mat", "240.mat"], '2': ["111.mat", "176.mat", "215.mat"],
          # '3': ["124.mat", "191.mat", "228.mat"]
          }  # 1750rpm
cwru12 = {'0': ["100.mat"], '1': ["138.mat", "204.mat", "241.mat"], '2': ["112.mat", "177.mat", "217.mat"],
          # '3': ["125.mat", "192.mat", "229.mat"]
          }  # 1730rpm
domain_cwru = {
    'A': {'data': [cwru1, cwru2, cwru3, cwru4], 'subroot': 0, 'axis': 0, 'frequency': 12000, 'down_f': 12000},
    'B': {'data': [cwru5, cwru6, cwru7, cwru8], 'subroot': 1, 'axis': 1, 'frequency': 12000, 'down_f': 12000},
    'C': {'data': [cwru9, cwru10, cwru11, cwru12], 'subroot': 2, 'axis': 0, 'frequency': 48000, 'down_f': 16000},
    'D': {'data': [c1, c2, c3, c4], 'subroot': 0, 'axis': 0, 'frequency': 12000, 'down_f': 12000},
    'E': {'data': [c5, c6, c7, c8], 'subroot': 1, 'axis': 1, 'frequency': 12000, 'down_f': 12000},
    # 'F': {'data': [cwru9, cwru10, cwru11, cwru12], 'subroot': 2, 'axis': 0, 'frequency': 48000, 'down_f': 16000}
}


# TODO: cwru data for construct_domain
# cwru1 = {'0': ["97.mat"],
#          '1': [
#                # "130.mat",
#                "197.mat",
#                "234.mat"
#                ],
#          '2': [
#                # "105.mat",
#                "169.mat",
#                "209.mat"
#                ],
#          }  # 1797rpm
# cwru2 = {'0': ["98.mat"],
#          '1': [
#                # "131.mat",
#                "198.mat",
#                "235.mat"],
#          '2': [
#                # "106.mat",
#                "170.mat",
#                "210.mat"
#                ],
#          }  # 1772rpm
# cwru3 = {'0': ["99.mat"],
#          '1': [
#                # "132.mat",
#                "199.mat",
#                "236.mat"
#                ],
#          '2': [
#                # "107.mat",
#                "171.mat",
#                "211.mat"],
#          }  # 1750rpm
# cwru4 = {'0': ["100.mat"],
#          '1': [
#                # "133.mat",
#                "200.mat",
#                "237.mat"
#                ],
#          '2': [
#                # "108.mat",
#                "172.mat",
#                "212.mat"
#                ],
#          }  # 1730rpm
# # For 12k Fan End Bearing Fault Data      1的后两列数据不够统一，有些事@3的OR
# cwru5 = {'0': ["97.mat"],
#          '1': [
#                # "294.mat",
#                "310.mat",
#                "315.mat"
#                ],
#          '2': [
#                # "278.mat",
#                "274.mat",
#                "270.mat"
#                ],
#          }  # 1797rpm
# cwru6 = {'0': ["98.mat"],
#          '1': [
#                # "295.mat",
#                "309.mat",
#                "316.mat"
#                ],
#          '2': [
#                # "279.mat",
#                "275.mat",
#                "271.mat"
#                ],
#          }  # 1772rpm
# cwru7 = {'0': ["99.mat"],
#          '1': [
#                # "296.mat",
#                "311.mat",
#                "317.mat"
#                ],
#          '2': [
#                # "280.mat",
#                "276.mat",
#                "272.mat"
#                ],
#          }  # 1750rpm
# cwru8 = {'0': ["100.mat"],
#          '1': [
#                # "297.mat",
#                "312.mat",
#                "318.mat"
#                ],
#          '2': [
#                # "281.mat",
#                "277.mat",
#                "273.mat"
#                ],
#          }  # 1730rpm
# # # For 48k Drive End Bearing Fault Data                                               wrong?    217wrong?
# cwru9 = {'0': ["97.mat"],
#          '1': ["201.mat",
#                "238.mat"],
#          '2': ["173.mat",
#                "213.mat"]}  # 1797rpm
# cwru10 = {'0': ["98.mat"],
#           '1': ["202.mat",
#                 "239.mat"],
#           '2': ["175.mat",
#                 "214.mat"]}  # 1772rpm
# cwru11 = {'0': ["99.mat"],
#           '1': ["203.mat",
#                 "240.mat"],
#           '2': ["176.mat",
#                 "215.mat"]}  # 1750rpm
# cwru12 = {'0': ["100.mat"],
#           '1': ["204.mat",
#                 "241.mat"],
#           '2': ["177.mat",
#                 "217.mat"]}  # 1730rpm
# domain_cwru = {
#     'A': {'data': [cwru1, cwru2, cwru3, cwru4], 'subroot': 0, 'axis': 0, 'frequency': 12000, 'down_f': 12000},
#     'B': {'data': [cwru5, cwru6, cwru7, cwru8], 'subroot': 1, 'axis': 1, 'frequency': 12000, 'down_f': 12000},
#     'C': {'data': [cwru9, cwru10, cwru11, cwru12], 'subroot': 2, 'axis': 0, 'frequency': 48000, 'down_f': 16000}
# }


class CWRUEnv(object):
    def __init__(self, dir, domain='A', task=None, sub_root=None, test=False, num_class=3, balance=1):
        ba_num = {1: [105, 52],  # 三分类。只用到了0.014和0.021
                  2: [210, 70],
                  3: [110, 110]
                  }
        self.test = test
        self.num_class = num_class
        self.balance = ba_num[balance]
        if task is None:
            self.domain = domain_cwru[domain]
            self.data = self.domain['data']
            self.root = os.path.join(dir, 'CWRU', datasetname[self.domain['subroot']])
            self.frequency = self.domain['frequency']
            self.down_f = self.domain['down_f']
            self.axis = axis[self.domain['axis']]
        else:
            self.data = eval(task)
            self.sub_root = datasetname[sub_root]
            self.root = os.path.join(dir, 'CWRU', self.sub_root)
            if sub_root == 1:  # 12k FE
                self.axis = axis[1]
            else:
                self.axis = axis[0]
            if sub_root == 2:  # 48k DE
                self.frequency = 48000
                self.down_f = 12000
            else:
                self.frequency = 12000
                self.down_f = 12000

    def get_files(self):
        data = {}
        lab = {}
        raw = {}
        domain_num = 0
        data[str(domain_num)] = []
        lab[str(domain_num)] = []

        for i in self.data:
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            raw[str(domain_num)] = []
            for (k, v) in i.items():
                # print(k)
                for j in v:
                    path1 = os.path.join(self.root, j)
                    data1, lab1, raw1 = self.data_load(path1, j, label=int(k))
                    # print(j, len(lab1))
                    data[str(domain_num)] += data1
                    lab[str(domain_num)] += lab1
                    raw[str(domain_num)] += raw1

            domain_num += 1
            # print('\n')
        return data, lab, raw, len(lab)

    def envelope_signal(self):
        data, lab, rawdata, num = self.get_files()
        amplitude_envelope = {}
        for k, v in data.items():
            amplitude_envelope[str(k)] = []
            for signals in v:
                signals = signals.reshape(-1)

                analytic_signal = hilbert(signals)
                envelope = np.abs(analytic_signal).reshape(1, -1)
                amplitude_envelope[str(k)].append(envelope)

        return data, lab, amplitude_envelope, num

    def get_files_agg(self):
        data = []
        lab = []
        raw = []

        for c in range(self.num_class):
            for g in self.data:
                for j in range(len(g[str(c)])):
                    path1 = os.path.join(self.root, g[str(c)][j])
                    data1, lab1, raw1 = self.data_load(path1, g[str(c)][j], label=int(c))
                    data += data1
                    lab += lab1
                    raw += raw1

        data0 = {}
        lab0 = {}
        raw0 = {}
        data0['0'] = data
        lab0['0'] = lab
        raw0['0'] = raw
        return data0, lab0, len(lab0)

    def data_load(self, filename, axisname, label):
        datanumber = axisname.split(".")
        if eval(datanumber[0]) < 100:
            realaxis = "X0" + datanumber[0] + self.axis
        else:
            realaxis = "X" + datanumber[0] + self.axis
        fl = loadmat(filename)[realaxis]
        # fl = fl.reshape(1, -1)
        if eval(datanumber[0]) <= 100:
            data = []
            lab = []
            raw = []
            nn = len(fl) - signal_size
            for i in range(self.balance[0]):  # 划分样本： 随机选择一个起点，然后截取规定长度的样本
                # random.seed(0)
                n = random.randint(0, nn)
                x = fl[n:n + signal_size]
                raw.append(x)
                x = np.fft.fft(x)
                x = np.abs(x) / len(x)
                x = x[range(int(x.shape[0] / 2))]
                x = x.reshape(1, -1)
                data.append(x)
                lab.append(label)
        else:
            # fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0) if self.frequency == 48000 else fl
            fl = np.vstack((fl, fl, fl))
            data = []
            lab = []
            raw = []
            start, end = 0, signal_size
            while (end <= fl.shape[0] and len(lab) < self.balance[1]):
                x = fl[start:end]
                raw.append(x)
                x = np.fft.fft(x)
                x = np.abs(x) / len(x)
                x = x[range(int(x.shape[0] / 2))]
                x = x.reshape(1, -1)
                data.append(x)
                lab.append(label)
                start += signal_size
                end += signal_size

        return data, lab, raw


class CWRU(object):
    def __init__(self, dir, domain='A', task=None, sub_root=None, test=False, num_class=3, balance=1):
        ba_num = {1: [105, 52],  # 三分类。只用到了0.014和0.021
                  2: [270, 90],
                  3: [110, 110],
                  4: [210, 70],
                  5: [70, 70]
                  }
        self.test = test
        self.num_class = num_class
        self.balance = ba_num[balance]
        if task is None:
            self.domain = domain_cwru[domain]
            self.data = self.domain['data']
            # TODO: fix the 'flags' to 'flags.data_dir'
            self.root = os.path.join(dir, 'CWRU', datasetname[self.domain['subroot']])
            self.frequency = self.domain['frequency']
            self.down_f = self.domain['down_f']
            self.axis = axis[self.domain['axis']]
        else:
            self.data = eval(task)
            self.sub_root = datasetname[sub_root]
            self.root = os.path.join(dir, 'CWRU', self.sub_root)
            if sub_root == 1:  # 12k FE
                self.axis = axis[1]
            else:
                self.axis = axis[0]
            if sub_root == 2:  # 48k DE
                self.frequency = 48000
                self.down_f = 12000
            else:
                self.frequency = 12000
                self.down_f = 12000

    def get_files(self):
        data = {}
        lab = {}
        # data = []
        # lab = []
        domain_num = 0
        data[str(domain_num)] = []
        lab[str(domain_num)] = []

        for i in self.data:
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            for (k, v) in i.items():
                # print(k)
                for j in v:
                    path1 = os.path.join(self.root, j)
                    data1, lab1 = self.data_load(path1, j, label=int(k))
                    # print(j, len(lab1))
                    data[str(domain_num)] += data1
                    lab[str(domain_num)] += lab1
                    # data += data1
                    # lab += lab1

            domain_num += 1
            # print('\n')
        return data, lab, len(lab)

    def get_files_agg(self):
        data = []
        lab = []

        for c in range(self.num_class):
            for g in self.data:
                for j in range(len(g[str(c)])):
                    path1 = os.path.join(self.root, g[str(c)][j])
                    data1, lab1 = self.data_load(path1, g[str(c)][j], label=int(c))
                    data += data1
                    lab += lab1

        data0 = {}
        lab0 = {}
        data0['0'] = data
        lab0['0'] = lab
        return data0, lab0, len(lab0)

    # def data_load(self, filename, axisname, label):
    #     datanumber = axisname.split(".")
    #     if eval(datanumber[0]) < 100:
    #         realaxis = "X0" + datanumber[0] + self.axis
    #     else:
    #         realaxis = "X" + datanumber[0] + self.axis
    #     fl = loadmat(filename)[realaxis]
    #     # fl = fl.reshape(1, -1)
    #     if eval(datanumber[0]) <= 100:
    #         data = []
    #         lab = []
    #         nn = len(fl) - signal_size
    #         for i in range(self.balance[0]):  # 划分样本： 随机选择一个起点，然后截取规定长度的样本
    #             n = random.randint(0, nn)
    #             x = fl[n:n + signal_size]
    #             x = np.fft.fft(x)
    #             x = np.abs(x) / len(x)
    #             x = x[range(int(x.shape[0] / 2))]
    #             x = x.reshape(1, -1)
    #             data.append(x)
    #             lab.append(label)
    #     else:
    #         fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0) if self.frequency == 48000 else fl
    #         data = []
    #         lab = []
    #         start, end = 0, signal_size
    #         while (end <= fl.shape[0] and len(lab) < self.balance[1]):
    #             x = fl[start:end]
    #             x = np.fft.fft(x)
    #             x = np.abs(x) / len(x)
    #             x = x[range(int(x.shape[0] / 2))]
    #             x = x.reshape(1, -1)
    #             data.append(x)
    #             lab.append(label)
    #             start += signal_size
    #             end += signal_size
    #
    #     return data, lab

    def data_load(self, filename, axisname, label):
        datanumber = axisname.split(".")
        if eval(datanumber[0]) < 100:
            realaxis = "X0" + datanumber[0] + self.axis
        else:
            realaxis = "X" + datanumber[0] + self.axis
        fl = loadmat(filename)[realaxis]
        # fl = fl.reshape(1, -1)
        if eval(datanumber[0]) <= 100:
            data = []
            lab = []
            nn = len(fl) - signal_size
            for i in range(self.balance[0]):  # 划分样本： 随机选择一个起点，然后截取规定长度的样本
                n = random.randint(0, nn)
                x = fl[n:n + signal_size]
                x = np.fft.fft(x)
                x = np.abs(x) / len(x)
                x = x[range(int(x.shape[0] / 2))]
                x = x.reshape(1, -1)
                data.append(x)
                lab.append(label)
        else:
            fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0) if self.frequency == 48000 else fl
            data = []
            lab = []
            start, end, lap = 0, signal_size, 2000
            fl = np.vstack((fl, fl))
            while (end <= fl.shape[0] and len(lab) < self.balance[1]):
                x = fl[start:end]
                x = np.fft.fft(x)
                x = np.abs(x) / len(x)
                x = x[range(int(x.shape[0] / 2))]
                x = x.reshape(1, -1)
                data.append(x)
                lab.append(label)
                start += lap
                end += lap

        return data, lab


class AugCWRU(object):
    def __init__(self, dir, domain='A', task=None, sub_root=None, test=False, balance=1, alpha=1.0):
        ba_num = {1: [105, 52],
                  2: [270, 90],
                  3: [110, 110],
                  4: [210, 70]}
        self.test = test
        self.balance = ba_num[balance]
        self.alpha = alpha
        if task is None:
            self.domain = domain_cwru[domain]
            self.data = self.domain['data']
            self.root = os.path.join(dir, 'CWRU', datasetname[self.domain['subroot']])
            self.frequency = self.domain['frequency']
            self.down_f = self.domain['down_f']
            self.axis = axis[self.domain['axis']]
        else:
            self.data = eval(task)
            self.sub_root = datasetname[sub_root]
            self.root = os.path.join(dir, 'CWRU', self.sub_root)
            if sub_root == 1:  # 12k FE
                self.axis = axis[1]
            else:
                self.axis = axis[0]
            if sub_root == 2:  # 48k DE
                self.frequency = 48000
                self.down_f = 12000
            else:
                self.frequency = 12000
                self.down_f = 12000

    def get_files(self, seed):
        data = {}
        lab = {}
        raw = {}
        # data = []
        # lab = []
        domain_num = 0
        data[str(domain_num)] = []
        lab[str(domain_num)] = []

        for i in self.data:  # CWRU1,2 ...
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            raw[str(domain_num)] = []
            for (k, v) in i.items():  # .mat / labels
                if k == '0':
                    fl = self.data_load(v[0])
                    nn = len(fl) - signal_size
                    for num in range(self.balance[0]):  # 划分样本： 随机选择一个起点，然后截取规定长度的样本
                        n = random.randint(0, nn)
                        x = fl[n:n + signal_size]
                        raw[str(domain_num)].append(x)
                        x = np.fft.fft(x)
                        x = np.abs(x) / len(x)
                        x = x[range(int(x.shape[0] / 2))]
                        x = x.reshape(1, -1)
                        data[str(domain_num)].append(x)
                        lab[str(domain_num)].append(int(k))
                else:
                    fls = []
                    mixup_dir_list = []
                    for j in v:
                        fls.append(self.data_load(j).reshape(1, -1)[:, :120000])
                        mixup_dir_list.append(self.alpha)

                    RG = np.random.default_rng(seed)

                    for n in range(len(v)):
                        mixup_ratio = RG.dirichlet(mixup_dir_list, size=1).T
                        print(mixup_ratio)
                        mixup_ratio = np.expand_dims(mixup_ratio, axis=-1)
                        fl = np.stack(fls)
                        fl = np.sum(fl * mixup_ratio, axis=0).reshape(-1, 1)
                        fl = np.vstack((fl, fl, fl))
                        start, end, lap = 0, signal_size, int(signal_size)
                        count = 0
                        while (end <= fl.shape[0] and count < self.balance[1]):
                            x = fl[start:end]
                            x = np.fft.fft(x)
                            x = np.abs(x) / len(x)
                            x = x[range(int(x.shape[0] / 2))]
                            x = x.reshape(1, -1)
                            data[str(domain_num)].append(x)
                            lab[str(domain_num)].append(int(k))
                            start += lap
                            end += lap
                            count += 1
            domain_num += 1
        return data, lab, len(lab)

    def data_load(self, axisname):
        filename = os.path.join(self.root, axisname)
        datanumber = axisname.split(".")
        if eval(datanumber[0]) < 100:
            realaxis = "X0" + datanumber[0] + self.axis
        else:
            realaxis = "X" + datanumber[0] + self.axis
        file = loadmat(filename)[realaxis]
        return file


WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
# health, OR, IR
# pu1 = {'0': ['K001', "K002", 'K003'], '1': ['KA01', 'KA03', 'KA05'], '2': ['KI01', 'KI07', 'KI08']}
# pu2 = {'0': ['K004', "K005", 'K006'], '1': ['KA04', 'KA16', 'KA30'], '2': ['KI14', 'KI17', 'KI18']}

pu1 = {'0': ['K001', 'K002', 'K004'], '1': ['KA01', 'KA03', 'KA05'], '2': ['KI01', 'KI07', 'KI08']}
pu2 = {'0': ['K001', 'K002', 'K004'], '1': ['KA06', 'KA07', 'KA08', 'KA09'], '2': ['KI03', 'KI05']}
pu3 = {'0': ['K003', 'K005', 'K006'], '1': ['KA04', 'KA16', 'KA30'], '2': ['KI14', 'KI17', 'KI18']}


class PU(object):
    def __init__(self, dir, task='[pu1, pu2]', condition='[0]', data_num=2, test=False):
        # self.WC = eval(condition)
        if test:
            self.WC = [WC[0], WC[3]]  # 2  3  // 0 3 for '2'
        else:
            self.WC = WC
        self.test = test
        self.frequency = 64000
        self.down_f = 12000
        self.root = os.path.join(dir, 'PU')
        self.data = eval(task)
        self.d_num = data_num + 1
        # self.d_num = 3

    def get_files(self):
        data = {}
        lab = {}
        # data = []
        # lab = []
        domain_num = 0

        for i in self.data:
            for state in self.WC:
                data[str(domain_num)] = []
                lab[str(domain_num)] = []
                for (k, v) in i.items():
                    # print(k)
                    for j in v:
                        for num in range(1, self.d_num):
                            name1 = state + "_" + j + "_" + '1'
                            path1 = os.path.join(self.root, j, name1 + ".mat")
                            data1, lab1 = self.data_load(path1, name=name1, label=int(k))
                            # print(j, len(lab1))
                            data[str(domain_num)] += data1
                            lab[str(domain_num)] += lab1
                            # data += data1
                            # lab += lab1

                domain_num += 1
                # print('\n')
        return data, lab, len(lab)   # 数据，标签，域数量

    def get_files_agg(self):
        # data = {}
        # lab = {}
        data = []
        lab = []
        domain_num = 0

        for i in self.data:
            for state in self.WC:
                # data[str(domain_num)] = []
                # lab[str(domain_num)] = []
                for (k, v) in i.items():
                    # print(k)
                    for j in v:
                        for num in range(1, self.d_num):
                            name1 = state + "_" + j + "_" + '1'
                            path1 = os.path.join(self.root, j, name1 + ".mat")
                            data1, lab1 = self.data_load(path1, name=name1, label=int(k))
                            # print(j, len(lab1))
                            # data[str(domain_num)] += data1
                            # lab[str(domain_num)] += lab1
                            data += data1
                            lab += lab1

        data0 = {}
        lab0 = {}
        data0['0'] = data
        lab0['0'] = lab
        return data0, lab0, len(lab0)

    def data_load(self, filename, name, label):
        fl = loadmat(filename)[name]
        fl = fl[0][0][2][0][6][2]  # Take out the data
        # fl = fl.reshape(-1,)
        fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=1)
        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[1]:
            x = fl[:, start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[:, range(int(x.shape[1] / 2))]
            # x = x.reshape(-1, 1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size

        return data, lab


class MFPT(object):
    def __init__(self, dir):
        # self.WC = WC
        self.down_f = 12000
        self.root = os.path.join(dir, 'MFPT')

    def get_files(self):
        '''
        This function is used to generate the final training set and test set.
        root:The location of the data set
        '''
        m = os.listdir(self.root)
        m.sort(key=lambda x1: int(x1.split('-')[0]))
        # datasetname = os.listdir(os.path.join(root, m[0]))
        # '1 - Three Baseline Conditions'
        # '2 - Three Outer Race Fault Conditions'
        # '3 - Seven More Outer Race Fault Conditions'
        # '4 - Seven Inner Race Fault Conditions'
        # '5 - Analyses',
        # '6 - Real World Examples
        # Generate a list of data
        dataset1 = os.listdir(os.path.join(self.root, m[0]))  # 'Three Baseline Conditions'
        dataset2 = os.listdir(os.path.join(self.root, m[2]))  # 'Seven More Outer Race Fault Conditions'
        dataset2.sort(key=lambda x1: int(x1.split('.')[0].split('_')[2]))
        dataset3 = os.listdir(os.path.join(self.root, m[3]))  # 'Seven Inner Race Fault Conditions'
        dataset3.sort(key=lambda x1: int(x1.split('.')[0].split('_')[2]))
        data_root1 = os.path.join(self.root, m[0])  # Path of Three Baseline Conditions
        data_root2 = os.path.join(self.root, m[2])  # Path of Seven More Outer Race Fault Conditions
        data_root3 = os.path.join(self.root, m[3])  # Path of Seven Inner Race Fault Conditions

        data = []
        lab = []

        # print(0)
        for i in range(len(dataset1)):
            path1 = os.path.join(data_root1, dataset1[i])
            data0, lab0 = self.data_load(path1, label=0)  # The label for normal data is 0
            # print(len(lab0))
            data += data0
            lab += lab0

        # print(1)
        for i in range(6):
            path2 = os.path.join(data_root2, dataset2[i])
            data1, lab1 = self.data_load(path2, label=1)
            # print(len(lab1))
            data += data1
            lab += lab1

        # print(2)
        for j in range(6):
            path3 = os.path.join(data_root3, dataset3[j])
            data2, lab2 = self.data_load(path3, label=2)
            # print(len(lab2))
            data += data2
            lab += lab2

        all_data, all_lab = {}, {}
        all_data['0'] = data
        all_lab['0'] = lab
        return all_data, all_lab, len(all_lab)

    def data_load(self, filename, label):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        if label == 0:
            fl = (loadmat(filename)["bearing"][0][0][1])  # Take out the data
            self.frequency = 97656
        else:
            fl = (loadmat(filename)["bearing"][0][0][2])  # Take out the data
            self.frequency = 48000  # 48828

        # fl = fl.reshape(-1, )
        fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0)

        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[0]:
            x = fl[start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x = x.reshape(1, -1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size
        return data, lab


class JNU(object):
    # 351 117 117 117
    # Three working conditions, 4 health conditions
    domain_JNU = {'1': ["n600_short.txt", "ob600_2.csv", "ib600_2.csv", "tb600_2.csv"],
                  '2': ["n800_short.txt", "ob800_2.csv", "ib800_2.csv", "tb800_2.csv"],
                  '3': ["n1000_short.txt", "ob1000_2.csv", "ib1000_2.csv", "tb1000_2.csv"]
                  }

    def __init__(self, dir, condition=None, num_class=3):
        self.root = os.path.join(dir, 'JNU')
        self.down_f = 12000
        self.frequency = 50000
        if condition is not None:
            condition = eval(condition)
            self.WC = {i: self.domain_JNU[i] for i in condition}
        else:
            self.WC = self.domain_JNU
        self.num_class = num_class

    def get_files(self):
        data = {}
        lab = {}
        domain_num = 0

        for wc, file in self.WC.items():
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            # print(wc)
            for i in range(self.num_class):
                path1 = os.path.join(self.root, file[i])
                data1, lab1 = self.data_load(path1, label=i)
                data[str(domain_num)] += data1
                lab[str(domain_num)] += lab1
                # print(i, len(lab1))

            domain_num += 1
            # print('\n')
        return data, lab, len(lab)

    def get_files_agg(self):
        data = []
        lab = []

        for i in range(self.num_class):
            for wc, file in self.WC.items():
                path1 = os.path.join(self.root, file[i])
                data1, lab1 = self.data_load(path1, label=i)
                data += data1
                lab += lab1

        data0 = {}
        lab0 = {}
        data0['0'] = data
        lab0['0'] = lab
        return data0, lab0, len(lab0)

    def data_load(self, filename, label):
        fl = np.loadtxt(filename)
        fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0)
        # fl = fl.reshape(1, -1)
        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[0]:
            x = fl[start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x = x.reshape(1, -1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size

        return data, lab


class SEU(object):
    # Data names of 5 bearing fault types under two working conditions
    # without down sampling, the numbers of samples in each class are 102 102 102
    domain_SEU = {'1': ["health_20_0.csv", "outer_20_0.csv", "inner_20_0.csv", "ball_20_0.csv", "comb_20_0.csv"],
                  '2': ["health_30_2.csv", "outer_30_2.csv", "inner_30_2.csv", "ball_30_2.csv", "comb_30_2.csv"]}

    def __init__(self, dir, condition=None, num_class=3):
        self.root = os.path.join(dir, 'SEU')
        self.down_f = 12000
        self.frequency = 12000  # TODO: check out the sampling rate
        if condition is not None:
            condition = eval(condition)
            self.WC = {i: self.domain_SEU[i] for i in condition}
        else:
            self.WC = self.domain_SEU
        self.num_class = num_class

    def get_files(self):
        data = {}
        lab = {}
        domain_num = 0

        for wc, file in self.WC.items():
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            # print(wc)
            for i in range(self.num_class):
                path1 = os.path.join(self.root, file[i])
                data1, lab1 = self.data_load(path1, dataname=file[i], label=i)
                data[str(domain_num)] += data1
                lab[str(domain_num)] += lab1
                # print(i, len(lab1))

            domain_num += 1
            # print('\n')
        return data, lab, len(lab)

    def get_files_agg(self):
        data = []
        lab = []

        # for wc, file in self.WC.items():
        #     print(wc)
        #     for i in range(self.num_class):
        #         path1 = os.path.join(self.root, file[i])
        #         data1, lab1 = self.data_load(path1, dataname=file[i], label=i)
        #         data += data1
        #         lab += lab1
        #         print(i, len(lab1))
        #
        #     print('\n')

        for i in range(self.num_class):
            for wc, file in self.WC.items():
                path1 = os.path.join(self.root, file[i])
                data1, lab1 = self.data_load(path1, dataname=file[i], label=i)
                data += data1
                lab += lab1

        data0 = {}
        lab0 = {}
        data0['0'] = data
        lab0['0'] = lab
        return data0, lab0, len(lab0)

    def data_load(self, filename, dataname, label):
        f = open(filename, "r", encoding='gb18030', errors='ignore')
        fl = []
        if dataname == "ball_20_0.csv":  # TODO: 取某个坐标下的一组数据，保存为txt文件，加速
            for line in islice(f, 16, None):  # Skip the first 16 lines
                line = line.rstrip()
                word = line.split(",", 8)  # Separated by commas
                fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
        else:
            for line in islice(f, 16, None):  # Skip the first 16 lines
                line = line.rstrip()
                word = line.split("\t", 8)  # Separated by \t
                fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
        fl = np.array(fl)
        # fl = fl.reshape(-1, 1)
        fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0)
        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[0] / 10:
            x = fl[start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x = x.reshape(1, -1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size

        return data, lab


class CPUMP(object):
    # 20*5 20*5 20*5
    def __init__(self, dir):
        self.root = os.path.join(dir, 'CPUMP')

    def get_files(self):
        m = os.listdir(self.root)
        m.sort(key=lambda x1: int(x1.split('_')[0]))

        dataset1 = os.listdir(os.path.join(self.root, m[0]))  # Normal
        dataset1.sort(key=lambda x1: int(x1.split('#')[0].split('C')[1]))
        dataset2 = os.listdir(os.path.join(self.root, m[1]))  # Outer
        dataset2.sort(key=lambda x1: x1.split('#')[0])
        dataset3 = os.listdir(os.path.join(self.root, m[2]))  # Inner
        dataset3.sort(key=lambda x1: x1.split('#')[0])
        data_root1 = os.path.join(self.root, m[0])  # normal
        data_root2 = os.path.join(self.root, m[1])  # Outer
        data_root3 = os.path.join(self.root, m[2])  # Inner

        data = []
        lab = []

        # print(0)
        for i in range(len(dataset1)):
            path1 = os.path.join(data_root1, dataset1[i])
            data0, lab0 = self.data_load(path1, label=0)  # The label for normal data is 0
            # print(len(lab0))
            data += data0
            lab += lab0

        # print(1)
        for i in range(len(dataset2)):
            path2 = os.path.join(data_root2, dataset2[i])
            data1, lab1 = self.data_load(path2, label=1)
            # print(len(lab1))
            data += data1
            lab += lab1

        # print(2)
        for j in range(len(dataset3)):
            path3 = os.path.join(data_root3, dataset3[j])
            data2, lab2 = self.data_load(path3, label=2)
            # print(len(lab2))
            data += data2
            lab += lab2

        all_data, all_lab = {}, {}
        all_data['0'] = data
        all_lab['0'] = lab

        return all_data, all_lab, len(all_lab)

    def data_load(self, filename, label):
        fl = []
        f = codecs.open(filename, mode='r', encoding='utf-8')
        for i in range(3):
            next(f)
        line = f.readline()
        while line:
            line = line.split()
            fl.append(eval(line[0]))
            line = f.readline()

        fl = np.array(fl)

        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[0]:
            x = fl[start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x = x.reshape(1, -1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size
        return data, lab


class BearingRUL(object):

    def __init__(self, dir, dataname):
        self.name = dataname
        self.dir = dir

    def get_data(self):
        bearing_names = os.listdir(self.dir)
        bearing_names.sort()
        bearing_data = {}
        bearing_lab = {}
        # count = 0
        for i in self.name:
            bearing_data[i] = []
            bearing_lab[i] = []
            for j in bearing_names:
                if i in j:
                    data = pd.read_csv(os.path.join(self.dir, j), header=None)
                    data = np.array(data).reshape(data.shape[0], 1, -1)
                    # np.expand_dims(data, axis=1)
                    bearing_data[i].append(data[:, :, :-1])
                    bearing_lab[i].append(data[:, :, -1].reshape(-1))

            # count += 1

        return bearing_data, bearing_lab, len(self.name)

    def load_data(self):
        # dir = r'/data/yfy/FD-data/RUL/phm_dict.npy'
        # data = np.load(dir, allow_pickle=True).item() # (time-steps, 2560, 2)
        bearing_data = {}
        bearing_lab = {}
        # count = 0
        y = np.load(os.path.join(self.dir, 'labels.npy'), allow_pickle=True).item()
        for i in self.name:
            bearing_data[i] = []
            bearing_lab[i] = []
            x = np.load(os.path.join(self.dir, i+'.npy'), allow_pickle=True).item()
            for k, v in x.items():
                bearing_data[i].append(v.transpose(0, 2, 1))

            for j, v1 in y.items():
                if i in j:
                    bearing_lab[i].append(v1)

                # count += 1

        return bearing_data, bearing_lab, len(self.name)


