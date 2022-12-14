import copy
import os

import torch.nn
from torch.autograd import Variable

import mymodels
from datasets.data_gen_PACS import *
from utils.FCutils import *
import datetime
from datautil.getdataloader import get_data_loader


class ModelBaseline_PACS:
    def __init__(self, flags):
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.configure(flags)   # 日志路径
        self.setup_path(flags)  # 数据，数据生成器
        self.init_network_parameter(flags)  # 网络初始化参数，预训练Alexnet
        
        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        if not os.path.exists(flags.model_path):
            os.mkdir(flags.model_path)

    def __del__(self):
        print('release source')

    def configure(self, flags):
        self.flags_log = os.path.join(flags.logs, '%s.txt'%(flags.method))
        self.activate_load_model = False

    def setup_path(self, flags):
        self.best_accuracy_val = -1
        self.train_loaders, self.meta_test_loaders, self.eval_loaders, self.test_loaders, self.num_domain = get_data_loader(flags)

        self.num_test_domain = 0
        self.num_train_domain = self.num_domain - self.num_test_domain

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log('get data loaders', flags_log)

    def init_network_parameter(self, flags):

        self.weight_decay = 5e-5
        self.batch_size = flags.batch_size

        self.h = 64  # 64   1024pu 表现好
        self.hh = 16

        ######################################################
        # self.feature_extractor_network = alexnet(pretrained=True).cuda()
        # self.feature_extractor_network = mymodels.CNN_1d.CNN(in_channel=1).cuda()
        self.feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1,
                                                                             out_channel=flags.num_classes).cuda()

        # self.feature_extractor_network.apply(weights_init)
        # phi means the classifer network parameter, from h (the output feature layer of input data) to c (the number of classes).
        self.para_theta = self.feature_extractor_network.parameters()   # nn.parameters()将‘参数’转为可不断优化的参数
        self.phi = classifier_homo(self.h, flags.num_classes)
        self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr, weight_decay=self.weight_decay, momentum=0.9)
        self.opt_theta = torch.optim.SGD(self.para_theta, lr=flags.lr,
                                         weight_decay=self.weight_decay, momentum=0.9)
        # self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=flags.lr,
        #                             weight_decay=self.weight_decay)
        # self.opt_theta = torch.optim.Adam(self.para_theta, lr=flags.lr,
        #                                   weight_decay=self.weight_decay)

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def pre_train(self, flags):
        model_path = os.path.join(flags.load_path, 'best_model.tar')
        if os.path.exists(model_path):
            self.load_state_dict(state_dict=model_path)

    def reinit_network_P(self,flags):
        self.beta = flags.beta
        self.alpha = flags.lr
        self.eta = flags.lr
        self.omega_para = flags.omega
        self.heldout_p = flags.heldout_p
        if flags.method == 'Feature_Critic':
            self.opt_omega = torch.optim.SGD(self.omega.parameters(), lr=self.omega_para, weight_decay=self.weight_decay, momentum=0.9)
        # self.opt_omega = torch.optim.Adam(self.omega.parameters(), lr=self.omega_para,
        #                                   weight_decay=self.weight_decay)

    def load_state_dict(self, state_dict=''):

        tmp = torch.load(state_dict)
        pretrained_dict = tmp[0]
        # load the new state dict
        self.feature_extractor_network.load_state_dict(pretrained_dict)
        self.phi.load_state_dict(tmp[1])

    def heldout_test(self, flags):

        # load the best model on the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)
        self.feature_extractor_network.eval()
        self.phi.eval()
        with torch.no_grad():
            for count, bat in enumerate(self.test_loaders):
                test_preds = []
                test_labels = []
                for data in bat:
                    x_data = data[0].cuda().float()
                    y_label = data[1].view(len(data[1]), 1)
                    y_data = torch.zeros(y_label.shape[0], flags.num_classes).scatter_(1, y_label, 1)
                    y_data = y_data.numpy()
                    if flags.method == 'Feature_Critic':
                        classifier_out = self.phi(self.feature_extractor_network(x_data)).data.cpu().numpy()
                    else:
                        classifier_out = self.feature_extractor_network(x_data).data.cpu().numpy()
                    test_preds.append(classifier_out)
                    test_labels.append(y_data)

                test_classifier_output = np.concatenate(test_preds)
                labels_test = np.concatenate(test_labels)
                torch.cuda.empty_cache()
                accuracy = compute_accuracy(predictions=test_classifier_output, labels=labels_test)
                # precision = np.mean(test_classifier_output == labels_test)
                print(accuracy)
                # print(precision)
                # accuracy
                # accuracy_info = '\n the test domain %s' % (flags.test_envs[count])
                accuracy_info = '\n the test task'
                flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
                write_log(accuracy_info, flags_log)
                write_log(accuracy, flags_log)
                # write_log(precision, flags_log)

    def validate_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count), count=count)

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            if flags.method == 'baseline':
                torch.save((self.feature_extractor_network.state_dict(), self.phi.state_dict()), outfile)
            if flags.method == 'Feature_Critic':
                torch.save((self.feature_extractor_network.state_dict(), self.phi.state_dict(), self.omega.state_dict()), outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None, count=0):
        # args, iterations, words in logs,
        self.feature_extractor_network.eval()
        self.phi.eval()
        with torch.no_grad():
            if batImageGenTest is None:
                batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', metatest=False, b_unfold_label=False)

            test_preds = []
            test_labels = []
            for data in batImageGenTest:
                x_data = data[0].cuda().float()
                y_label = data[1].view(len(data[1]), 1)
                y_data = torch.zeros(y_label.shape[0], flags.num_classes).scatter_(1, y_label, 1)
                y_data = y_data.numpy()
                if flags.method == 'Feature_Critic':
                    classifier_out = self.phi(self.feature_extractor_network(x_data)).data.cpu().numpy()
                else:
                    classifier_out = self.feature_extractor_network(x_data).data.cpu().numpy()
                test_preds.append(classifier_out)
                test_labels.append(y_data)

            # concatenate the test predictions first
            predictions = np.concatenate(test_preds)
            labels = np.concatenate(test_labels)
            accuracy = compute_accuracy(predictions=predictions, labels=labels)
            print('----------accuracy test of domain----------:', accuracy)

            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

            log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
            write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)

        return accuracy

    def train_loss(self, flags):
        # the algorithm is the same as the FC-algorithm
        write_log(flags, self.flags_log)  # 输出配置信息
        self.pre_train(flags)  # 路径准备
        self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr / 10, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr / 10,
                                                 weight_decay=self.weight_decay, momentum=0.9)
                # self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=flags.lr/10,
                #                             weight_decay=self.weight_decay)
                # self.opt_theta = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr/10,
                #                                   weight_decay=self.weight_decay)

            self.feature_extractor_network.train()  # 特征提取
            self.phi.train()  # 分类器

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]
            # meta_train_idx = index[0:]
            # meta_test_idx = index[0:]
            # meta_train_idx = index[0:5]
            # meta_test_idx = index[5:]
            write_log('-----------------iteration_%d--------------' % (_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_train_loss_dg = 0.0
                meta_loss_held_out = 0.0

                for i in meta_train_idx:
                    minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    feat_a = self.feature_extractor_network(x_subset_a)  # 提取特征
                    pred_a = self.phi(feat_a)  # 预测类别

                    loss_main = self.ce_loss(pred_a, y_subset_a)  # 对训练数据，计算ce_loss eq.1
                    meta_train_loss_main += loss_main  # 特征提取器、分类的loss

                    self.omega = Critic_Network_MLP(self.h, self.hh).cuda()
                    loss_dg = self.beta * self.omega(feat_a)  # aux_loss eq.6/7

                    meta_train_loss_dg += loss_dg  # 辅助loss

                self.opt_phi.zero_grad()
                self.opt_theta.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)

                grad_theta = {j: theta_i.grad for (j, theta_i) in
                              self.feature_extractor_network.named_parameters()}  # 记录网络参数的梯度
                theta_updated_old = {}  # 对应算法theta(old)的计算，相当于自己实现了opt.step()
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''
                # Todo: fix the new running_mean and running_var
                # Because Resnet18 network contains BatchNorm structure, there is no gradient in BatchNorm with running_mean and running_var.
                # Therefore, these two factors should be avoided in the calculation process of theta_old and theta_new.
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):
                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_old[k] = v
                        else:
                            theta_updated_old[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                meta_train_loss_dg.backward(create_graph=True)

                grad_theta = {m: theta_i.grad for (m, theta_i) in self.feature_extractor_network.named_parameters()}
                theta_updated_new = {}  # 对应算法theta(old)的计算
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_new[k] = v
                        else:
                            theta_updated_new[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                temp_new_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained,
                                                                                         in_channel=1).cuda()
                fix_nn(temp_new_feature_extractor_network, theta_updated_new)  # 把更新后的new网络参数保存下来
                temp_new_feature_extractor_network.train()

                temp_old_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained,
                                                                                         in_channel=1).cuda()
                fix_nn(temp_old_feature_extractor_network, theta_updated_old)
                # temp_old_feature_extractor_network.load_state_dict(theta_updated_old)   # 常规网络，故不用自编函数保存参数
                temp_old_feature_extractor_network.train()

                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    feat_b_old = temp_old_feature_extractor_network(x_subset_b).detach()
                    feat_b_new = temp_new_feature_extractor_network(x_subset_b).detach()

                    cls_b_old = self.phi(feat_b_old)
                    cls_b_new = self.phi(feat_b_new)

                    loss_main_old = self.ce_loss(cls_b_old, y_subset_b)
                    loss_main_new = self.ce_loss(cls_b_new, y_subset_b)
                    reward = loss_main_old - loss_main_new  # 新、旧ce_loss的差
                    # calculate the updating rule of omega, here is the max function of h.
                    utility = torch.tanh(reward)  # eq.5
                    # so, here is the min value transfering to the backpropogation.
                    loss_held_out = - utility.sum()  # eq.4
                    meta_loss_held_out += loss_held_out * self.heldout_p  # meta-test loss，用于更新feature-critic

                self.opt_theta.step()
                self.opt_phi.step()

                self.opt_omega.zero_grad()
                meta_loss_held_out.backward()
                self.opt_omega.step()
                torch.cuda.empty_cache()  # 释放GPU显存空间

                print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                      meta_train_loss_dg.data.cpu().numpy(),
                      meta_loss_held_out.data.cpu().numpy(),
                      )

            if _ % 50 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60  # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (
                _, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()

    def train_mldg(self, flags):

        write_log(flags, self.flags_log)  # 输出配置信息
        self.pre_train(flags)  # 路径准备
        self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        # self.model_all = torch.nn.Sequential(self.feature_extractor_network, self.phi)
        # self.opt_all = torch.optim.SGD(self.model_all.parameters(), lr=flags.lr, weight_decay=self.weight_decay, momentum=0.9)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr / 10, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr / 10,
                                                 weight_decay=self.weight_decay, momentum=0.9)

            self.feature_extractor_network.train()  # 特征提取
            # self.phi.train()  # 分类器
            # self.model_all.train()


            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]
            # meta_train_idx = index[0:]
            # meta_test_idx = index[0:]
            write_log('-----------------iteration_%d--------------' % (_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_loss_held_out = 0.0

                self.opt_theta.zero_grad()
                for p in self.feature_extractor_network.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                for i in meta_train_idx:
                    minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    inner_net = copy.deepcopy(self.feature_extractor_network)
                    inner_opt = torch.optim.SGD(inner_net.parameters(), lr=flags.lr, weight_decay=self.weight_decay,
                                                momentum=0.9)

                    inner_obj = self.ce_loss(inner_net(x_subset_a), y_subset_a)
                    # feat_a = self.feature_extractor_network(x_subset_a)  # 提取特征
                    # pred_a = self.phi(feat_a)  # 预测类别
                    # loss_main = self.ce_loss(pred_a, y_subset_a)  # 对训练数据，计算ce_loss eq.1

                    meta_train_loss_main += inner_obj  # 特征提取器、分类的loss

                inner_opt.zero_grad()
                # self.model_all.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)
                inner_opt.step()
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''

                for p_tgt, p_src in zip(self.feature_extractor_network.parameters(),
                                        inner_net.parameters()):
                    if p_src.grad is not None:
                        # p_tgt.grad.data.add_(p_src.grad.data / len(meta_train_idx))   # add_：加和，存到原向量中, network/tgt中的梯度原本为零的
                        p_tgt.grad.data.add_(p_src.grad.data)

                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    loss_b = self.ce_loss(inner_net(x_subset_b), y_subset_b)
                    # loss_b.requires_grad_(True)
                    meta_loss_held_out += loss_b

                grad_b = torch.autograd.grad(meta_loss_held_out, inner_net.parameters(),
                                             allow_unused=True)

                for p, g_j in zip(self.feature_extractor_network.parameters(), grad_b):
                    if g_j is not None:
                        # pass
                        # p.grad.data.add_(self.args.mldg_beta * g_j.data / num_mb)
                        p.grad.data.add_(self.beta * g_j.data)

                self.opt_theta.step()
                # self.opt_phi.step()
                # self.opt_all.step()

                torch.cuda.empty_cache()  # 释放GPU显存空间

                print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                      # meta_train_loss_dg.data.cpu().numpy(),
                      meta_loss_held_out.data.cpu().numpy(),
                      )

            if _ % 50 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60  # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (
                _, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()

    def train_irm(self, flags):

        write_log(flags, self.flags_log)  # 输出配置信息
        self.pre_train(flags)  # 路径准备
        self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr / 10, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr / 10,
                                                 weight_decay=self.weight_decay, momentum=0.9)

            self.feature_extractor_network.train()  # 特征提取

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]
            # meta_train_idx = index[0:]
            # meta_test_idx = index[0:]
            write_log('-----------------iteration_%d--------------' % (_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_loss_held_out = 0.0

                self.opt_theta.zero_grad()
                for p in self.feature_extractor_network.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                inner_net = copy.deepcopy(self.feature_extractor_network)
                inner_opt = torch.optim.SGD(inner_net.parameters(), lr=flags.lr, weight_decay=self.weight_decay,
                                            momentum=0.9)

                minibatches_device = [data for data in next(train_minibatches_iterator)]
                for i in meta_train_idx:
                    # minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    logits = inner_net(x_subset_a)
                    inner_obj = self.ce_loss(logits, y_subset_a)
                    meta_train_loss_main += inner_obj

                    scale = torch.tensor(1.).cuda().requires_grad_()
                    loss_g = self.ce_loss(logits * scale, y_subset_a)
                    grad = torch.autograd.grad(loss_g, [scale], create_graph=True)[0]
                    loss_irm = torch.sum(grad ** 2)
                    meta_train_loss_main += 2 * loss_irm

                meta_train_loss_main /= len(meta_train_idx)

                inner_opt.zero_grad()
                # self.model_all.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)
                inner_opt.step()

                for p_tgt, p_src in zip(self.feature_extractor_network.parameters(),
                                        inner_net.parameters()):
                    if p_src.grad is not None:
                        # pass
                        # p_tgt.grad.data.add_(p_src.grad.data / len(meta_train_idx))
                        p_tgt.grad.data.add_(p_src.grad.data)

                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    logits_b = inner_net(x_subset_b)
                    loss_b = self.ce_loss(logits_b, y_subset_b)
                    meta_loss_held_out += loss_b

                grad_b = torch.autograd.grad(meta_loss_held_out, inner_net.parameters(),
                                             allow_unused=True)

                for p, g_j in zip(self.feature_extractor_network.parameters(), grad_b):
                    if g_j is not None:
                        # pass
                        # p.grad.data.add_(self.alpha * g_j.data / len(meta_test_idx))
                        p.grad.data.add_(self.alpha * g_j.data)

                self.opt_theta.step()
                # self.opt_phi.step()
                # self.opt_all.step()

                torch.cuda.empty_cache()  # 释放GPU显存空间

                print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                      # meta_train_loss_dg.data.cpu().numpy(),
                      meta_loss_held_out.data.cpu().numpy(),
                      )

            if _ % 50 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60  # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (
                _, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()


class Model_Feature_Critic_PACS(ModelBaseline_PACS):
    def __init__(self, flags):
        ModelBaseline_PACS.__init__(self, flags)
        self.init_dg_function(flags)

    def __del__(self):
        print('release source')

    def init_dg_function(self,flags):
        self.dg_function = {'MLP': 1, 'Flatten_FTF': 2}
        self.id_dg = self.dg_function[flags.type]

        if self.id_dg == 1:
            self.omega = Critic_Network_MLP(self.h, self.hh).cuda()
        if self.id_dg == 2:
            self.omega = Critic_Network_Flatten_FTF(self.h, self.hh).cuda()

    def train(self, flags):

        write_log(flags, self.flags_log)   # 输出配置信息
        self.pre_train(flags)    # 路径准备
        self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr/100, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr/100,
                                                 weight_decay=self.weight_decay, momentum=0.9)
                # self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=flags.lr/10,
                #                             weight_decay=self.weight_decay)
                # self.opt_theta = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr/10,
                #                                   weight_decay=self.weight_decay)

            self.feature_extractor_network.train()    # 特征提取
            self.phi.train()                          # 分类器

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]
            # meta_train_idx = index[0:]
            # meta_test_idx = index[0:]
            # meta_train_idx = index[0:5]
            # meta_test_idx = index[5:]
            write_log('-----------------iteration_%d--------------'%(_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_train_loss_dg = 0.0
                meta_loss_held_out = 0.0

                for i in meta_train_idx:
                    minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    feat_a = self.feature_extractor_network(x_subset_a)   # 提取特征
                    pred_a = self.phi(feat_a)   # 预测类别

                    loss_main = self.ce_loss(pred_a, y_subset_a)  # 对训练数据，计算ce_loss eq.1
                    meta_train_loss_main += loss_main    # 特征提取器、分类的loss
                    if self.id_dg == 1:
                        loss_dg = self.beta * self.omega(feat_a)  # aux_loss eq.6/7
                    if self.id_dg == 2:
                        loss_dg = self.beta * self.omega(torch.matmul(torch.transpose(feat_a, 0, 1), feat_a).view(1, -1))

                    meta_train_loss_dg += loss_dg  # 辅助loss

                self.opt_phi.zero_grad()
                self.opt_theta.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)

                grad_theta = {j: theta_i.grad for (j, theta_i) in self.feature_extractor_network.named_parameters()}   # 记录网络参数的梯度
                theta_updated_old = {}   # 对应算法theta(old)的计算，相当于自己实现了opt.step()
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''
                # Todo: fix the new running_mean and running_var
                # Because Resnet18 network contains BatchNorm structure, there is no gradient in BatchNorm with running_mean and running_var.
                # Therefore, these two factors should be avoided in the calculation process of theta_old and theta_new.
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):
                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_old[k] = v
                        else:
                            theta_updated_old[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                meta_train_loss_dg.backward(create_graph=True)

                grad_theta = {m: theta_i.grad for (m, theta_i) in self.feature_extractor_network.named_parameters()}
                theta_updated_new = {}   # 对应算法theta(old)的计算
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_new[k] = v
                        else:
                            theta_updated_new[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                temp_new_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1).cuda()
                fix_nn(temp_new_feature_extractor_network, theta_updated_new)   # 把更新后的new网络参数保存下来
                temp_new_feature_extractor_network.train()

                temp_old_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1).cuda()
                fix_nn(temp_old_feature_extractor_network, theta_updated_old)
                # temp_old_feature_extractor_network.load_state_dict(theta_updated_old)   # 常规网络，故不用自编函数保存参数
                temp_old_feature_extractor_network.train()

                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    feat_b_old = temp_old_feature_extractor_network(x_subset_b).detach()
                    feat_b_new = temp_new_feature_extractor_network(x_subset_b).detach()

                    cls_b_old = self.phi(feat_b_old)
                    cls_b_new = self.phi(feat_b_new)

                    loss_main_old = self.ce_loss(cls_b_old, y_subset_b)
                    loss_main_new = self.ce_loss(cls_b_new, y_subset_b)
                    reward = loss_main_old - loss_main_new   # 新、旧ce_loss的差
                    # calculate the updating rule of omega, here is the max function of h.
                    utility = torch.tanh(reward)   # eq.5
                    # so, here is the min value transfering to the backpropogation.
                    loss_held_out = - utility.sum()  # eq.4
                    meta_loss_held_out += loss_held_out*self.heldout_p  # meta-test loss，用于更新feature-critic

                self.opt_theta.step()
                self.opt_phi.step()

                self.opt_omega.zero_grad()
                meta_loss_held_out.backward()
                self.opt_omega.step()
                torch.cuda.empty_cache()   # 释放GPU显存空间

                # print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                #           meta_train_loss_dg.data.cpu().numpy(),
                #           meta_loss_held_out.data.cpu().numpy(),)

            if _ % 100 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60    # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (_, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()

    def kd(self, data1, label1, data2, label2, bool_indicator, n_class=3, temperature=2.0):
        kd_loss = 0.0
        eps = 1e-16

        prob1s = []
        prob2s = []

        for cls in range(n_class):  # 前四行计算条件概率，对应paper公式2
            mask1 = torch.tile(torch.unsqueeze(label1[:, cls], -1), [1, n_class])  # one-hot 堆叠/广播（增维）
            logits_sum1 = torch.sum(torch.multiply(data1, mask1), dim=0)  # 逐元素相乘，相加
            num1 = torch.sum(label1[:, cls])  # mini-batch中，类别cls的概率和
            activations1 = logits_sum1 * 1.0 / (num1 + eps)  # add eps for prevent un-sampled class resulting in NAN
            prob1 = torch.nn.functional.softmax(activations1 / temperature, dim=0)  # paper公式3
            prob1 = torch.clamp(prob1, min=1e-8, max=1.0)  # for preventing prob=0 resulting in NAN/把prob1压缩在min-max中

            mask2 = torch.tile(torch.unsqueeze(label2[:, cls], -1), [1, n_class])
            logits_sum2 = torch.sum(torch.multiply(data2, mask2), dim=0)
            num2 = torch.sum(label2[:, cls])
            activations2 = logits_sum2 * 1.0 / (num2 + eps)
            prob2 = torch.nn.functional.softmax(activations2 / temperature)
            prob2 = torch.clamp(prob2, min=1e-8, max=1.0)

            KL_div = (torch.sum(prob1 * torch.log(prob1 / prob2)) + torch.sum(
                prob2 * torch.log(prob2 / prob1))) / 2.0  # KL散度，公式4
            kd_loss += KL_div * bool_indicator[cls]

            prob1s.append(prob1)
            prob2s.append(prob2)

        kd_loss = kd_loss / n_class

        return kd_loss  # , prob1s, prob2s

    def train_newloss(self, flags):   # fc + irm

        write_log(flags, self.flags_log)   # 输出配置信息
        self.pre_train(flags)    # 路径准备
        self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr/100, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr/100,
                                                 weight_decay=self.weight_decay, momentum=0.9)

            self.feature_extractor_network.train()    # 特征提取
            self.phi.train()                          # 分类器

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]

            write_log('-----------------iteration_%d--------------'%(_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_train_loss_dg = 0.0
                meta_loss_held_out = 0.0
                meta_global_loss = 0.0

                mtr_save = []

                for i in meta_train_idx:
                    minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    mtr_save.append(minibatches_device[i])
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    feat_a = self.feature_extractor_network(x_subset_a)   # 提取特征
                    pred_a = self.phi(feat_a)   # 预测类别

                    loss_main = self.ce_loss(pred_a, y_subset_a)  # 对训练数据，计算ce_loss eq.1
                    meta_train_loss_main += loss_main    # 特征提取器、分类的loss

                    if self.id_dg == 1:
                        loss_dg = self.beta * self.omega(feat_a)  # aux_loss eq.6/7
                    if self.id_dg == 2:
                        loss_dg = self.beta * self.omega(torch.matmul(torch.transpose(feat_a, 0, 1), feat_a).view(1, -1))

                    meta_train_loss_dg += loss_dg  # 辅助loss //在这里加loss_irm 就会out of memory

                # meta_train_loss_main /= len(meta_train_idx)

                self.opt_phi.zero_grad()
                self.opt_theta.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)

                grad_theta = {j: theta_i.grad for (j, theta_i) in self.feature_extractor_network.named_parameters()}   # 记录网络参数的梯度
                theta_updated_old = {}   # 对应算法theta(old)的计算，相当于自己实现了opt.step()
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''
                # Todo: fix the new running_mean and running_var
                # Because Resnet18 network contains BatchNorm structure, there is no gradient in BatchNorm with running_mean and running_var.
                # Therefore, these two factors should be avoided in the calculation process of theta_old and theta_new.
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):
                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_old[k] = v
                        else:
                            theta_updated_old[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                grad_theta = {j: theta_i.grad for (j, theta_i) in self.phi.named_parameters()}  # 记录网络参数的梯度
                phi_update = {}

                for i, (k, v) in enumerate(self.phi.state_dict().items()):
                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            phi_update[k] = v
                        else:
                            phi_update[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                meta_train_loss_dg.backward(create_graph=True)

                grad_theta = {m: theta_i.grad for (m, theta_i) in self.feature_extractor_network.named_parameters()}
                theta_updated_new = {}   # 对应算法theta(old)的计算
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_new[k] = v
                        else:
                            theta_updated_new[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                temp_new_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1)
                fix_nn(temp_new_feature_extractor_network, theta_updated_new)   # 把更新后的new网络参数保存下来
                temp_new_feature_extractor_network.train()
                temp_new_feature_extractor_network.requires_grad_()

                temp_old_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1)
                fix_nn(temp_old_feature_extractor_network, theta_updated_old)
                # temp_old_feature_extractor_network.load_state_dict(theta_updated_old)   # 常规网络，故不用自编函数保存参数
                temp_old_feature_extractor_network.train()
                temp_old_feature_extractor_network.requires_grad_()

                temp_update_classifier = classifier(flags.num_classes)
                fix_nn(temp_update_classifier, phi_update, phi=True)
                # temp_old_feature_extractor_network.load_state_dict(theta_updated_old)   # 常规网络，故不用自编函数保存参数
                temp_update_classifier.train()
                temp_update_classifier.requires_grad_()

                net_new = torch.nn.Sequential(temp_old_feature_extractor_network, temp_update_classifier)
                net_old = torch.nn.Sequential(self.feature_extractor_network, self.phi)
                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    feat_b_old = temp_old_feature_extractor_network(x_subset_b).detach()
                    feat_b_new = temp_new_feature_extractor_network(x_subset_b).detach()

                    cls_b_old = self.phi(feat_b_old)
                    cls_b_new = self.phi(feat_b_new)

                    loss_main_old = self.ce_loss(cls_b_old, y_subset_b)
                    loss_main_new = self.ce_loss(cls_b_new, y_subset_b)
                    reward = loss_main_old - loss_main_new   # 新、旧ce_loss的差
                    # calculate the updating rule of omega, here is the max function of h.
                    utility = torch.tanh(reward)   # eq.5
                    # so, here is the min value transferring to the back-propagation.
                    loss_held_out = - utility.sum()  # eq.4
                    meta_loss_held_out += loss_held_out*self.heldout_p  # meta-test loss，用于更新feature-critic

                    for mtr in mtr_save:
                        fea_a = temp_old_feature_extractor_network(mtr[0].cuda().float())
                        # fea_a.requires_grad_()
                        label_a = torch.nn.functional.one_hot(mtr[1].cuda().long(), num_classes=flags.num_classes)
                        pred_a = temp_update_classifier(fea_a)

                        fea_b = temp_old_feature_extractor_network(x_subset_b)
                        # fea_b.requires_grad_()
                        label_b = torch.nn.functional.one_hot(y_subset_b, num_classes=flags.num_classes)
                        pred_b = temp_update_classifier(fea_b)

                        bool_instructor = [1.0] * flags.num_classes

                        global_loss = self.kd(pred_a, label_a, pred_b, label_b, bool_instructor, flags.num_classes)

                        meta_global_loss += global_loss

                meta_global_loss /= 3*len(meta_test_idx)

                grad_theta = torch.autograd.grad(meta_global_loss, net_new.parameters(),
                                                 allow_unused=True,
                                                 # retain_graph=True
                                                 )

                for p, g_j in zip(net_old.parameters(), grad_theta):
                    if g_j is not None:
                        # pass
                        # p.grad.data.add_(self.args.mldg_beta * g_j.data / num_mb)
                        p.grad.data.add_(self.beta * 50 * g_j.data)

                self.opt_theta.step()
                self.opt_phi.step()

                self.opt_omega.zero_grad()
                meta_loss_held_out.backward()
                self.opt_omega.step()
                torch.cuda.empty_cache()   # 释放GPU显存空间

                # print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                #           meta_train_loss_dg.data.cpu().numpy(),
                #           meta_loss_held_out.data.cpu().numpy(),)

            if _ % 200 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60    # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (_, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()



