import time
import math
import pandas as pd
import torch

from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, Model_adults, Model_purchase
from utils import init_network, test_class_forget, test_client_forget
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
import objgraph
import matplotlib.pyplot as plt
from utils import *
import random
from models.Model_base import *

class FUSED(Base):
    def __init__(self, args):
        super(FUSED, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/fused_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}

    def train_normal(self, global_model, client_all_loaders, proxy_loader, test_loaders):
        print('\n')
        print(5 * "#" + "  FUSED Federated Training Start  " + 5 * "#")

        checkpoints_ls = []
        result_list = []
        for name, param in global_model.named_parameters():
            # print(name)
            self.param_change_dict[name] = 0
            self.param_size[name] = 0

        for epoch in range(self.args.global_epoch):
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)

            global_model = self.fedavg(client_models)

            all_idx = list(range(self.args.num_user))

            client_test_acc = []

            if self.args.forget_paradigm == 'sample':

                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                              self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero,
                                                                                 avg_test_acc))
                result_list.extend(test_result_ls)

            elif self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

        torch.save(global_model.state_dict(), 'save_model/global_model_{}.pth'.format(self.args.data_name))

        if self.args.forget_paradigm == 'sample':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        elif self.args.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Test_acc', 'Test_loss'])
        elif self.args.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        df.to_csv('./results/Acc_loss_fl_{}_data_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha))

        return global_model, client_models

    def forget_client_train(self, global_model, client_all_loaders, proxy_loader, test_loaders):
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args,
                                                                  test_loaders)
        print('Fused-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('xxxx', avg_r_acc,
                                                                                 avg_f_acc))
        fused_model = Fused(self.args, global_model)
        torch.save(fused_model.state_dict(), 'save_model/global_fusedmodel_{}.pth'.format(self.args.data_name))
        print('\n')
        print(5 * "#" + "  FUSED Federated Client Unlearning Start  " + 5 * "#")

        checkpoints_ls = []
        result_list = []
        consume_time = 0

        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = [i for i in range(self.args.num_user) if i not in self.args.forget_client_idx]
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            avg_model = self.fedavg(client_models)
            consume_time += end_time - std_time
            fused_model.load_state_dict(avg_model.state_dict())

            fused_model.eval()

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, fused_model, self.args,
                                                                      test_loaders)

            result_list.extend(test_result_ls)

            print('Fused-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(epoch, avg_r_acc,
                                                                                    avg_f_acc))


        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        df.to_csv(
            './results/client/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm,
                                                                                    self.args.data_name,
                                                                                    self.args.alpha,
                                                                                    len(self.args.forget_client_idx)))

        print(5 * "#" + "  FUSED Federated Client Unlearning End  " + 5 * "#")

        return fused_model

    def forget_class(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  FUSED Federated Class Unlearning Start  " + 5 * "#")
        num_selected_clients = self.args.num_user * self.args.forget_client_idx

        checkpoints_ls = []
        result_list = []
        consume_time = 0
        fused_model = Fused(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))

            select_client_loaders = list()
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
            std_time = time.time()

            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time

            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time
        df.to_csv(
            './results/class/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))

        print(5 * "#" + "  FUSED Federated Class Unlearning End  " + 5 * "#")
        return global_model

    def forget_sample(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  FUSED Federated Sample Unlearning Start  " + 5 * "#")

        checkpoints_ls = []
        result_list = []
        consume_time = 0
        fused_model = Fused(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))# 将需要遗忘的客户端排除在外

            self.select_forget_idx = list()
            select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time

            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        df['Comsume_time'] = consume_time
        df.to_csv(
            './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))

        print(5 * "#" + "  FUSED Federated Sample Unlearning End  " + 5 * "#")
        return global_model