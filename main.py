import argparse
import copy

from dataset.generate_data import data_init
import torch

from algs import fused
from utils import *
import random
import numpy as np
from models.Model_base import Fused

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=False, default='CNN_Cifar10', help= 'choose a model: LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100,TextCNN')
    parser.add_argument('--data_name', type=str, required=False, default='cifar10', help= 'choose: mnist, fashionmnist, purchase, adult, cifar10, text')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--train_with_test', default=True, type=bool, help='')
    parser.add_argument('--temperature', default=0.5, type=float, help='the temperature for distillation loss')
    parser.add_argument('--max_checkpoints', default=3, type=int)
    # ======================unlearning setting==========================
    parser.add_argument('--forget_paradigm', default='class', type=str, help='choose from client or class')
    parser.add_argument('--paradigm', default='fused', type=str,
                        help='choose the training paradigm:fused, federaser, retrain, infocom22, exactfun, fl, eraseclient')
    parser.add_argument('--forget_client_idx', type=list, default=[0])#[0], [0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]
    parser.add_argument('--forget_class_idx', type=list, default=[0])#[0],[0,4],[1,3,5],[2,5,7,8],[2,6,7,8,9]
    parser.add_argument('--if_retrain', default=False, type=bool, help='')
    parser.add_argument('--if_unlearning', default=False, type=bool, help='')
    parser.add_argument('--baizhanting', default=True, type=bool, help='')
    parser.add_argument('--backdoor', default=False, type=bool, help='')
    parser.add_argument('--backdoor_frac', default=0.2, type=float, help='')
    # ======================batch size setting===========================
    parser.add_argument('--local_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    # ======================training epoch===========================
    parser.add_argument('--global_epoch', default=10, type=int)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--distill_epoch', default=10, type=int)
    parser.add_argument('--distill_pretrain_epoch', default=2, type=int)
    parser.add_argument('--fraction', default=1.0, type=float, help='the fraction of training data')
    parser.add_argument('--num_user', default=50, type=int)
    # ======================data process============================
    parser.add_argument('--niid', default=True, type=bool, help='')
    parser.add_argument('--balance', default=True, type=bool, help='')
    parser.add_argument('--partition', default='dir', type=str, help='choose from pat or dir')
    parser.add_argument('--alpha', default=1.0, type=float, help='for Dirichlet distribution')#alpha越小，数据越倾斜
    parser.add_argument('--proxy_frac', default=0.01, type=float, help='the fraction of training data')
    parser.add_argument('--seed', default=50, type=int)
    # ======================federaser========================
    parser.add_argument('--unlearn_interval', default=1, type=int, help='')
    parser.add_argument('--forget_local_epoch_ratio', default=0.2, type=float)

    # ======================eraseclient========================
    parser.add_argument('--epoch_unlearn', default=20, type=int, help='')
    parser.add_argument('--num_iterations', default=50, type=int, help='')


    args = parser.parse_args()
    return args
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Parameters Setting
    args = get_args()
    # set_random_seed(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model Setting
    model = model_init(args)
    # draw_model(args, model)
    # Data Setting
    client_all_loaders, test_loaders, proxy_loader = data_init(args)

    args.if_unlearning = False
    case = fused.FUSED(args)

    if args.forget_paradigm == 'client':
        client_all_loaders_process, test_loaders_process = baizhanting_attack(args, copy.deepcopy(client_all_loaders), copy.deepcopy(test_loaders))
        model, all_client_models = case.train_normal(model, client_all_loaders_process, proxy_loader, test_loaders_process)
        args.if_unlearning = True
        unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders), proxy_loader,
                                                    test_loaders_process)
    elif args.forget_paradigm == 'class':
        model, all_client_models = case.train_normal(model, copy.deepcopy(client_all_loaders), proxy_loader, test_loaders)
        args.if_unlearning = True
        for user in range(args.num_user):
            train_ls = []
            for data, target in client_all_loaders[user]:
                data = data.tolist()
                targets = target.tolist()
                for idx, label in enumerate(targets):
                    if label in args.forget_class_idx:
                        label_ls = [i for i in range(args.num_classes)]
                        label_ls.remove(label)
                        inverse_label = np.random.choice(label_ls)
                        label = inverse_label
                    train_ls.append((torch.tensor(data[idx]), torch.tensor(label)))
            train_loader = DataLoader(train_ls, batch_size=args.test_batch_size, shuffle=True)
            client_all_loaders[user] = train_loader
        client_all_loaders_process = erase_forget_class(args, client_all_loaders)
        unlearning_model = case.forget_class(copy.deepcopy(model), client_all_loaders_process, test_loaders)
    elif args.forget_paradigm == 'sample':
        client_all_loaders_process = backdoor_attack(args, copy.deepcopy(client_all_loaders))
        model, all_client_models = case.train_normal(model, client_all_loaders_process, proxy_loader, test_loaders)
        args.if_unlearning = True
        client_all_loaders_process = erase_backdoor(args, copy.deepcopy(client_all_loaders))
        unlearning_model = case.forget_sample(copy.deepcopy(model), client_all_loaders_process, test_loaders)