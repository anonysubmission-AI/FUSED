import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, Model_adults, Model_purchase, TextCNN
import numpy as np
import torch.nn.functional as F
import copy

def model_init(args):
    if args.data_name == 'mnist':
        args.num_classes = 10
        args.lr = 0.01
        args.midimension = 84
        model = LeNet_FashionMNIST.Model(args)
        init_network(model)
    elif args.data_name == 'fashionmnist':
        args.num_classes = 10
        args.lr = 0.005
        # args.midimension = 84
        model = LeNet_FashionMNIST.Model(args)
        init_network(model)
    elif args.data_name == 'cifar10':
        args.num_classes = 10
        args.lr = 0.001
        # args.midimension = 512
        model = CNN_Cifar10.Model(args)
        init_network(model)
    elif args.data_name == 'cifar100':
        args.num_classes = 100
        args.lr = 0.01
        # args.midimension = 512
        model = CNN_Cifar100.Model(args)
        init_network(model)
    elif args.data_name == 'adult':
        args.num_classes = 2
        args.lr = 0.01
        # args.midimension = 10
        model = Model_adults.Model(args)
        init_network(model)
    elif args.data_name == 'purchase':
        args.num_classes = 100
        args.lr = 0.001
        # args.midimension = 50
        model = Model_purchase.Model(args)
        init_network(model)
    elif args.data_name == 'text':
        args.num_classes = 10
        args.lr = 0.01
        # args.midimension = 100
        embedding = 'embedding_SougouNews.npz'
        args.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None
        args.embed = args.embedding_pretrained.size(1) \
            if args.embedding_pretrained is not None else 300
        args.filter_sizes = (2, 3, 4)  # kernel size of CNN
        args.num_filters = 256
        args.dropout = 0.1
        args.vocab_path = 'dataset/vocab.pkl'
        model = TextCNN.Model(args)
        init_network(model)
    return model


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        # print('Name: ', name)
        if exclude in name:
            continue
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if 'batch' in name:
                nn.init.normal_(w)
                continue
            if method == 'xavier':
                if len(w.shape) < 2:
                    nn.init.kaiming_normal_(w.unsqueeze(0))
                else:
                    nn.init.kaiming_normal_(w)
                # nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)

def save_checkpoint(state, is_best, args, epoch, filename='checkpoint.pth'):
    if is_best == False:
        filename = 'checkpoint-{}-{}.pth'.format(args.dataset, epoch)
    elif is_best == True:
        filename = 'bestcheckpoint-{}.pth'.format(args.dataset)
    torch.save(state, filename)

def load_checkpoint(filename):
    model, optimizer = torch.load(filename)
    return model, optimizer

def test_class_forget(obj, epoch, model, args, test_loaders):
    forget_acc_ls = []
    remember_acc_ls = []
    test_result_ls = []
    for k in range(args.num_user):
        test_loader = test_loaders[k]
        label_data_dict = {}
        for data, target in test_loader:
            data = data.tolist()
            targets = target.tolist()
            for idx, label in enumerate(targets):
                if label not in label_data_dict:
                    label_data_dict[label] = []
                label_data_dict[label].append((torch.tensor(data[idx]), torch.tensor(label)))
        label_data_loaders = {}
        for label, data_list in label_data_dict.items():
            # class_dataset = Dataset(data_list)
            class_loader = DataLoader(data_list, batch_size=args.test_batch_size, shuffle=True)
            label_data_loaders[label] = class_loader
        # 测试遗忘类的准确率
        for label, loader in label_data_loaders.items():
            (test_loss, test_acc) = obj.test(model, label_data_loaders[label], args)
            test_result_ls.append([epoch, k, label, test_acc, float(test_loss)])
            if label in args.forget_class_idx:
                forget_acc_ls.append(test_acc)
            else:
                remember_acc_ls.append(test_acc)
    avg_f_acc = sum(forget_acc_ls) / len(forget_acc_ls)
    avg_r_acc = sum(remember_acc_ls) / len(remember_acc_ls)

    return avg_f_acc, avg_r_acc, test_result_ls

def test_backdoor_forget(obj, epoch, model, args, test_loaders):
    jingdu_ls = []
    acc_zero_ls = []
    test_acc_ls = []
    test_result_ls = []
    for k in range(args.num_user):
        test_loader = test_loaders[k]

        dataset_x = []
        dataset_y = []
        for data, target in test_loader:
            x_bk = copy.deepcopy(data)
            x_bk[:, :, -1, -1] = 255
            # data, target = insert_backdoor(args, data, target)
            dataset_x.extend(data.cpu().detach().numpy())
            dataset_y.extend(target.cpu().detach().numpy())
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_x = torch.Tensor(dataset_x).type(torch.float32)
        dataset_y = torch.Tensor(dataset_y).type(torch.int64)
        dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]

        test_data = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size,
                                                                 shuffle=True)
        (jingdu, acc_zero, test_acc) = obj.test(model, test_data, args)
        test_result_ls.append([epoch, k, jingdu, acc_zero, test_acc])
        jingdu_ls.append(jingdu)
        acc_zero_ls.append(acc_zero)
        test_acc_ls.append(test_acc)

    avg_jingdu = sum(jingdu_ls) / len(jingdu_ls)
    avg_acc_zero = sum(acc_zero_ls) / len(acc_zero_ls)
    avg_test_acc = sum(test_acc_ls) / len(test_acc_ls)

    return avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls


def test_client_forget(obj, epoch, model, args, test_loaders):
    all_idx = [k for k in range(args.num_user)]

    forget_acc_ls = []
    remember_acc_ls = []
    test_result_ls = []
    for forget_idx in args.forget_client_idx:
        (test_loss, test_acc) = obj.test(model, test_loaders[forget_idx], args)
        test_result_ls.append([epoch, forget_idx, test_acc, float(test_loss)])
        all_idx.remove(forget_idx)
        forget_acc_ls.append(test_acc)

    remember_idx = all_idx
    for r_idx in remember_idx:
        (test_loss, test_acc) = obj.test(model, test_loaders[r_idx], args)
        test_result_ls.append([epoch, r_idx, test_acc, float(test_loss)])
        remember_acc_ls.append(test_acc)
    avg_f_acc = sum(forget_acc_ls) / len(forget_acc_ls)
    avg_r_acc = sum(remember_acc_ls) / len(remember_acc_ls)
    return avg_f_acc, avg_r_acc, test_result_ls


def erase_forget_class(args, client_loaders):
    for user in range(args.num_user):
        dataset_image = []
        dataset_label = []
        for x, y in client_loaders[user]:
            dataset_image.extend(x)
            dataset_label.extend(y)
        data_x = []
        data_y = []
        for idx, cls in enumerate(dataset_label):
            if cls not in args.forget_class_idx:
                data_x.append(dataset_image[idx].cpu().detach().numpy())
                data_y.append(dataset_label[idx].cpu().detach().numpy())

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_x = torch.Tensor(data_x).type(torch.float32)
        data_y = torch.Tensor(data_y).type(torch.int64)
        train_data = [(x1, y1) for x1, y1 in zip(data_x, data_y)]
        client_loaders[user] = torch.utils.data.DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True)
    return client_loaders

def baizhanting_attack(args, client_loaders, test_loaders):
    for user in args.forget_client_idx:
        dataset_image = []
        dataset_label = []
        test_image = []
        test_label = []
        for x, y in client_loaders[user]:
            dataset_image.extend(x)
            dataset_label.extend(y)
        for x, y in test_loaders[user]:
            test_image.extend(x)
            test_label.extend(y)
        data_x = []
        data_y = []
        test_x = []
        test_y = []
        for idx, cls in enumerate(dataset_label):
            # label_ls = [i for i in range(args.num_classes)]
            # label_ls.remove(cls)
            # inverse_label = np.random.choice(label_ls)
            if int(dataset_label[idx]) < args.num_classes-1:
                dataset_label[idx] = int(dataset_label[idx])+1
            elif int(dataset_label[idx])==args.num_classes-1:
                dataset_label[idx] = 0
            data_x.append(np.array(dataset_image[idx]))
            data_y.append(np.array(dataset_label[idx]))

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_x = torch.Tensor(data_x).type(torch.float32)
        data_y = torch.Tensor(data_y).type(torch.int64)
        train_data = [(x1, y1) for x1, y1 in zip(data_x, data_y)]
        client_loaders[user] = torch.utils.data.DataLoader(train_data, batch_size=args.local_batch_size,
                                                           shuffle=True)
        for idx, cls in enumerate(test_label):
            if int(test_label[idx]) < args.num_classes-1:
                test_label[idx] = int(test_label[idx])+1
            elif int(test_label[idx])==args.num_classes-1:
                test_label[idx] = 0
            test_x.append(np.array(test_image[idx]))
            test_y.append(np.array(test_label[idx]))

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_x = torch.Tensor(test_x).type(torch.float32)
        test_y = torch.Tensor(test_y).type(torch.int64)
        test_data = [(x1, y1) for x1, y1 in zip(test_x, test_y)]
        test_loaders[user] = torch.utils.data.DataLoader(test_data, batch_size=args.local_batch_size,
                                                           shuffle=True)
    return client_loaders, test_loaders

def backdoor_attack(args, client_loaders):
    for client in args.forget_client_idx:
        dataset_x = []
        dataset_y = []
        for idx, (data, target) in enumerate(client_loaders[client]):
            if idx <= 1:
                data_, target_ = insert_backdoor(args, data, target)
                dataset_x.extend(data_.cpu().detach().numpy())
                dataset_y.extend(target_.cpu().detach().numpy())
            else:
                dataset_x.extend(data.cpu().detach().numpy())
                dataset_y.extend(target.cpu().detach().numpy())
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_x = torch.Tensor(dataset_x).type(torch.float32)
        dataset_y = torch.Tensor(dataset_y).type(torch.int64)
        dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]
        client_loaders[client] = torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size,
                                                             shuffle=False)
    return client_loaders

def erase_backdoor(args, client_loaders):
    for client in args.forget_client_idx:
        dataset_x = []
        dataset_y = []
        for idx, (data, target) in enumerate(client_loaders[client]):
            if idx <= 1:
                continue
            dataset_x.extend(data.cpu().detach().numpy())
            dataset_y.extend(target.cpu().detach().numpy())
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_x = torch.Tensor(dataset_x).type(torch.float32)
        dataset_y = torch.Tensor(dataset_y).type(torch.int64)
        dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]
        client_loaders[client] = torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size,
                                                             shuffle=False)
    return client_loaders

def insert_backdoor(args, data, target, trigger_label=0, trigger_pixel_value=255):
    x_bk = copy.deepcopy(data)
    # backdoor = torch.ones(4, 4) * trigger_pixel_value
    x_bk[:, :, -1, -1] = trigger_pixel_value
    y_bk = torch.full_like(target, trigger_label)
    return x_bk, y_bk

class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()

    def forward(self, output, old_target, temperature, frac):
        T = temperature
        alpha = frac
        outputs_S = F.log_softmax(output / T, dim=1)
        outputs_T = F.softmax(old_target / T, dim=1)
        l_old = outputs_T.mul(outputs_S)
        l_old = -1.0 * torch.sum(l_old) / outputs_S.shape[0]

        return l_old * alpha

