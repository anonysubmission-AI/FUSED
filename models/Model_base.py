#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import torchvision
from peft import LoraConfig, get_peft_model
from utils import *
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    @staticmethod
    def split_weight_name(name):
        if 'weight' or 'bias' in name:
            return ''.join(name.split('.')[:-1])
        return name


class Fused(nn.Module):
    def __init__(self, args, global_model):
        super(Fused, self).__init__()
        if args.data_name == 'cifar10':
            # global_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            # num_ftrs = global_model.fc.in_features
            # global_model.fc = nn.Linear(num_ftrs, args.num_classes)

            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
            # target_modules = ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
            target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
        elif args.data_name == 'cifar100':
            global_model = torchvision.models.resnet18(pretrained=True)
            num_ftrs = global_model.fc.in_features
            global_model.fc = nn.Linear(num_ftrs, args.num_classes)
            # global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))

            # target_modules = ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
            target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
        elif args.data_name == 'fashionmnist':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))

            target_modules = ['conv1', 'fc3']
        elif args.data_name == 'adult':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
            target_modules = ['fc3']

        config = LoraConfig(
        r = 16,
        lora_alpha = 32,
        target_modules = target_modules,
        lora_dropout = 0.1,
        bias = "none",
        )
        self.lora_model = get_peft_model(global_model, config)
        for name, param in self.lora_model.named_parameters():
            if not any(target in name for target in config.target_modules):
                param.requires_grad = False

    def forward(self, x):
        return self.lora_model(x)