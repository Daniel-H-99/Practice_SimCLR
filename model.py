import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

class ShortCut(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ShortCut, self).__init__()
        self.downsample = input_channels != output_channels
        if self.downsample:
            assert output_channels == 2 * input_channels
            self.layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=2, bias=False), nn.BatchNorm2d(output_channels))
    def forward(self, input):
        if self.downsample:
            return self.layers(input)
        else:
            return input
            
class Res18Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Res18Block, self).__init__()
        self.bn = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.bn1 = self.bn(output_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=int(output_channels / input_channels), padding=1, bias=False)
        self.bn2 = self.bn(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1, bias=False)
        self.shortcut = ShortCut(input_channels, output_channels)
    def forward(self, input):
        residual = self.shortcut(input)
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = output + residual
        output = self.relu(output)
        return output
                           
class Res18ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, shape):
        super(Res18ConvLayer, self).__init__()
        self.layers = nn.Sequential(Res18Block(input_channels, output_channels), Res18Block(output_channels, output_channels))
        self.shape = shape
    def forward(self, input):
        assert input.shape[1:] == self.shape
        return self.layers(input)
                           
class Res18(nn.Module):
    def __init__(self, args):
        super(Res18, self).__init__()
        shape = (64, args.width // 4, args.height // 4)
        conv_configs = [(64, 64), (64, 128), (128, 256), (256, 512)]
        self.layers = nn.Sequential(*(Res18ConvLayer(*conv_configs[i], (shape[0] * 2 ** max(0, i - 1), shape[1] * 2 ** (-max(0, i - 1)), shape[2] * 2 ** (-max(0, i - 1)))) for i in range(len(conv_configs))))
        self.pre_layers = nn.Sequential(nn.Conv2d(args.channels, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.post_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.is_finetune = False
        for m in self.modules():        # initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if args.triple_layer_projection:
            self.post_fc0 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        if args.use_z:
            self.post_fc = nn.Linear(args.out_dim, args.classes)
        else:
            self.post_fc = nn.Linear(512, args.classes)
        self.post_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, args.out_dim))
        self.args = args
    def forward(self, input):
        output = self.pre_layers(input)
        output = self.layers(output)
        output = self.post_pool(output)
        output = output.flatten(1)
        if self.args.triple_layer_projection:           # 3-layer projection
            output = self.post_fc0(output)
        if (not self.is_finetune) or self.args.use_z:   # for a case using z instead of h
            output = self.post_proj(output)
        if not self.is_finetune:                        # calculate cosine similarity
            output = self.similarity(output)           
        if self.is_finetune:                            # for finetuning, use fc
            output = self.post_fc(output)
        return output
    def similarity(self, output):   # calculate cosine similarity  which will be passed to CrossEntropyLoss
        output = output / output.norm(p=2, dim=1, keepdim=True)
        output = output.matmul(output.transpose(0, 1)) / self.args.temperature
        output = output[(1 - torch.eye(output.shape[0])).bool()].view(output.shape[0], output.shape[0] - 1)
        return output
    def pretrain(self):     # pretraining
        self.is_finetune = False
        for param in self.parameters():
            param.requires_grad = True
    def lin_eval(self):     # linear evaluation
        self.is_finetune = True
        for param in self.parameters():
            param.requires_grad = False
        for param in self.post_fc.parameters():
            param.requires_grad = True
    def finetune(self):     # finetuning
        self.is_finetune = True
        for param in self.parameters():
            param.requires_grad = True

                
