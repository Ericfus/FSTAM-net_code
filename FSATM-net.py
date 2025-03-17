"""
author:Fu Shuo
data: 2025/3/1
version: 1.0
"""
# dataset 
#https://bnci-horizon-2020.eu/database/data-sets


# remember to change paths
import numpy as np


from scipy.linalg import fractional_matrix_power
import argparse
import os
gpus = [0]
from math import log
from math import pi
from scipy.io import loadmat
import mne
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import csv
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from scipy.io import loadmat
cudnn.benchmark = False
cudnn.deterministic = True


#Define hook function
#Step 1: Define a function to receive features#
#Here we define a class that has a function called hook_fun that receives features. Defining classes is to facilitate the extraction of multiple intermediate layers.

# features_in_hook = []
features_out_hook = []

def hook(module, fea_in, fea_out):
    # features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None
def to_categorical(labels, num_classes=None, dtype="float32"):
    """
    Convert integer class labels into one hot encoding format.

    Parameters:
    labels (array-like):  Input category label (in integer form).
    Num_classes (int, optional): Total number of categories. If not specified, it will be inferred based on the input data.
    Dtype (str, optional): Output the data type of the matrix, default is "float32".

    return:
    Np.ndarray: One hot encoded array with a shape of (number of samples, num_classes).
    """
    # 转为 numpy 数组
    labels = np.array(labels, dtype="int32")

    # 自动推断 num_classes
    if num_classes is None:
        num_classes = np.max(labels) + 1

    # 初始化 one-hot 矩阵
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=dtype)

    # 设置对应位置为 1
    one_hot[np.arange(labels.shape[0]), labels] = 1

    return one_hot

#EA
def EA(x):
    """
    x : data of shape (num_samples, num_channels, num_time_samples)
    XEA : data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA

def get_data_0(subject, training, path, all_trials = True):
    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = 7 * 250

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial = 0
    if training:
        a = loadmat(path + 'A0' + str(subject) + 'T.mat')
    else:
        a = loadmat(path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_fs = a_data3[3]
        a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        a_gender = a_data3[6]
        a_age = a_data3[7]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


def prepare_features(path, subject, crossValidation=False):
    fs = 250
    t1 = int(2 * fs)
    t2 = int(6 * fs)
    T = t2 - t1
    X_train, y_train = get_data_0(subject, True, path)
    # if crossValidation:
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X_train, y_train, test_size=0.2, random_state=0)
    # else:
    X_test, y_test = get_data_0(subject, False, path)

    # prepare training data
    N_tr, N_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2]
    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    # prepare testing data
    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, t1:t2]

    #自己加的
    y_train = (y_train - 1).astype('float32')
    y_test = (y_test - 1).astype('float32')
    y_test_onehot = (y_test - 1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

class Fre_SELayer(nn.Module):
    def __init__(self, channel, window_length,outputkernel,reduction=8):
        super(Fre_SELayer, self).__init__()
        self.avg_pool = nn.AvgPool2d((1,window_length),stride = (1,1))
        self.F1 = channel
        self.convs = nn.Sequential(
            nn.Conv2d(1, outputkernel, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(outputkernel, 1, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1,eps = 0.001),
            nn.Sigmoid()
        )
        self.conv11 = nn.Conv2d(1,outputkernel,(1,1),stride = (1,1))
        self.conv12 = nn.Conv2d(outputkernel,1,(1,1),stride = (1,1))

        self.R1 = nn.LeakyReLU(inplace = True)
        self.bn1 = nn.BatchNorm2d(1,eps = 0.001)
        self.bn2 = nn.BatchNorm2d(outputkernel,eps = 0.001)
        self.ln1 = nn.LayerNorm([1,self.F1,22])
        self.ln2 = nn.LayerNorm([outputkernel,self.F1,22])
        # self.f2 = nn.Linear(channel//reduction,channel,bias = False)
        self.S = nn.Sigmoid()
    def forward(self, x):
        b, c,d, _ = x.size()
        # y = self.avg_pool(torch.pow(x,2)).view(b, c,d)
        y = self.avg_pool(x).permute([0,3,1,2])
        # y = self.convs(y).permute([0,2,3,1])
        y = self.ln1(y)
        y = self.conv11(y)
        y = self.R1(y)
        y = self.ln2(y)
        y = self.conv12(y)

        y = self.S(y).permute([0,2,3,1])


        return x * y.expand_as(x)
        # return y
# writer = SummaryWriter('./TensorBoardX/')
class Morlet_fast_mine_all_channel_3_parameters(nn.Module):
    def __init__(self,wavelet_filters = 4,wavelet_kernel = 125,fs = 250):
        super(Morlet_fast_mine_all_channel_3_parameters, self).__init__()
        if (torch.cuda.is_available()):
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # self.device = 'cpu'

        self.wavelet_filters = wavelet_filters
        self.wavelet_kernel = wavelet_kernel
        self.fs = fs
       # self.wavelet_padding = nn.ZeroPad2d((int(self.wavelet_kernel/2)-1,int(self.wavelet_kernel/2),0,0))
        self.freq = nn.Parameter(torch.tensor([[4+(i+1)*36/(self.wavelet_filters+1)]for i in range(self.wavelet_filters)],
                                 requires_grad = True,device=self.device))
        self.fwhm = nn.Parameter(torch.tensor([[.1] for _ in range(self.wavelet_filters)], requires_grad=True,
                                 device=self.device))
        self.coefficient = nn.Parameter(torch.tensor([[4 * log(2)] for _ in range(self.wavelet_filters)],
                                        requires_grad=True,device=self.device))
    def _design_wavelet(self, kernLength, freq, fwhm, coefficient, fs = 250):
        timevec = torch.arange(kernLength) / fs
        timevec = timevec - torch.mean(timevec)
        timevec = timevec.repeat(self.wavelet_filters).reshape(self.wavelet_filters, kernLength).to(self.device)
        csw = torch.cos(2*pi*freq*timevec)
        gus = torch.exp(-(coefficient * torch.pow(timevec, 2)/torch.pow(fwhm,2)))
        return (csw * gus).unsqueeze(1).unsqueeze(1)
    def forward(self,x):
        """
        x:   input - shape: [N, 1, NEc, Tp]
        N:   Batch size
        NEc: Number of EEG channels
        Tp:  Time point
        """
        data = {}
        ### Wavelet:
        # x = self.wavelet_padding(x)
        self.wavelet_weight = self._design_wavelet(self.wavelet_kernel,self.freq,self.fwhm,self.coefficient,self.fs)
        x = F.conv2d(x,weight = self.wavelet_weight,bias = None)
        return x

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            # nn.Conv2d(1, 40, (1, 25), (1, 1)),
            Morlet_fast_mine_all_channel_3_parameters(wavelet_filters=40,wavelet_kernel=25,fs=250),
            nn.BatchNorm2d(40),
            Fre_SELayer(40, 976, outputkernel=4),
            # Fre_SELayer(40, 937, outputkernel=4),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class FSTAM_net(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
