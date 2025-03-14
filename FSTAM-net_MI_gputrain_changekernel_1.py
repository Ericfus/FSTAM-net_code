"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

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
import h5py
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
cudnn.benchmark = False
cudnn.deterministic = True

# 定义钩子函数
# -------------------- 第一步：定义接收feature的函数 ---------------------- #
# 这里定义了一个类，类有一个接收feature的函数hook_fun。定义类是为了方便提取多个中间层。


features_in_hook = []
features_out_hook = []

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None
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
        self.ln1 = nn.LayerNorm([1,self.F1,32])
        self.ln2 = nn.LayerNorm([outputkernel,self.F1,32])
        # self.f2 = nn.Linear(channel//reduction,channel,bias = False)
        self.S = nn.Sigmoid()
    def forward(self, x):
        b, c,d, _ = x.size()
        # print(x.shape)
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
    def __init__(self,wavelet_filters = 4,wavelet_kernel = 500,fs = 250):
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
       #  self.freq = nn.Parameter(torch.tensor([[4+(i+1)*36/(self.wavelet_filters+1)]for i in range(self.wavelet_filters)],
       #                           requires_grad = True,device=self.device))
        ## 非均匀分布的wavelet_filters在0.5-2.5hz内布置五分之一的wavelet_filters，而在2.5-35hz内布置五分之四的wavelet_filters，
        num_low_freq = wavelet_filters // 5  # Number of filters in low-frequency range
        num_high_freq = wavelet_filters - num_low_freq  # Number of filters in high-frequency range

        # Frequencies for low range [0.5, 2.5]
        low_freqs = torch.linspace(0.5, 2.5, steps=num_low_freq + 1)[1:]  # Exclude 0.5

        # Frequencies for high range [2.5, 35]
        high_freqs = torch.linspace(2.5, 35, steps=num_high_freq + 1)[1:]  # Exclude 2.5

        # Combine frequencies
        freq_values = torch.cat((low_freqs, high_freqs)).unsqueeze(1)  # Shape: (wavelet_filters, 1)

        # Define as trainable parameter
        self.freq = nn.Parameter(freq_values.to(self.device), requires_grad=True)

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
            Morlet_fast_mine_all_channel_3_parameters(wavelet_filters=40,wavelet_kernel=500,fs=250),
            nn.BatchNorm2d(40),
            Fre_SELayer(40, 1437, outputkernel=4),
            nn.Conv2d(40, 40, (32, 1), (1, 1)),
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
            nn.Linear(3640, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub,XX,YY):
        super(ExP, self).__init__()
        ## 这里原始的batchsize=72
        self.XX = XX
        self.YY = YY
        keys = XX.files
        print('xx.files',keys)
        self.X = []
        self.Y = []
        for i in range(len(keys)):
            target_mean = np.mean(XX[keys[i]])
            target_std = np.std(XX[keys[i]])
            # XX[keys[i]]=(XX[keys[i]]-target_mean)/target_std
            self.X.append((XX[keys[i]]-target_mean)/target_std)
            self.Y.append(YY[keys[i]])

        self.batch_size = 64
        self.n_epochs = 100
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        # self.root = 'D:\\Desktop\\WaveletKernelNet-master\\dataset\\BCICIV_2a_gdf\\'

        # self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")
        self.log_write = open("./results/Wavelet_Se_log_subject_ck%d_1.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 22, 1000))


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(2):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 64, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8,dtype=np.int64)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        # ! please  recheck if you need validation set
        # ! and the data segement compared methods used
        # 导入数据

        # eegdatalist = []
        # eeglabellist = []
        # filepath = r'F:\dataset\GigaScience\sorted\dataset22'
        # fnames = glob.glob(os.path.join(filepath, '*.mat'))
        # num_samples = len(fnames)
        # for i in range(num_samples):
        #     fname = fnames[i]
        #     srate = h5py.File(fname, 'r')['eeg']['srate']
        #     eegEvent = h5py.File(fname, 'r')['eeg']['imagery_event'][()]
        #     eegXEA = h5py.File(fname, 'r')['eeg']['noXEA'][()]
        #     sizeL = int(h5py.File(fname, 'r')['eeg']['sizeL'][()])
        #     sizeR = int(h5py.File(fname, 'r')['eeg']['sizeR'][()])
        #
        #     # eegr = h5py.File(fname, 'r')['eeg']['tempListR'][()]
        #     # eegl = h5py.File(fname, 'r')['eeg']['tempListL'][()]
        #     # eegXEA = np.concatenate((eegr, eegl), 0)
        #
        #     n_imagery_trials = h5py.File(fname, 'r')['eeg']['n_imagery_trials'][()]
        #     n_imagery_trials = int(n_imagery_trials)
        #     indices = np.where(eegEvent == 1)[0]
        #     # 右手标签为0，左手标签为1
        #     # eeglabel = np.concatenate((np.zeros((1, eegr.shape[0])) - 2, np.ones((1, eegl.shape[0]))), 1)
        #     eeglabel = np.concatenate((np.zeros((1, sizeR)), np.ones((1, sizeL))), 1)
        #
        #     eegdatalist.append(eegXEA)
        #     eeglabellist.append(eeglabel)
        #
        testid = self.nSub-1
        # #保存
        # np.savez(r'F:\dataset\GigaScience\sorted\eegdataset.npz',*eegdatalist)
        # np.savez(r'F:\dataset\GigaScience\sorted\eeglabellist.npz',*eeglabellist)
        # 标准化

        # 测试集和训练集
        self.allData = np.expand_dims(np.concatenate(self.X[:testid]+self.X[testid+1:],0),axis=1).transpose([0,1,3,2])
        self.allLabel = np.concatenate(self.Y[:testid]+self.Y[testid+1:],1).squeeze()+1

        self.testData = np.expand_dims(self.X[testid],axis=1).transpose([0,1,3,2])
        self.testLabel = self.Y[testid].squeeze()+1





        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]




        # standardize
        # target_mean = np.mean(self.allData)
        # target_std = np.std(self.allData)
        # self.allData = (self.allData - target_mean) / target_std
        # test_mean = np.mean(self.testData)
        # test_std = np.std(self.testData)
        #
        # self.testData = (self.testData - test_mean) / test_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr
        bestweight=[]
        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                # aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # img = torch.cat((img, aug_data))
                # label = torch.cat((label, aug_label))


                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label)
                # if i == len(self.dataloader)-1:
                #     for (name,module) in self.model.named_modules():
                #         # print(name)
                #         if name == 'module.0.shallownet.2.S':
                #             features_out_hook=[]
                #             module.register_forward_hook(hook=hook)
                # if features_out_hook != []:
                #     weight = (sum(features_out_hook[-1])/len(features_out_hook[-1])).squeeze().detach().cpu().numpy()
                #     weight = weight.T
                #     if i % 10 == 0:
                #         np.set_printoptions(precision=2)#保留两位小数
                #         # print(weight[:,0:-1:3])
                #         for name, param in self.model.named_parameters():
                #             if name == '0.shallownet.0.freq':
                #                 xlabel = param.data.detach().cpu().numpy().T
                #                 weight = np.concatenate((xlabel,weight),axis = 0)
                #                 weight = np.concatenate((np.expand_dims(np.arange(0,65),-1),weight),axis=1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()


            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    torch.save(self.model.module.state_dict(), './model/modelbest_ck_S%d_1.pth'% (self.nSub))
                    # for (name, module) in self.model.named_modules():
                    #     # print(name)
                    #     # if name == '0.shallownet.2.S':
                    #     #     module.register_forward_hook(hook=hook)
                    # if features_out_hook != []:
                    #     weight = (sum(features_out_hook[-1]) / len(
                    #         features_out_hook[-1])).squeeze().detach().cpu().numpy()
                    #     weight = weight.T
                    #
                    #     np.set_printoptions(precision=2)  # 保留两位小数
                    #     # print(weight[:,0:-1:3])
                    #     for name, param in self.model.named_parameters():
                    #         if name == 'module.0.shallownet.0.freq':
                    #             xlabel = param.data.detach().cpu().numpy().T
                    #             weight = np.concatenate((xlabel, weight), axis=0)
                    #             weight = np.concatenate((np.expand_dims(np.arange(0, 65), -1), weight), axis=1)
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
                    # try:
                    #     bestweight = weight
                    #     for name, param in self.model.named_parameters():
                    #         if name == '0.shallownet.0.freq':
                    #             xlabel = param.data.detach().cpu().numpy().T
                    #             bestweight = np.concatenate((xlabel,bestweight),axis = 0)
                    #             bestweight = np.concatenate((np.expand_dims(np.arange(0,65),-1),bestweight),axis=1)
                    # except:
                    #     pass
        # torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        # 保存weight
        # csv_file1 = './output/weight2.csv'
        # with open(csv_file1, 'a', newline='') as file:
        #     # 获取当前时间
        #     current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     writer = csv.writer(file)
        #     writer.writerows([current_time, str(self.nSub)])
        #     writer.writerows(list(np.array(bestweight)))
        #     file.write('\n')
        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0
    result_write = open("./results/Wavelet_se_sub_result_ck.txt_1", "w")
    XX = np.load('./data/rimiXEA_data.npz')
    YY = np.load('./data/rimiXEA_labels.npz')
    for i in range(12):

        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1,XX,YY)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 12
    aver = aver / 12

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
