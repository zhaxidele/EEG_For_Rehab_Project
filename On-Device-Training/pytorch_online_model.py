import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Identity




##set model specifications
fs= 160                  #sampling frequency
#channel= 64              #number of electrode
num_input= 1             #number of channel picture (for EEG signal is always : 1)
num_class= 4             #number of classes
#signal_length = 480      #number of sample in each tarial


channel= 8              #number of electrode
signal_length = 160      #number of sample in each tarial


F1= 8                    #number of temporal filters
D= 2                     #depth multiplier (number of spatial filters)
F2= D*F1                 #number of pointwise filters

kernel_size_1= (1,round(fs/2))
#kernel_size_1= (1,128)
kernel_size_2= (channel, 1)
kernel_size_3= (1, round(fs/8))
#kernel_size_3= (1, 16)
kernel_size_4= (1, 1)

kernel_avgpool_1= (1,4)
kernel_avgpool_2= (1,4)
#kernel_avgpool_2= (1,4)
dropout_rate= 0.25

#ks0= int(round((kernel_size_1[0]-1)/2))
#ks1= int(round((kernel_size_1[1]-1)/2))
#kernel_padding_1= (ks0, ks1-1)
#ks0= int(round((kernel_size_3[0]-1)/2))
#ks1= int(round((kernel_size_3[1]-1)/2))
#kernel_padding_3= (ks0, ks1)

class EEGNet_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1
        self.conv2d = nn.Conv2d(num_input, F1, kernel_size_1, stride = 1, padding= "same")
        self.Batch_normalization_1 = nn.BatchNorm2d(F1, momentum=0.99, affine=True, eps=1e-3)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D * F1, momentum=0.99, affine=True, eps=1e-3)
        #self.Elu = nn.ELU()
        self.Elu = nn.ReLU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout_rate)
        # layer 3
        #self.Separable_conv2D_depth = nn.Conv2d(D * F1, D * F1, kernel_size_3, padding=kernel_padding_3, groups=D * F1)
        self.Separable_conv2D_depth = nn.Conv2d(D * F1, D * F1, kernel_size_3, stride = 1, padding="same", groups=D * F1)
        self.Separable_conv2D_point = nn.Conv2d(D * F1, F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(F2, momentum=0.99, affine=True, eps=1e-3)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)
        # layer 4
        self.Flatten = nn.Flatten()
        #self.Dense = nn.Linear(F2 * round(signal_length / 32), num_class)
        #self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x))  # .relu()
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        #y = self.Dense(y)
        #y = self.Softmax(y)

        return y


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.Dense = nn.Linear(F2 * int(signal_length / 16), num_class)
        #self.Dense = nn.Linear(round(signal_length / 4), num_class)
        #self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Dense(x)
        #out = self.Softmax(x)
        return out


class EEGnet(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        out = self.softmax(out)
        return out