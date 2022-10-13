# real value net work real part and imag part at two channels
# import cv2
import torch
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
        
class make_dense(nn.Module):
    def __init__(self, inChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(inChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        out = F.relu(self.conv(x))
        # out = (self.cvconv(x))
        out = torch.cat((x, out), 1)## input conbine with output, therefore has the code "inChannels_ += growthRate"
        return out

# CV Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, inChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        inChannels_ = inChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(inChannels_, growthRate))
            inChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        #############################################
        self.conv_1x1 = nn.Conv2d(inChannels_, inChannels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        # out = self.cvconv_1x1(x)
        return out
# CV Residual Dense Network
class RDN(nn.Module):
    def __init__(self, args, nDenselayer, inChannel, outChannel):
        super(RDN, self).__init__()
        # inChannel = args.inChannel_road1
        # nDenselayer = nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args
        # F-1
        self.conv1 = nn.Conv2d(inChannel, nFeat, kernel_size=1, padding=0, bias=True)
        # # F0
        # self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3 
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, outChannel, kernel_size=3, padding=1, bias=True)
    def forward(self, x):

        F_  = F.relu(self.conv1(x))
        # F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)     
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        # FGF = self.GFF_3x3(FdLF)
        output = FdLF + F_
        # output = self.conv3(FDF)
        # us = self.cvconv_up(FDF)
        # us = self.upsample(us)

        # output = self.cvconv3(us)

        return output

# Real Value CENet  
class CENet(nn.Module):
    def __init__(self, args):
        super(CENet, self).__init__()
        self.RDN1 = RDN(args, args.nDenselayer1, args.inChannel1, args.dof*2) #regular data feature extract layers
        self.RDN2 = RDN(args, args.nDenselayer2, args.inChannel2, args.dof*2) #primary data feature extract layers
        self.RDN3 = RDN(args, args.nDenselayer3, args.inChannel3, args.dof*2) #secondary data feature extract layers
        self.RDN_EC = RDN(args, args.nDenselayer, args.dof*4, args.dof*2) #Covariance estiamtion layers
        self.GFF1_3x3 = nn.Conv2d(args.dof*2, 2, kernel_size=3, padding=1, bias=True)
        self.GFF2_1x1 = nn.Conv2d(2,1, kernel_size=1, padding=0, bias=True)
        self.droupout2d = nn.Dropout2d()
        self.nb1 = nn.BatchNorm2d(args.inChannel1)
        self.nb2 = nn.BatchNorm2d(args.inChannel2)
        self.nb3 = nn.BatchNorm2d(args.inChannel3)
    def forward(self, regular, primary, secondary):
        Road1 = self.RDN1(self.nb1(regular))
        # Road1 = self.droupout2d(Road1)
        # Road2_1 = self.NormalBatch1(secondary)
        Road2 = self.RDN2(self.nb2(primary))
        # Road2 = self.droupout2d(Road2)
        Road3 = self.RDN3(self.nb3(secondary))
        # Road3 = self.droupout2d(Road3)
        FF = torch.cat((Road1, Road2, Road3), 1)
        F1 = self.RDN_EC(FF)
        F2 = self.GFF1_3x3(F1)
        output = self.GFF2_1x1(F2)

        return output      