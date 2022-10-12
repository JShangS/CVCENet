# import cv2
import torch
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexDropout2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
        
class make_cvdense(nn.Module):
    def __init__(self, inChannels, growthRate, kernel_size=3):
        super(make_cvdense, self).__init__()
        self.cvconv = ComplexConv2d(inChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        out = complex_relu(self.cvconv(x))
        # out = (self.cvconv(x))
        out = torch.cat((x, out), 1)## input conbine with output, therefore has the code "inChannels_ += growthRate"
        return out

# CV Residual dense block (RDB) architecture
class CVRDB(nn.Module):
    def __init__(self, inChannels, nDenselayer, growthRate):
        super(CVRDB, self).__init__()
        inChannels_ = inChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_cvdense(inChannels_, growthRate))
            inChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        #############################################
        self.cvconv_1x1 = ComplexConv2d(inChannels_, inChannels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.cvconv_1x1(out)
        out = out + x
        # out = self.cvconv_1x1(x)
        return out

# CV Residual Dense Network
class CVRDN(nn.Module):
    def __init__(self, args, nDenselayer, inChannel, outChannel):
        super(CVRDN, self).__init__()
        # inChannel = args.inChannel_road1
        # nDenselayer = nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args
        # F-1
        self.cvconv1 = ComplexConv2d(inChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.cvconv2 = ComplexConv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3 
        self.CVRDB1 = CVRDB(nFeat, nDenselayer, growthRate)
        self.CVRDB2 = CVRDB(nFeat, nDenselayer, growthRate)
        self.CVRDB3 = CVRDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = ComplexConv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = ComplexConv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.cvconv_up = ComplexConv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.cvconv3 = ComplexConv2d(nFeat, outChannel, kernel_size=3, padding=1, bias=True)
    def forward(self, x):

        F_  = self.cvconv1(x) # nFeat
        F_0 = self.cvconv2(F_) # nFeat
        F_1 = self.CVRDB1(F_0) # nFeat
        F_2 = self.CVRDB2(F_1) # nFeat
        F_3 = self.CVRDB3(F_2) # nFeat
        FF = torch.cat((F_1, F_2, F_3), 1) # 3*nFeat
        FdLF = self.GFF_1x1(FF)# nFeat         
        FGF = self.GFF_3x3(FdLF) # nFeat
        FDF = FGF + F_ # nFeat
        output = self.cvconv3(FDF) # outChannel
        # us = self.cvconv_up(FDF)
        # us = self.upsample(us)

        # output = self.cvconv3(us)

        return output

class CENet(nn.Module):
    def __init__(self, args):
        super(CENet, self).__init__()
        self.CVRDN1 = CVRDN(args, args.nDenselayer1, args.inChannel1, args.dof) #regular data feature extract layers
        self.CVRDN2 = CVRDN(args, args.nDenselayer2, args.inChannel2, args.dof) #primary data feature extract layers
        self.CVRDN3 = CVRDN(args, args.nDenselayer3, args.inChannel3, args.dof*2) #secondary data feature extract layers
        self.CVRDN_EC = CVRDN(args, args.nDenselayer, args.dof*4, args.dof*2) #Covariance estiamtion layers
        self.GFF1_1x1 = ComplexConv2d(args.dof*2, 2, kernel_size=1, padding=0, bias=True)
        self.GFF2_1x1 = ComplexConv2d(2,1, kernel_size=1, padding=0, bias=True)
        self.cdropout2d = ComplexDropout2d()
        self.nb1 = ComplexBatchNorm2d(args.inChannel1)
        self.nb2 = ComplexBatchNorm2d(args.inChannel2)
        self.nb3 = ComplexBatchNorm2d(args.inChannel3)
    def forward(self, regular, primary, secondary):
        Road1 = self.CVRDN1(self.nb1(regular))
        # Road1 = self.cdropout2d(Road1)
        Road2 = self.CVRDN2(self.nb2(primary))
        # Road2 = self.cdropout2d(Road2)
        Road3 = self.CVRDN3(self.nb3(secondary))
        # Road3 = self.cdropout2d(Road3)
        FF = torch.cat((Road1, Road2, Road3), 1)
        F1 = self.CVRDN_EC(FF)
        F2 = self.GFF1_1x1(F1)
        output = self.GFF1_1x1(F2)
        
        return output