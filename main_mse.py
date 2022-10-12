# 《CV-GMTINet: GMTI Using a Deep Complex-Valued Convolutional Neural Network for Multichannel SAR-GMTI System》
##
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
# import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from model import cvmodel
from model import realmodel
# from data import DIV2K
from utils import *
import time
from JSPY_function.covarianceEstimation import perturbMatrix


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)


def set_loss(args):
    lossType = args.lossType
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction


def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args, data_path):
    #  select network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    if args.model_name == 'CVCENet':
        args.inChannel1 = 1
        args.inChannel2 = 1
        args.inChannel3 = 16
        args.load = "MSELoss_cv"
        print(args.saveDir)
        my_model = cvmodel.CENet(args).to(device)
    elif args.model_name == 'CENet':
        args.inChannel1 = 2
        args.inChannel2 = 2
        args.inChannel3 = 32
        args.load = "MSELoss_rv"
        print(args.saveDir)
        print(args.inChannel1)
        print(args.inChannel2)
        print(args.inChannel3)
        my_model = realmodel.CENet(args).to(device)
    save = saveData(args)
    # fine-tuning or retrain
    if args.finetuning:
        my_model = save.load_model(my_model)
    # load data
    # dataloader = get_dataset(args, data_path)
    L1_lossfunction = set_loss(args)
    # L1_lossfunction = my_whiteloss()
    total_loss = 0
    loss_epoch = []
    loss_eval_epoch = []
    total_time_start = time.time()
    EYE = torch.complex(torch.eye(N, N), torch.zeros(N, N)).to(device) 
    regular_data_EYE = torch.zeros((args.batchSize, 1, N, N),
                                   dtype=torch.complex64).to(device) 
    nn = np.arange(0, N, 1)
    for i in range(args.batchSize):
        regular_data_EYE[i, 0, :, :] =  EYE
    for epoch in range(args.epochs):
        print(epoch + 1, "/", args.epochs)
        print("training......")
        my_model.train()
        dataloader = get_dataset(args, data_path)
        optimizer = optim.Adam(my_model.parameters())
        learning_rate = set_lr(args, epoch, optimizer)
        total_loss_ = 0
        L1_loss_ = 0
        tic = time.time()
        for batch, (data, label) in enumerate(dataloader):
            if batch>int(1e4/16):
                break
            data = Variable(data.to(device), volatile=False)
            label = Variable(label.to(device))
            my_model.zero_grad()
            primary_data = data[:, N:N + 1, :, :].to(device)
            secondary_data = torch.cat(
                [data[:, 0:N, :, :], data[:, N + 1:2 * N + 1, :, :]],
                dim=1).to(device)
            if np.random.uniform() < 0.5:
                regular_data = regular_data_EYE
            else:
                sigmaa = np.random.uniform(0.01, 0.9,(1, args.batchSize))
                perbMatrix = perturbMatrix(N, sigmaa, args.batchSize)
                perbMatrix = torch.tensor(perbMatrix).to(device)
                regular_data = label * perbMatrix
                regular_data = regular_data.reshape(args.batchSize, 1, N,
                                                    N).to(device)
            if np.random.uniform() < 0.5:  ## add target
                for i in range(args.batchSize):
                    SNR_db = np.random.uniform(-10, 30)
                    fd = np.random.uniform(-0.5, 0.5)
                    vt = np.exp(1j * 2 * math.pi * nn * fd).reshape(N,1)
                    SNR_num = 10**(SNR_db/10)
                    temp = np.abs(np.mat(vt.T.conjugate()) * np.mat(np.linalg.inv(label[i,:,:].to('cpu').detach().numpy())) * np.mat(vt))
                    alpha = np.sqrt(SNR_num / temp)
                    primary_data[i,:,:] += torch.tensor(vt*alpha).to(device)
            if args.model_name == 'CVCENet':
                output = my_model(regular_data, primary_data, secondary_data)
            elif args.model_name == 'CENet':
                primary_data = torch.cat(
                    (torch.real(primary_data), torch.imag(primary_data)), 1)
                regular_data = torch.cat(
                    (torch.real(regular_data), torch.imag(regular_data)), 1)
                secondary_data = torch.cat(
                    (torch.real(secondary_data), torch.imag(secondary_data)),
                    1)
                output = my_model(regular_data, primary_data, secondary_data)
                output = torch.complex(output[:, 0, :, :], output[:, 1, :, :])
            output = output.reshape(args.batchSize, N, N)
            L1_loss = L1_lossfunction(torch.abs(output - label),
                                      torch.zeros(output.shape).to(device))
            L1_loss.backward()
            optimizer.step()
            L1_loss_ += L1_loss.data.cpu().numpy()
        toc = time.time()
        L1_loss_ = L1_loss_ / (batch + 1)
        log = "[{} / {}] \tLearning_rate: {}\t loss: {:.8f}".\
            format(epoch+1,args.epochs, learning_rate, L1_loss_)
        print(log)
        print("time cost: ", (toc - tic), "s")
        save.save_log(log)
        save.save_model(my_model, epoch, interval=2)
        loss_epoch.append(L1_loss_)
        if args.model_name == 'CVCENet':
            loss_epoch_ = np.array(loss_epoch)
            np.save('./result/MSELoss_cv/loss_epoch_cv.npy',
                    loss_epoch_)
        if args.model_name == 'CENet':  #and epoch%10==0:
            loss_epoch_ = np.array(loss_epoch)
            np.save('./result/MSELoss_rv/loss_epoch_rv.npy', loss_epoch_)
        ###########evale############################################################################
        my_model.eval()
        print("evaling.....")
        dataloader = get_dataset(args,'./data/sim_data_eval.npz')
        L1_loss_eval_ = 0
        for batch, (data, label) in enumerate(dataloader):
            if batch>int(1e4/16):
                break
            data = Variable(data.to(device), volatile=False)
            label = Variable(label.to(device))
            my_model.zero_grad()
            primary_data = data[:, N:N + 1, :, :].to(device)
            secondary_data = torch.cat(
                [data[:, 0:N, :, :], data[:, N + 1:2 * N + 1, :, :]],
                dim=1).to(device)
            if np.random.uniform() < 0.5:
                regular_data = regular_data_EYE
            else:
                sigmaa = np.random.uniform(0.01, 0.9,(1, args.batchSize))
                perbMatrix = perturbMatrix(N, sigmaa, args.batchSize)
                perbMatrix = torch.tensor(perbMatrix).to(device)
                regular_data = label * perbMatrix
                regular_data = regular_data.reshape(args.batchSize, 1, N,
                                                    N).to(device)
            if np.random.uniform() < 0.5:  ## add target
                for i in range(args.batchSize):
                    SNR_db = np.random.uniform(-10, 30)
                    fd = np.random.uniform(-0.5, 0.5)
                    vt = (np.exp(1j * 2 * math.pi * nn * fd)/np.sqrt(N)).reshape(N,1)
                    SNR_num = 10**(SNR_db/10)
                    temp = np.abs(np.mat(vt.T.conjugate()) * np.mat(np.linalg.inv(label[i,:,:].to('cpu').detach().numpy())) * np.mat(vt))
                    alpha = np.sqrt(SNR_num / temp)
                    primary_data[i,:,:] += torch.tensor(vt*alpha).to(device)
            if args.model_name == 'CVCENet':
                output = my_model(regular_data, primary_data, secondary_data)
            elif args.model_name == 'CENet':
                primary_data = torch.cat(
                    (torch.real(primary_data), torch.imag(primary_data)), 1)
                regular_data = torch.cat(
                    (torch.real(regular_data), torch.imag(regular_data)), 1)
                secondary_data = torch.cat(
                    (torch.real(secondary_data), torch.imag(secondary_data)),
                    1)
                output = my_model(regular_data, primary_data, secondary_data)
                output = torch.complex(output[:, 0, :, :], output[:, 1, :, :])
            output = output.reshape(args.batchSize, N, N)
            L1_loss_eval = L1_lossfunction(torch.abs(output - label),
                                      torch.zeros(output.shape).to(device))
            L1_loss_eval_ += L1_loss_eval.data.cpu().numpy()
        L1_loss_eval_ = L1_loss_eval_ / (batch + 1)
        log = "eval: [{} / {}] \t loss: {:.8f}".format(epoch+1,args.epochs, L1_loss_eval_)
        print(log)
        loss_eval_epoch.append(L1_loss_eval_)
        if args.model_name == 'CVCENet':
            loss_eval_epoch_ = np.array(loss_eval_epoch)
            np.save('./result/MSELoss_cv/loss_eval_epoch_cv.npy',
                    loss_eval_epoch_)
            np.save('', loss_eval_epoch_)
        if args.model_name == 'CENet':  #and epoch%10==0:
            loss_eval_epoch_ = np.array(loss_eval_epoch)
            np.save('./result/MSELoss_rv/loss_eval_epoch_rv.npy', loss_eval_epoch_)
        


    total_time_end = time.time()
    print("total time cost: ", (total_time_end - total_time_start), "s")


if __name__ == '__main__':
    ### train #####
    data_path = "./data/sim_data.npz"
    train(args, data_path)
