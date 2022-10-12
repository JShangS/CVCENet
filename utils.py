import os
import os.path
import torch
import numpy as np
import scipy.io as io
import argparse
import torch.nn.modules as nn
import scipy
import math

CUT = 16  # cell under test,
N = 8  # freedom of system
K = 2 * N  #number of secondary data
guad_cell = 0

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

parser.add_argument('--dataDir', default='./data', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='datasave directory')
parser.add_argument('--load',
                    default='./model_name',
                    help='save result·')

parser.add_argument('--model_name', default='CVCENet', help='model to select')
parser.add_argument('--finetuning',
                    default=False,
                    help='finetuning the training')
parser.add_argument('--need_patch', default=False, help='get patch form image')

parser.add_argument('--nDenselayer1',
                    type=int,
                    default=3,
                    help='nDenselayer of CVRDB in regular')
parser.add_argument('--nDenselayer2',
                    type=int,
                    default=3,
                    help='nDenselayer of CVRDB in primary')
parser.add_argument('--nDenselayer3',
                    type=int,
                    default=3,
                    help='nDenselayer of CVRDB in secondary')
parser.add_argument('--nDenselayer',
                    type=int,
                    default=3,
                    help='nDenselayer of CVRDB in estiamtion roade')
parser.add_argument('--growthRate',
                    type=int,
                    default=16,
                    help='growthRate of dense net')
parser.add_argument('--nBlock',
                    type=int,
                    default=1,
                    help='number of CVRDB block')
parser.add_argument('--nFeat',
                    type=int,
                    default=8,
                    help='number of feature maps')
parser.add_argument('--inChannel1',
                    type=int,
                    default=1,
                    help='rgular complex data channel')
parser.add_argument('--inChannel2',
                    type=int,
                    default=1,
                    help='primary complex data channel')                    
parser.add_argument('--inChannel3',
                    type=int,
                    default=16,
                    help='secondary complex data channel')
                    
parser.add_argument('--patchSize', type=int, default=8, help='patch size')

parser.add_argument('--nThreads',
                    type=int,
                    default=3,
                    help='number of threads for data loading')
parser.add_argument('--batchSize',
                    type=int,
                    default=16,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=50, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='MSE', help='output SR video')

parser.add_argument('--scale',
                    type=int,
                    default=1,
                    help='scale output size /input size')

parser.add_argument('--dof', type=int, default=8, help='System degree of freedom')
parser.add_argument('--sc', type=int, default=16, help='number of scondary data')

args, unknown = parser.parse_known_args()


class saveData():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model,epoch,interval=10):
        torch.save(model.state_dict(),
                   self.save_dir_model + '/model_lastest.pt')
        if epoch%interval==0:
            torch.save(model.state_dict(), self.save_dir_model + '/model_obj'+str(epoch)+'.pt')


    def save_log(self, log):
        self.logFile.write(log + '\n')

    def load_model(self, model):
        print(self.args.load + '/model_lastest.pt')
        model.load_state_dict(torch.load(self.args.load + '/model_lastest.pt'))
        print("load mode_status frmo {}/model_lastest.pt".format(
            self.args.load))
        return model
        
    def load_model_epoch(self, model, epoch):
        load_path=self.args.load + '/model_obj'+str(epoch)+'.pt'
        # print(load_path)
        model.load_state_dict(torch.load(load_path))
        # print("load mode_status from {}".format(load_path))
        return model

def get_dataset(args, data_path, shuffle=True, batch_size=args.batchSize):
    # data_train = DIV2K(args)
    a = np.load(data_path)
    source_data = a["data"]
    source_label = a["label"]
    data = GetLoader(source_data, source_label)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             drop_last=True,
                                             shuffle=shuffle,
                                             num_workers=int(args.nThreads),
                                             pin_memory=False)
    # dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
    # 	drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader








# GetLoader class，Inherit Dataset method，rewrite__getitem__() and __len__()
class GetLoader(torch.utils.data.Dataset):
    # 
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # 
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 

    def __len__(self):
        return len(self.data)


