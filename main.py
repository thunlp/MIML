# -*- coding: utf-8 -*-
import time
from model import MIML
from train import train_model
import torch
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='./data/train.json',
        help='train file')
parser.add_argument('--val', default='./data/val.json',
        help='val file')
parser.add_argument('--test', default='./data/val.json',
        help='test file')
parser.add_argument('--meta_info', default='./data/meta-info.json',
        help='pre-prepared files stored meta-information')
parser.add_argument('--MI', action='store_true',
        help='use meta-information')
parser.add_argument('--MF', action='store_true',
        help='use attention(Dot) or Average')
parser.add_argument('--MAML', action='store_true',
        help='use MAML')
parser.add_argument('--noise_rate', default=0, type=int,
        help='noise rate, value range 0 to 10')
parser.add_argument('--VAT', action='store_true',
        help='use virtual adversarial training')
parser.add_argument('--B', default=8, type=int,
        help='batch number')
parser.add_argument('--N', default=5, type=int,
        help='N way')
parser.add_argument('--K', default=1, type=int,
        help='K shot')
parser.add_argument('--Q', default=1, type=int,
        help='number of query per class')
parser.add_argument('--Train_iter', default=10000, type=int,
        help='number of iters in training')
parser.add_argument('--Val_iter', default=200, type=int,
        help='number of iters in validing')
parser.add_argument('--Test_update_step', default=150, type=int,
        help='number of adaptation steps')
parser.add_argument('--max_length', default=50, type=int,
       help='max length')
opt = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
#setup_seed(998244353)
#setup_seed(1000000007)
#setup_seed(7)
#setup_seed(1)
setup_seed(57)

if opt.MAML:
    model_name = 'maml'
else:
    model_name = 'miml'
encoder_name='bert'
MI = opt.MI
MF = opt.MF
VAT = opt.VAT #use vat loss or not
if opt.MAML:
    MI, MF, VAT = False, False, False
noise_rate = opt.noise_rate #noise rate: [0,10]
B = opt.B
N = opt.N
K = opt.K
Q = opt.Q
Train_iter = opt.Train_iter
Val_iter = opt.Val_iter
Test_update_step = opt.Test_update_step

print('----------------------------------------------------')
print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))
print("Encoder: {}".format(encoder_name))
if not opt.MAML: print('Use meta-information to get better result.')
else: print('Without meta-information which might bring better result.')
if MI: print('Use parameter initialization.')
else: print('Without parameter initialization.')
if MF: print('Use attention.')
else: print('Without attention.')
if VAT: print('Use virtual adversarial training.')
else: print('Without virtual adversarial training.')
print('----------------------------------------------------')

max_length = opt.max_length
data_dir = {} # dataset
data_dir['noise_rate'] = noise_rate
data_dir['train'] = opt.train
data_dir['val'] = opt.val
data_dir['test'] = opt.test
data_dir['meta-info'] = opt.meta_info

start_time=time.time()

miml=MIML(B,N,K,max_length,data_dir,opt.MAML,MI,MF,VAT)
train_model(miml,B,N,K,Q,data_dir,
    train_iter=Train_iter,val_iter=Val_iter,test_update_step=Test_update_step)

time_use=time.time()-start_time
h=int(time_use/3600)
time_use-=h*3600
m=int(time_use/60)
time_use-=m*60
s=int(time_use)
print('Totally used',h,'hours',m,'minutes',s,'seconds')


