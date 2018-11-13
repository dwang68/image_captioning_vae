import torch
import os

CUDA_VISIBLE_DEVICES = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')