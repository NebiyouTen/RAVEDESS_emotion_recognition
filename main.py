import librosa.display
import numpy as np
import os
import torch 
import torch.utils.data
import os
import argparse
import sys

from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import sys
sys.path.append("..")

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Emotion Classifier')
parser.add_argument('--file', type=str, default=None, help=' Audio file name')
args = parser.parse_args()



class Conv1D(nn.Module):
    def __init__(self):
        super(Conv1D, self).__init__()
        self.c1 = nn.Conv1d(187,128,3,padding=1)
        self.c3 = nn.Conv1d(128,32,3,padding=1)
        self.batchNorm = nn.BatchNorm1d(32)
        self.linear = nn.Linear(32,3)
    def forward(self, inputs):
        inputs = torch.transpose(inputs,1,2)
        out = F.leaky_relu(self.c1(inputs))
        out =  F.leaky_relu(self.c3(out))
        out = F.dropout((F.leaky_relu(out)),0.3)
        out = self.batchNorm(out)
        out = torch.mean(out,2)
        out = self.linear(out)
        return out
    
def get_speechData(path):
    y, sr = librosa.load(path,duration=3,offset=1.0)
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    tonnetz = librosa.effects.harmonic(y=y )
    tonnetz = librosa.feature.tonnetz(y=tonnetz, sr=sr)
    rms = librosa.feature.rmse(y=y)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    abc = feature_zero_mean(data_zero_mean(np.concatenate((mfccs,
                          S,
                          tonnetz,
                          rms,
                          chroma_cqt
                         ),axis=0)))
    abc = np.transpose(abc,(1,0))
    return torch.from_numpy(abc).float().unsqueeze(0)
def sample_zero_mean(x):
    
    mean=((x.sum(axis=2)) / x.shape[2])
    
    zero_mean = x - mean.reshape(mean.shape[0],mean.shape[1],1)
    return zero_mean
    pass

def feature_zero_mean(x):
    
    mean=((x.sum(axis=1)) / x.shape[1])
    
    zero_mean = x - mean.reshape(mean.shape[0],1)
    return zero_mean
    pass

def data_zero_mean(x):
    
    mean=((x.sum(axis=0)) / x.shape[0])
    
    zero_mean = x - mean.reshape(1,mean.shape[0])
    return zero_mean
    pass

if __name__ == '__main__':
    emotion= ["neutral","happy","angry"]
    net = Conv1D()
    net.load_state_dict(torch.load('models/models_First_2510.pt'))
    outputs = net(Variable(get_speechData(args.file)))
    _, predicted = torch.max(outputs.data, 1)
    
    print ("predicted State ",emotion[np.asscalar(predicted.numpy())])
    
    
    