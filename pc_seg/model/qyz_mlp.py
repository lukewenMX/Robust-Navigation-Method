import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(in_channels,16,kernel_size=1)#512,16
        self.conv3 = nn.Conv1d(16, 32, kernel_size=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, out_channels,kernel_size=1)#512
        self.bn1 = nn.BatchNorm1d(16)#512
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bnm1 = nn.BatchNorm1d(256)
        self.bnm2 = nn.BatchNorm1d(128)
        self.bnm3 = nn.BatchNorm1d(512)
        #self.fc1 = nn.Linear(2000* out_channels, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(2000*out_channels, 512)#512
        self.fc4 = nn .Linear(512,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,1)#256

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))#bn1
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        #out = torch.max(out, 2, keepdim=True)[0]
        #out = out.reshape(-1, 512)
        out = out.reshape(out.size(0), -1)
        #out =self.fc1(out)

        out = F.relu(self.bnm3((self.fc1(out))))#bnm3
        out = F.relu(self.bnm1(self.dropout(self.fc4(out))))
        out = F.relu(self.bnm2(self.dropout(self.fc2(out))))
        out = self.fc3(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,layer_dim,output_dim):
        super(LSTM, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0),self.hidden_dim)).cuda()
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0),self.hidden_dim)).cuda()
        out, (hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


