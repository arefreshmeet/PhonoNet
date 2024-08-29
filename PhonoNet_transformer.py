import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def NeDos(dos, energy):
    loc = int(energy.item())  # Convert tensor to python scalar
    ls_dos = dos.clone()
    ls_dos[loc:] = -ls_dos[loc:]
    return ls_dos

def Energy_loc(tot_num, energy): 
    loc = torch.round(tot_num / 150 * energy).long()  # Converted to long for indexing
    return loc

def Around_dos(aroundpath, oh_data, One_hot, nkpt, NumOneHot):
    print('Start Around_dos')
    print('path=',aroundpath)
    print('oh_data:',oh_data.shape)
    print('One_hot:',One_hot.shape)
    print('nkpt:',nkpt)
    print('NumOneHot',NumOneHot)
    rank = 15
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    around_dos = torch.zeros((nkpt, rank, NumOneHot), device=device)

    for qp in range(nkpt):
        tempdata = One_hot.clone()
        a, b, c = oh_data[qp][0], oh_data[qp][1], oh_data[qp][2]
        energy = oh_data[qp][3]
        tempdata = torch.roll(tempdata, shifts=(15-a).item(), dims=0)
        tempdata = torch.roll(tempdata, shifts=(15-b).item(), dims=1)
        tempdata = torch.roll(tempdata, shifts=(15-c).item(), dims=2)
        if qp % 100 == 0:
            print(qp)
        around_dos[qp][0] = torch.sum(tempdata[15-1:15+1, 15-1:15+1, 15-1:15+1], dim=(0, 1, 2)) / tempdata[15-1:15+1, 15-1:15+1, 15-1:15+1].numel()
        for i in range(1, rank, 1):
            around_dos[qp][i] = torch.sum(tempdata[15-1-i:15+1+i, 15-1-i:15+1+i, 15-1-i:15+1+i], dim=(0, 1, 2)) - torch.sum(tempdata[15-i:15+i, 15-i:15+i, 15-i:15+i], dim=(0, 1, 2))
            size = tempdata[15-1-i:15+1+i, 15-1-i:15+1+i, 15-1-i:15+1+i].numel() - tempdata[15-i:15+i, 15-i:15+i, 15-i:15+i].numel()
            around_dos[qp][i] = around_dos[qp][i] / size
            around_dos[qp][i] = NeDos(around_dos[qp][i], energy)

    print(oh_data[3][3])
    print(aroundpath, 'built')
    print('End around_dos')
    torch.save(around_dos, aroundpath)
    return around_dos

def Prepare(path,aroundpath):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(path)
    dpath = path + 'DataSet/'
    energy = torch.from_numpy(np.loadtxt(dpath + 'EnergyFull')).to(device)
    nband = energy.shape[1]
    energy = energy.flatten().view(len(energy)*nband,1)
    QP = torch.from_numpy(np.loadtxt(dpath + 'QPFull')).to(device)
    QP = QP.repeat(nband,1)
    Tau = torch.from_numpy(np.loadtxt(dpath + 'TauFull')).to(device)
    Tau = Tau.flatten().view(len(Tau)*nband,1)
    vx = torch.from_numpy(np.loadtxt(dpath + 'VXFull')).to(device)
    vx = vx[:,1:]
    vx = vx.flatten().view(len(vx)*nband,1)
    vx = vx.pow(2)
    print(QP.shape, energy.shape, vx.shape, Tau.shape)
    all_data = torch.cat((QP, energy, vx, Tau), 1)
    nkpt = all_data.size(0)
    print(all_data.shape)
    print('Data:Qpoints, energy, square velocity, Tau')

    NumOneHot = 300

    print(nkpt)
    na, nb,nc,ne = 30,30,30, NumOneHot 
    One_hot = torch.zeros((na,nb,nc,ne), device=device)
    print(One_hot.shape)
    oh_data = all_data[:,0:4].clone()
    oh_data[:,0:3] = oh_data[:,0:3] / 0.033333
    oh_data[:,3] = Energy_loc(NumOneHot,oh_data[:,3])
    oh_data = oh_data.type(torch.int)
    print(torch.max(oh_data[:,3]))
    for qp in range(nkpt):
        One_hot[oh_data[qp][0],oh_data[qp][1],oh_data[qp][2],oh_data[qp][3]] += 1

    dos = torch.zeros((nkpt,NumOneHot ), device=device)
    Totdos = torch.sum(One_hot,dim=(0,1,2))
    for qp in range(nkpt):
        dos[qp] = NeDos(Totdos,oh_data[qp][3])
    dos = dos / One_hot[20,:,:].numel() * NumOneHot 

    qdos = torch.zeros((nkpt,NumOneHot ), device=device)
    print(One_hot[20,:,:].numel() )
    for qp in range(nkpt):
        xindex = oh_data[qp][0]
        qpdos = torch.sum(One_hot[xindex,:,:],dim=(0,1))
        qdos[qp] = NeDos(qpdos,oh_data[qp][3]) / One_hot[xindex,:,:].numel() * NumOneHot 

    if os.path.exists(path+aroundpath):
        print('around Dos exists')
        around_dos = torch.load(path + aroundpath,map_location=device)
        around_dos = around_dos.view(nkpt,15 * NumOneHot )
    else:
        print(path+aroundpath, 'do not exists,start Around_dos')
        around_dos = Around_dos(path+aroundpath,oh_data,One_hot,nkpt,NumOneHot)
        around_dos = around_dos.view(nkpt,15 * NumOneHot ) 
    labels = all_data[:,all_data.shape[1]-1:]

    print(path, 'oh_data, One_hot, dos, qdos, around_dos,labels')
    return oh_data, One_hot, dos, qdos, around_dos,labels

# Assuming Prepare function is already converted to return PyTorch tensors
oh_data1, One_hot1, dos1, qdos1, around_dos1, labels1 = Prepare('MgO/', 'around_dos_tensor.pth')
oh_data2, One_hot2, dos2, qdos2, around_dos2, labels2 = Prepare('InSb/', 'around_dos_tensor.pth')

# The ThreadPoolExecutor is not necessary in PyTorch as operations are automatically parallelized on GPU
features1 = torch.cat((dos1, qdos1, around_dos1), dim=1)
features2 = torch.cat((dos2, qdos2, around_dos2), dim=1)
all_features = torch.cat((features1, features2), dim=0)
all_features = all_features.float()

del features1,features2,oh_data1,One_hot1,dos1,qdos1,around_dos1 # 删除不再需要的Tensor
del oh_data2,One_hot2,dos2,qdos2,around_dos2
torch.cuda.empty_cache()  # 释放未使用的显存

print('features:', all_features.shape)

# Convert the labels to PyTorch tensors and perform the operations on the GPU
mass_labels1 = labels1 /  40.3 * 200
mass_labels2 = labels2 / 236.6 * 200
print('MgO label mean',mass_labels1.mean().item())
print('InSb label mean',mass_labels2.mean().item())

all_labels = torch.cat((mass_labels1, mass_labels2), dim=0) 
all_labels = all_labels.float()
del mass_labels1,mass_labels2
torch.cuda.empty_cache()  # 释放未使用的显存

print('labels:', all_labels.shape)
print(all_labels.mean().item())

# import torch
# from torch import nn
# from torch.optim import Adam

# torch.cuda.is_available()
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device)



# class TransformerRegressor(nn.Module):
#     def __init__(self, input_dim, num_heads, num_layers):
#         super(TransformerRegressor, self).__init__()
#         self.transformer = nn.Transformer(input_dim, num_heads, num_layers)
#         self.linear = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         x = self.transformer(x)
#         x = self.linear(x)
#         return x

# # 假设我们有一组特征all_features和一组标签all_labels
# # all_features = torch.rand((100, 5100))  # 100个样本，每个样本5100个特征
# # all_labels = torch.rand(100)  # 100个样本，每个样本对应一个实数标签

# # 创建模型
# model = TransformerRegressor(5100, 10, 6)
# model.to(device)

# # 创建优化器
# optimizer = Adam(model.parameters())

# # 使用MSE作为损失函数
# loss_fn = nn.MSELoss()

# for epoch in range(100):  # 训练100个epoch
#     # 前向传播
#     outputs = model(all_features)
#     # 计算损失
#     loss = loss_fn(outputs, all_labels)
#     # 反向传播
#     loss.backward()
#     # 更新参数
#     optimizer.step()
#     # 清空梯度
#     optimizer.zero_grad()

#     if epoch % 10 == 0:  # 每10个epoch打印一次损失
#         print(f"Epoch: {epoch}, Loss: {loss.item()}")


# torch.save(model.state_dict(), 'model_mix_trans.pth')

import torch
import torch.nn as nn
import torch.optim as optim
import math
# 假设我们有一些样本
# n_samples = 100
n_features = 5100
# all_features = torch.randn(n_samples, n_features)
# all_labels = torch.randn(n_samples, 1)

# 调整输入的维度
all_features = all_features.view(-1, 1, n_features)  # (seq_len, batch_size, n_features)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, 1)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

model = TransformerModel(n_features, nhead=2, nhid=200, nlayers=2)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(100):  # 训练100个epoch
    model.train()
    optimizer.zero_grad()
    outputs = model(all_features)
    loss = loss_fn(outputs.squeeze(), all_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item()}")