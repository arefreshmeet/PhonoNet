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
    device = One_hot.device
    around_dos = torch.zeros((nkpt, rank, NumOneHot), device=device)

    for qp in range(nkpt):
        tempdata = One_hot.clone()
        a, b, c = oh_data[qp][0], oh_data[qp][1], oh_data[qp][2]
        energy = oh_data[qp][3]
        tempdata = torch.roll(tempdata, shifts=(15-a).item(), dims=0)
        tempdata = torch.roll(tempdata, shifts=(15-b).item(), dims=1)
        tempdata = torch.roll(tempdata, shifts=(15-c).item(), dims=2)
        if qp % 10000 == 0:
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

def Prepare(path,aroundpath,NumOneHot):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(path)
    dpath = path + 'DataSet/'
    energy = torch.from_numpy(np.loadtxt(dpath + 'EnergyFull')).to(device)
    nband = energy.shape[1]
    energy = torch.flatten(energy.t()).view(len(energy)*nband,1)
    QP = torch.from_numpy(np.loadtxt(dpath + 'QPFull')).to(device)
    QP = QP.repeat(nband,1)
    Tau = torch.from_numpy(np.loadtxt(dpath + 'TauFull')).to(device)
    Tau = torch.flatten(Tau.t()).view(len(Tau)*nband,1)
    vx = torch.from_numpy(np.loadtxt(dpath + 'VXFull')).to(device)
    vx = vx[:,1:]
    vx = torch.flatten(vx.t()).view(len(vx)*nband,1)
    vx = vx.pow(2)
    print(QP.shape, energy.shape, vx.shape, Tau.shape)
    all_data = torch.cat((QP, energy, vx, Tau), 1)
    nkpt = all_data.size(0)
    print('all_data.shape',all_data.shape)
    print('Data:Qpoints, energy, square velocity, Tau')
    print(nkpt)
    na, nb,nc,ne = 30,30,30, NumOneHot 
    One_hot = torch.zeros((na,nb,nc,ne), device=device)
    print(One_hot.shape)
    oh_data = all_data[:,0:4].clone()
    oh_data[:,0:3] = oh_data[:,0:3] / 0.033333
    oh_data[:,3] = Energy_loc(NumOneHot,oh_data[:,3])
    oh_data = oh_data.type(torch.int)
    print('max energy',torch.max(oh_data[:,3]))
    for qp in range(nkpt):
        One_hot[oh_data[qp][0],oh_data[qp][1],oh_data[qp][2],oh_data[qp][3]] += 1

    # Weight:
    Weight = torch.ones(nkpt)
    # print(all_data[3,:].detach())

    # print(torch.where(Weight == 0))
    # print(torch.where[all_data[:,0:3] == torch.tensor([0,0,0],device=device)])
    loc = torch.where((all_data[:,0] == 0) & (all_data[:,1] == 0) & (all_data[:,2] == 0))
    print('loc0',loc[0][3])
    lowOptic = all_data[loc[0][3]][3]
    print('low Optic',lowOptic)
    indice = torch.where(all_data[:,3] <= lowOptic)
    print('number of low energy:',indice[0].__len__())
    Weight[indice[0]] = 5
    # print(Weight.__len__())
    # print(Weight)
    print('zero energy',torch.where(all_data[:,3] == 0))
    indice = torch.where(all_data[:,3] == 0)
    Weight[indice[0]] = 0

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
    return oh_data, One_hot, dos, qdos, around_dos,labels,Weight

# Assuming Prepare function is already converted to return PyTorch tensors
NumOneHot = 300
oh_data1, One_hot1, dos1, qdos1, around_dos1, labels1,Weight1 = Prepare('MgO/', 'around_dos_tensor.pth',    NumOneHot)
oh_data2, One_hot2, dos2, qdos2, around_dos2, labels2,Weight2 = Prepare('InSb/', 'around_dos_tensor.pth',    NumOneHot)

# The ThreadPoolExecutor is not necessary in PyTorch as operations are automatically parallelized on GPU
features1 = torch.cat((dos1, qdos1, around_dos1), dim=1)
features2 = torch.cat((dos2, qdos2, around_dos2), dim=1)
all_features = torch.cat((features1, features2), dim=0)
all_features = all_features.float()

print('features:', all_features.shape)

# Convert the labels to PyTorch tensors and perform the operations on the GPU
mass_labels1 = labels1 /  40.3 * 200
mass_labels2 = labels2 / 236.6 * 200
print(mass_labels1.mean().item())
print(mass_labels2.mean().item())

all_labels = torch.cat((mass_labels1, mass_labels2), dim=0) 
all_labels = all_labels.float()

all_weights = torch.cat((Weight1,Weight2), dim=0) 


print('labels:', all_labels.shape)
print('all labels mean',all_labels.mean().item())

del features1,features2,oh_data1,One_hot1,dos1,qdos1,around_dos1 # 删除不再需要的Tensor
del oh_data2,One_hot2,dos2,qdos2,around_dos2
del mass_labels1,mass_labels2,Weight1,Weight2
torch.cuda.empty_cache()  # 释放未使用的显存

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.norm = nn.BatchNorm1d(input_size)  # 归一化层
        # self.layers = nn.Sequential(
        #     nn.Linear(input_size, 1500),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),  # Dropout层
        #     nn.Linear(1500, 900),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),  # Dropout层
        #     nn.Linear(900, 450),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),  # Dropout层
        #     nn.Linear(450, 150),
        #     nn.Softmax(),
        #     nn.Linear(150, 1)
        # )
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # Dropout层
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # Dropout层
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),  # Dropout层
            nn.Linear(512, 256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.layers(x)


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建数据集
dataset = MyDataset(all_features, all_labels)

# 创建权重列表
# 假设我们想要让索引为10的样本被抽样的概率是其他样本的10倍
weights = all_weights

batch_size, learning_rate,weight_decay, num_epochs = 256, 0.0001, 0.001,5000

# 创建样本抽样器
sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

model = Net(NumOneHot*17)
model.to(device)
# criterion = nn.MSELoss()
criterion = nn.HuberLoss()
# criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2正则化

# 在训练循环中使用数据加载器
best_val_loss = float('inf')
early_stopping_counter = 0
train_ls, epoch_ls= [],[]

for epoch in range(num_epochs):  # 训练100个epoch
    for batch_features, batch_labels in dataloader:
        model.train()
        optimizer.zero_grad()
        # 注意：需要调整输入的维度
        # batch_features = batch_features.view(-1, 1, n_features)  # (seq_len, batch_size, n_features)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item()}")

 # 验证阶段
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_features, batch_labels in dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
        val_loss /= len(dataloader)

    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
    train_ls.append(val_loss)
    epoch_ls.append(epoch+1)

    # 早期停止
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= 500:  # 如果验证损失在10个epoch内没有改善，就停止训练
            print('Early stopping')
            break
print(epoch_ls)
print(train_ls)
torch.save(model.state_dict(), 'model_tensor_weight_Huber.pth')


# # train_features = torch.tensor(all_features, dtype=torch.float32,device=device)
# # train_labels = torch.tensor(all_labels, dtype=torch.float32,device=device)
# # train_featues = all_features.clone().detach().requires_grad_(True)
# # train_labels = train_label.clone().detach().requires_grad_(True)
# print('features:',all_features.shape)
# print('labels:',all_labels.shape)

# # dataset = MyDataset(train_features, train_labels)
# dataset = MyDataset(all_features, all_labels)
# train_size = int(0.8 * len(dataset))  # 80%的数据用于训练
# val_size = len(dataset) - train_size

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# net = Net(NumOneHot*17)
# net.to(device)
# criterion = nn.MSELoss()
# # criterion = nn.HuberLoss()
# # criterion = nn.SmoothL1Loss()
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2正则化

# # 加载预训练
# # net.load_state_dict(torch.load('model_mix3.pth'))

# # 训练模型
# best_val_loss = float('inf')
# early_stopping_counter = 0
# train_ls, epoch_ls= [],[]
# for epoch in range(num_epochs):
#     net.train()
#     for i, (netfeatures, netlabels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = net(netfeatures)
#         loss = criterion(outputs, netlabels)
#         loss.backward()
#         optimizer.step()

#     # 验证阶段
#     net.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for netfeatures, netlabels in val_loader:
#             outputs = net(netfeatures)
#             loss = criterion(outputs, netlabels)
#             val_loss += loss.item()
#         val_loss /= len(val_loader)

#     print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
#     train_ls.append(val_loss)
#     epoch_ls.append(epoch+1)

#     # 早期停止
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         early_stopping_counter = 0
#     else:
#         early_stopping_counter += 1
#         if early_stopping_counter >= 500:  # 如果验证损失在10个epoch内没有改善，就停止训练
#             print('Early stopping')
#             break
# print(epoch_ls)
# print(train_ls)


# torch.save(net.state_dict(), 'model_mix5.pth')