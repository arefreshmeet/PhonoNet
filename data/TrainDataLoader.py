import torch
import numpy as np
import os

class TrainDataLoader:
    def __init__(self, num_one_hot, base_path):
        self.num_one_hot = num_one_hot
        self.base_path = base_path
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    def make_dataset(self,subpath):
        nkpt_command = f"wc -l {os.path.join(subpath, 'BTE.qpoints_full')}"
        nband_command = f"wc -l {os.path.join(subpath, 'BTE.v_full')}"
        irkp_command = f"wc -l {os.path.join(subpath, 'BTE.omega')}"
        nkpt = int(os.popen(nkpt_command).read().split()[0])
        nband = int(os.popen(nband_command).read().split()[0]) / nkpt
        nbands = int(nband)
        irkp = int(os.popen(irkp_command).read().split()[0])
        print(nkpt,nband,irkp)
        #整理BTE.v_full成qpoints_full*nbands，第一列是在qpoints的行数，
        os.mkdir(subpath + '/DataSet')
        VFull = np.loadtxt(subpath + '/BTE.v_full')
        QPFull = np.loadtxt(subpath + '/BTE.qpoints_full')
        # print('max V:',np.max(VFull[:,0]),np.max(VFull[:,1]),np.max(VFull[:,2]))
        nptk = np.shape(QPFull)[0]
        VXFull = np.zeros((nptk,nbands+1))
        for i in range(0,nptk):
            VXFull[i][0] = QPFull[i][1]
        for i in range(0,nbands):
            for j in range(0,nptk):
                VXFull[j][i+1] = VFull[nptk * i + j][0]
        np.savetxt(subpath + '/DataSet/VXFull', VXFull)
        VYFull = np.zeros((nptk,nbands+1))
        for i in range(0,nptk):
            VYFull[i][0] = QPFull[i][1]
        for i in range(0,nbands):
            for j in range(0,nptk):
                VYFull[j][i+1] = VFull[nptk * i + j][1]
        np.savetxt(subpath + '/DataSet/VYFull', VYFull)

        VZFull = np.zeros((nptk,nbands+1))
        for i in range(0,nptk):
            VZFull[i][0] = QPFull[i][1]
        for i in range(0,nbands):
            for j in range(0,nptk):
                VZFull[j][i+1] = VFull[nptk * i + j][2]
        np.savetxt(subpath + '/DataSet/VZFull', VZFull)
        # 将BTE.omega 扩充成 OmegaFull
        qp = np.loadtxt(subpath + '/BTE.qpoints')
        qpFull = np.loadtxt(subpath + '/BTE.qpoints_full')
        Energy = np.loadtxt(subpath + '/BTE.omega')    
        # print(np.shape(qp),np.shape(qpFull),np.shape(Energy))
        # print(qp)  
        EneFull = np.zeros((nkpt,nbands))
        for i in range(0,nkpt):
            line = int(qpFull[i][1] - 1)
            EneFull[i][:] = Energy[line]
            # if i == 50:
                # print('第50行', line, EneFull[i])
        np.savetxt(subpath+'/DataSet/EnergyFull',EneFull)

        zero = np.loadtxt(subpath + '/T300K/BTE.w')
        anh_zero = np.zeros((irkp, nbands))
        # print(irkp, np.shape(anh_zero))
        for i in range(0, nbands, 1):
            for j in range(0, irkp):
                anh_zero[j][i] = zero[j + irkp * i][1]
        tau_0= np.zeros((irkp, nbands))
        # print(irkp, np.shape(anh_zero),anh_zero[0])
        for i in range(0, np.shape(tau_0)[0], 1):
            for j in range(0, np.shape(tau_0)[1]):
                # anh = anh_plus[i][j] + 0.5 * anh_minus[i][j] + anh_zero[i][j]
                anh = anh_zero[i][j]
                # anh = anharmonic[i][j] + AnhIso[i][j]
                if anh <= 0 or np.isnan(anh):
                   tau_0[i][j] = 0
                else:
                    tau_0[i][j] = 1 / anh
        np.savetxt(subpath + '/DataSet/tau0', tau_0)
        Tau = tau_0
        print(np.shape(Tau))
        TauFull = np.zeros((nkpt,nbands))
        for i in range(0,nkpt):
            line = int(qpFull[i][1] - 1)
            TauFull[i][:] = Tau[line] 
        np.savetxt(subpath+'/DataSet/TauFull',TauFull) 
        np.savetxt(subpath+'/DataSet/QPFull',qpFull[:,2:5])
        
        # 分布计算
        fbe= np.zeros(np.shape(EneFull))
        hbar = 1.05457172647e-22 # J*ps
        kB =1.38064852e-23 # J/K
        te = 300 # temperature
        for i in range(0,np.shape(EneFull)[0]):
            for j in range(0,np.shape(EneFull)[1]):
                if EneFull[i][j] == 0:
                    fbe[i][j] = 0
                else:
                    fbe[i][j] = 1 / (np.exp(hbar * EneFull[i][j] / kB / te)-1)
        np.savetxt(subpath + '/DataSet/f_Full',fbe)

    def load_data(self, path):

        dpath = os.path.join(path, 'DataSet/')
        energy = torch.from_numpy(np.loadtxt(os.path.join(dpath, 'EnergyFull'))).to(self.device)
        nband = energy.shape[1]
        energy = torch.flatten(energy.t()).view(len(energy) * nband, 1)

        QP = torch.from_numpy(np.loadtxt(os.path.join(dpath, 'QPFull'))).to(self.device)
        QP = QP.repeat(nband, 1)

        Tau = torch.from_numpy(np.loadtxt(os.path.join(dpath, 'TauFull'))).to(self.device)
        Tau = torch.flatten(Tau.t()).view(len(Tau) * nband, 1)

        vx = torch.from_numpy(np.loadtxt(os.path.join(dpath, 'VXFull'))).to(self.device)
        vx = vx[:, 1:]
        vx = torch.flatten(vx.t()).view(len(vx) * nband, 1)
        vx = vx.pow(2)

        all_data = torch.cat((QP, energy, vx, Tau), 1)
        return all_data

    def prepare_dataset(self, subpath, aroundpath):
        all_data = self.load_data(subpath)
        nkpt = all_data.size(0)

        na, nb, nc, ne = 30, 30, 30, self.num_one_hot
        One_hot = torch.zeros((na, nb, nc, ne), device=self.device)

        oh_data = all_data[:, 0:4].clone()
        oh_data[:, 0:3] = oh_data[:, 0:3] / 0.033333
        oh_data[:, 3] = self.energy_loc(self.num_one_hot, oh_data[:, 3])
        oh_data = oh_data.type(torch.int)

        for qp in range(nkpt):
            One_hot[oh_data[qp][0], oh_data[qp][1], oh_data[qp][2], oh_data[qp][3]] += 1

        Weight = torch.ones(nkpt)

        loc = torch.where((all_data[:, 0] == 0) & (all_data[:, 1] == 0) & (all_data[:, 2] == 0))
        lowOptic = all_data[loc[0][3]][3]
        indice = torch.where(all_data[:, 3] <= lowOptic)
        Weight[indice[0]] = 5

        indice = torch.where(all_data[:, 3] == 0)
        Weight[indice[0]] = 0

        dos = torch.zeros((nkpt, self.num_one_hot), device=self.device)
        Totdos = torch.sum(One_hot, dim=(0, 1, 2))
        for qp in range(nkpt):
            dos[qp] = self.ne_dos(Totdos, oh_data[qp][3])
        dos = dos / One_hot[20, :, :, :].numel() * self.num_one_hot

        qdos = torch.zeros((nkpt, self.num_one_hot), device=self.device)
        for qp in range(nkpt):
            xindex = oh_data[qp][0]
            qpdos = torch.sum(One_hot[xindex, :, :], dim=(0, 1))
            qdos[qp] = self.ne_dos(qpdos, oh_data[qp][3]) / One_hot[xindex, :, :].numel() * self.num_one_hot

        if os.path.exists(subpath + aroundpath):
            print(subpath,'arounddos exist')
            around_dos = torch.load(subpath + aroundpath, map_location=self.device)
            around_dos = around_dos.view(nkpt, 15 * self.num_one_hot)
        else:
            around_dos = self.around_dos(oh_data, One_hot, nkpt,subpath)
            around_dos = around_dos.view(nkpt, 15 * self.num_one_hot)

        labels = all_data[:, all_data.shape[1] - 1:]
        return oh_data, One_hot, dos, qdos, around_dos, labels, Weight

    def energy_loc(self, tot_num, energy):
        loc = torch.round(tot_num / 150 * energy).long()
        return loc

    def ne_dos(self, dos, energy):
        loc = int(energy.item())
        ls_dos = dos.clone()
        ls_dos[loc:] = -ls_dos[loc:]
        return ls_dos

    def around_dos(self, oh_data, One_hot, nkpt,subpath):
        rank = 15
        around_dos = torch.zeros((nkpt, rank, self.num_one_hot), device=self.device)

        for qp in range(nkpt):
            tempdata = One_hot.clone()
            a, b, c = oh_data[qp][0], oh_data[qp][1], oh_data[qp][2]
            energy = oh_data[qp][3]
            tempdata = torch.roll(tempdata, shifts=(15-a).item(), dims=0)
            tempdata = torch.roll(tempdata, shifts=(15-b).item(), dims=1)
            tempdata = torch.roll(tempdata, shifts=(15-c).item(), dims=2)
            if qp % 10000 == 0:
                print(qp)
            around_dos[qp][0] = torch.sum(tempdata[14:16, 14:16, 14:16], dim=(0, 1, 2)) / tempdata[14:16, 14:16, 14:16].numel()
            for i in range(1, rank):
                around_dos[qp][i] = (torch.sum(tempdata[14-i:16+i, 14-i:16+i, 14-i:16+i], dim=(0, 1, 2)) -
                                     torch.sum(tempdata[15-i:15+i, 15-i:15+i, 15-i:15+i], dim=(0, 1, 2)))
                size = tempdata[14-i:16+i, 14-i:16+i, 14-i:16+i].numel() - tempdata[15-i:15+i, 15-i:15+i, 15-i:15+i].numel()
                around_dos[qp][i] = around_dos[qp][i] / size
                around_dos[qp][i] = self.ne_dos(around_dos[qp][i], energy)

        torch.save(around_dos, subpath + self.aroundpath)
        return around_dos

    def process(self, aroundpath):
        print('start process')
        all_features = []
        all_labels = []
        all_weights = []

        for subfolder in os.listdir(self.base_path):
            subpath = os.path.join(self.base_path, subfolder)
            print(subpath)
            print(os.path.exists(subpath + '/DataSet'))
            if not os.path.isdir(subpath):
                continue
            if not(os.path.exists(subpath + '/DataSet')):
                self.make_dataset(subpath)
            oh_data, One_hot, dos, qdos, around_dos, labels, Weight = self.prepare_dataset(subpath, aroundpath)

            features = torch.cat((dos, qdos, around_dos), dim=1)
            all_features.append(features)
            all_labels.append(labels)
            all_weights.append(Weight)

        all_features = torch.cat(all_features, dim=0).float()
        all_labels = torch.cat(all_labels, dim=0).float()
        all_weights = torch.cat(all_weights, dim=0)
        torch.save(all_features, self.base_path + "features")
        torch.save(all_labels, self.base_path + "labels")
        torch.save(all_weights, self.base_path + "weights")

        print('features:', all_features.shape)
        print('labels:', all_labels.shape)
        print('all labels mean', all_labels.mean().item())

        torch.cuda.empty_cache()

