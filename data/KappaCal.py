# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:44:22 2018

@author: 李正
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

print('Begin')
path = os.getcwd()
if not(os.path.exists('conductivity')):
    os.mkdir('conductivity')
# os.mkdir("test")
volume = os.popen('vaspkit -task 601 | grep Volume').read()
volume = volume.split(':')[1]
volume = float(volume)
nkpt = os.popen('wc -l BTE.qpoints_full').read()
nkpt = nkpt.split('BTE.')[0]
nkpt = int(nkpt)
nband = os.popen('wc -l BTE.v_full').read()
nband = int(nband.split('BTE.')[0]) / nkpt 
# print(path, volume,nkpt,nband)





class conductivity(object):
    def __init__(self):
        self.path = path
        self.Tpath = path + '/T300K'
        self.Cpath = path + '/conductivity'
        self.nbands = int(nband) # number of phonon bands
        self.Volumn = volume # volume of cell
        self.te = 300 # K, Temperature, default is 300K
        self.npkt = int(nkpt) # number of k points

        # some constants
        self.hbar = 1.05457172647e-22 # J*ps
        self.kB =1.38064852e-23 # J/K
        self.cont = 1e21 * self.hbar ** 2 / self.kB / self.te / self.te / self.Volumn /  self.npkt
        print('Target path:', path,
            '\n Number of phonon bands:',self.nbands,
              '\n Volumn of Cell:', self.Volumn, 
              '\n Number of K-points:', self.npkt,
              '\n Total constant:', self.cont)

    def DrawKpoints(self):
        QPFull = np.loadtxt(self.path+'/BTE.qpoints_full')
        OmegaFull = np.zeros((self.npkt,self.nbands))

        fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
        ax = Axes3D(fig) # 将画布作用于 Axes3D 对象上。

        ax.scatter(QPFull[:,2],QPFull[:,3],QPFull[:,4],alpha=0.1) # 画出(xs1,ys1,zs1)的散点图。
        # plt.colorbar()
        # ax.scatter(xs2,ys2,zs2,c='r',marker='^')
        # ax.scatter(xs3,ys3,zs3,c='g',marker='*')

        ax.set_xlabel('X label') # 画出坐标轴
        ax.set_ylabel('Y label')
        ax.set_zlabel('Z label')

        plt.show()
        pass
    
    def mondify_data(self):
        # 读入omega（换单位），v，计算出分布f，弛豫时间tau，存储在文件夹中
        temp = np.loadtxt(self.path + '/BTE.omega')
        nbands = np.shape(temp)[1]
        irkp = np.shape(temp)[0]
        print('能带数量，k点数',nbands,irkp)
        # temp = temp / 2 / np.pi
        np.savetxt(self.path + '/conductivity/omega',temp)
        fbe= np.zeros(np.shape(temp))
        for i in range(0,np.shape(temp)[0]):
            for j in range(0,np.shape(temp)[1]):
                if temp[i][j] == 0:
                    fbe[i][j] = 0
                else:
                    fbe[i][j] = 1 / (np.exp(self.hbar * temp[i][j] / self.kB / self.te)-1)
        np.savetxt(self.path + '/conductivity/f_BE',fbe)
        del temp,fbe

        # 读入群速度，三个方向写入三个文件
        temp = np.zeros((irkp,nbands))
        vel = np.loadtxt(self.path + '/BTE.v')
        print(irkp,np.shape(vel))
        for i in range(0,nbands,1):
            for j in range(0,irkp):
                temp[j][i] = vel[j + irkp * i][0]
        np.savetxt(self.path + '/conductivity/v_alpha', temp)
        del temp
        temp = np.zeros((irkp,nbands))
        vel = np.loadtxt(self.path + '/BTE.v')
        print(irkp,np.shape(vel))
        for i in range(0,nbands,1):
            for j in range(0,irkp):
                temp[j][i] = vel[j + irkp * i][1]
        np.savetxt(self.path + '/conductivity/v_beta', temp)
        del temp
    
        temp = np.zeros((irkp,nbands))
        vel = np.loadtxt(self.path + '/BTE.v')
        print(irkp,np.shape(vel))
        for i in range(0,nbands,1):
            for j in range(0,irkp):
                temp[j][i] = vel[j + irkp * i][2]
        np.savetxt(self.path + '/conductivity/v_gamma', temp)
        del temp
        temp = np.zeros((irkp,nbands))
        qpoints = np.loadtxt(self.path + '/BTE.qpoints')
        print(irkp,np.shape(qpoints),np.sum(qpoints))
        for i in range(0,nbands,1):
            for j in range(0,irkp):
                temp[j][i] = qpoints[j][2]
        np.savetxt(self.path + '/conductivity/degeneracy', temp)
        del temp
        temp = np.zeros((irkp,nbands))
        anh_iso = np.loadtxt(self.path + '/BTE.w_isotopic')
        print(irkp,np.shape(qpoints))
        for i in range(0,nbands,1):
            for j in range(0,irkp):
                temp[j][i] = anh_iso[j][1]
        np.savetxt(self.path + '/conductivity/scat_iso', temp)
        del temp


        # 读入碰撞率，计算弛豫时间
        anh_minus = np.zeros((irkp, nbands))
        minus = np.loadtxt(self.Tpath + '/BTE.w_anharmonic_minus')
        print(irkp, np.shape(minus))
        for i in range(0, nbands, 1):
            for j in range(0, irkp):
                anh_minus[j][i] = minus[j + irkp * i][1]
        np.savetxt(self.path + '/conductivity/anharmonic_minus', anh_minus)
        del minus

        anh_plus = np.zeros((irkp, nbands))
        plus = np.loadtxt(self.Tpath + '/BTE.w_anharmonic_plus')
        print(irkp, np.shape(plus))
        for i in range(0, nbands, 1):
            for j in range(0, irkp):
                anh_plus[j][i] = plus[j + irkp * i][1]
        np.savetxt(self.path + '/conductivity/anharmonic_plus', anh_plus)
        del plus

        anh_zero = np.zeros((irkp, nbands))
        zero = np.loadtxt(self.Tpath + '/BTE.w')
        print(irkp, np.shape(vel))
        for i in range(0, nbands, 1):
            for j in range(0, irkp):
                anh_zero[j][i] = zero[j + irkp * i][1]
        np.savetxt(self.path + '/conductivity/anharmonic_zero', anh_zero)


        anharmonic = np.zeros((irkp, nbands))
        BteAnh = np.loadtxt(self.Tpath + '/BTE.w_anharmonic')
        print(irkp, np.shape(vel))
        for i in range(0, nbands, 1):
            for j in range(0, irkp):
                anharmonic[j][i] = BteAnh[j + irkp * i][1]
        np.savetxt(self.path + '/conductivity/anharmonic', anharmonic)


        AnhIso = np.zeros((irkp, nbands))
        BteIso = np.loadtxt(self.path + '/BTE.w_isotopic')
        print(irkp, np.shape(vel))
        for i in range(0, nbands, 1):
            for j in range(0, irkp):
                AnhIso[j][i] = BteIso[j + irkp * i][1]
        np.savetxt(self.path + '/conductivity/anharmonic', AnhIso)
        del zero

        tau_0= np.zeros((irkp, nbands))
        print(irkp, np.shape(tau_0))
        for i in range(0, np.shape(tau_0)[0], 1):
            for j in range(0, np.shape(tau_0)[1]):
                # anh = anh_plus[i][j] + 0.5 * anh_minus[i][j] + anh_zero[i][j]
                anh = anh_zero[i][j]
                # anh = anharmonic[i][j] + AnhIso[i][j]
                if anh == 0:
                    tau_0[i][j] = 0
                else:
                    tau_0[i][j] = 1 / anh
        np.savetxt(self.path + '/conductivity/tau0', tau_0)
        del anh_zero,anh_minus,anh_plus,tau_0

    def ModifyVFull(self,x=0,y=0,z=0):
        #整理BTE.v_full成qpoints_full*nbands，第一列是在qpoints的行数，
        VFull = np.loadtxt(self.path + '/BTE.v_full')
        QPFull = np.loadtxt(self.path + '/BTE.qpoints_full')
        print('max V:',np.max(VFull[:,0]),np.max(VFull[:,1]),np.max(VFull[:,2]))
        nptk = np.shape(QPFull)[0]
        nbands = self.nbands
        if x:
            VXFull = np.zeros((nptk,nbands+1))
            for i in range(0,nptk):
                VXFull[i][0] = QPFull[i][1]
            for i in range(0,nbands):
                for j in range(0,nptk):
                    VXFull[j][i+1] = VFull[nptk * i + j][0]
            np.savetxt(self.path + '/conductivity/VXFull', VXFull)
        if y:
            VYFull = np.zeros((nptk,nbands+1))
            for i in range(0,nptk):
                VYFull[i][0] = QPFull[i][1]
            for i in range(0,nbands):
                for j in range(0,nptk):
                    VYFull[j][i+1] = VFull[nptk * i + j][1]
            np.savetxt(self.path + '/conductivity/VYFull', VYFull)
        if z:
            VZFull = np.zeros((nptk,nbands+1))
            for i in range(0,nptk):
                VZFull[i][0] = QPFull[i][1]
            for i in range(0,nbands):
                for j in range(0,nptk):
                    VZFull[j][i+1] = VFull[nptk * i + j][2]
            np.savetxt(self.path + '/conductivity/VZFull', VZFull)

    #RTA近似下计算各个k点热导率
    def PointsCalculator(self,x=0,y=0,z=0,NV=0):
        omega = np.loadtxt(self.path + '/conductivity/omega')
        tau0 = np.loadtxt(self.path + '/conductivity/tau0')
        VXFull = np.loadtxt(self.path + '/conductivity/VXFull')
        fbe = np.loadtxt(self.path + '/conductivity/f_BE')
        print(np.shape(omega),np.shape(tau0),np.shape(VXFull))

        nband = np.shape(omega)[1]
        nptk = np.shape(VXFull)[0]
        if x:
            KappaFull_X = np.zeros((nptk,nband+1))
            for i in range(0,nptk):
                KappaFull_X[i][0] = VXFull[i][0]
            for i in range(0,nptk):
                for j in range(0,nband):
                    temp = int(VXFull[i][0] - 1) #python列表规则
                    # kappa[i][j+1] =
                    KappaFull_X[i][j+1] = np.square(omega[temp][j]) * np.square(VXFull[i][j+1]) * tau0[temp][j] * fbe[temp][j] * (1 + fbe[temp][j])
            np.savetxt(self.path + '/conductivity/KappaFUll_X',KappaFull_X)
            print('计算后热导率形状',np.shape(KappaFull_X))
            print(KappaFull_X[1000:1100,0])
            kappax = self.cont * np.sum(KappaFull_X)
            print('kappa x:',kappax)
        if y:
            VYFull = np.loadtxt(self.path + '/conductivity/VYFull')
            KappaFull_Y = np.zeros((nptk, nband + 1))
            for i in range(0, nptk):
                KappaFull_Y[i][0] = VYFull[i][0]
            for i in range(0, nptk):
                for j in range(0, nband):
                    temp = int(VYFull[i][0] - 1)  # python列表规则
                    # kappa[i][j+1] =
                    KappaFull_Y[i][j + 1] = omega[temp][j] ** 2 * VYFull[i][j + 1] ** 2 * tau0[temp][j] * fbe[temp][j] * (
                    1 + fbe[temp][j])
            np.savetxt(self.path + '/conductivity/KappaFUll_Y', KappaFull_Y)
            kappay = self.cont * np.sum(KappaFull_Y)
            print('kappa y:', kappay)
        if z:
            VZFull = np.loadtxt(self.path + '/conductivity/VZFull')
            KappaFull_Z = np.zeros((nptk, nband + 1))
            for i in range(0, nptk):
                KappaFull_Z[i][0] = VZFull[i][0]
            for i in range(0, nptk):
                for j in range(0, nband):
                    temp = int(VZFull[i][0] - 1)  # python列表规则
                    # kappa[i][j+1] =
                    KappaFull_Z[i][j + 1] = omega[temp][j] ** 2 * VZFull[i][j + 1] ** 2 * tau0[temp][j] * fbe[temp][j] * (
                        1 + fbe[temp][j])
            np.savetxt(self.path + '/conductivity/KappaFUll_Z', KappaFull_Z)
            kappaz = self.cont * np.sum(KappaFull_Z)
            print('kappa z:', kappaz)
        if NV:
            irkp = np.shape(omega)[0]
            NV = np.zeros((irkp, nband))
            print('max omega:',np.max(omega))
            print('max tau0:',np.max(tau0))
            print('max fbe:',np.max(fbe))
            for i in range(0, irkp):
                for j in range(0, nband):
                    NV[i][j] = omega[i][j] ** 2 * tau0[i][j] * fbe[i][j] * (
                        1 + fbe[i][j])
            np.savetxt(self.path + '/conductivity/NV', NV)

    def symmetry_tensor(self,matrix,Crotations):
        NSymRot = np.shape(Crotations)[0]
        tmp = 0
        for i in range(0,NSymRot):
            tmp += np.dot(Crotations[i],np.dot(matrix,np.transpose(Crotations[i])))
        tensor = tmp / NSymRot
        return(tensor)

    def CalTConduct(self):
        # 打印两个方向的总热导率，并输出两个文件，cumulativekappa和differentialkappa
        # KappaX = np.loadtxt(self.path + '/conductivity/KappaFUll_X')
        # KappaY = np.loadtxt(self.path + '/conductivity/KappaFUll_Y')
        # KappaZ = np.loadtxt(self.path + '/conductivity/KappaFUll_Z')
        print('Gathering data......')
        Vx = np.loadtxt(self.path + '/conductivity/VXFull')
        Vy = np.loadtxt(self.path + '/conductivity/VYFull')
        Vz = np.loadtxt(self.path + '/conductivity/VZFull')
        print('max V:',np.max(Vx[:,1:]),np.max(Vy[:,1:]),np.max(Vz[:,1:]))
        # CRotation = np.load(self.path + '/conductivity/Crotations.npy')
        NV = np.loadtxt(self.path + '/conductivity/NV')
        print('max NV:',np.max(NV))
        nkpt = np.shape(Vx)[0]
        nbands = np.shape(NV)[1]
        # print(np.shape(Vx),np.shape(CRotation),np.shape(NV))
        kappa = np.zeros((nkpt,nbands,3,3))
        print(np.shape(kappa))
        n = 1
        for j in range(0,nbands):
            for i in range(0,nkpt):
                line = int(Vx[i][0])
                V = np.array([[Vx[i][j+1],Vy[i][j+1],Vz[i][j+1]]])
                kappa[i][j] = np.dot(np.transpose(V),V)
                kappa[i][j] = kappa[i][j] * float(NV[line-1][j])
            # print(n,'\n',V,'\n',kappa[i][j],'\n')
#            print(n)
            n = n+1
        kappa = kappa * self.cont
        np.save(self.path + '/conductivity/no_symmetry_kappa',kappa)
        print(kappa[0][0])
        ktensor = kappa.sum(axis=(0,1))
        print(ktensor)
        # 按照ShengBTE程序的方法，调整对称性，Crotations已经给出
#        Crotations = np.load('Srotations.npy')
        # 从程序中copy出的Crotations
        Crotations = np.loadtxt(self.path + '/conductivity/CrotationsFrom_readconfig', skiprows=1)
#        print(np.shape(Crotations))
#        num_operation = int(len(Crotations) / 9)
#        Crotations = Crotations.reshape((32, 3, 3))
        Crotations = Crotations.reshape((12, 3, 3))

        n=1
        for j in range(0,nbands):
            for i in range(0,nkpt):
                kappa[i][j] = self.symmetry_tensor(kappa[i][j],Crotations)
#            print(n)
            n += 1
        np.save(self.path + '/conductivity/symmetrized_kappa', kappa)
        ktensor = kappa.sum(axis=(0, 1))
        print(ktensor)
                    
        
        
        
    
if __name__ == "__main__":
    # pass
    condcut = conductivity()
    condcut.DrawKpoints()
#    condcut.ModifyVFull(x=1,y=1,z=1)
    # condcut.mondify_data()
    # condcut.PointsCalculator(x=1,y=0,z=0)
#    condcut.PointsCalculator(NV=1)
    # condcut.CalTConduct()