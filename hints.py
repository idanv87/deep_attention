import os
import sys
import math
import pyamg
from matplotlib.ticker import ScalarFormatter

import time

from scipy.stats import qmc
import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys


from utils import upsample
from constants import Constants
from utils import  grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D

from two_d_model import  deeponet
from test_deeponet import domain
from main import generate_f_g
from df_polygon import generate_example, generate_rect, generate_rect2
 
model=deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.05.07.11.22.05best_model.pth')
model.load_state_dict(best_model['model_state_dict'])

model_mu=deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.05.09.07.51.45best_model.pth')
model_mu.load_state_dict(best_model['model_state_dict'])
def NN2( F, X, Y, dom,mask):
    int_points=np.vstack([X,Y]).T
    with torch.no_grad():
       
        y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
        f=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        tensor_to_repeat= torch.tensor(dom, dtype=torch.float32)
        dom= tensor_to_repeat.unsqueeze(0).repeat(y1.shape[0], 1, 1)
        tensor_to_repeat= torch.tensor(mask, dtype=torch.float32)
        mask= tensor_to_repeat.unsqueeze(0).repeat(y1.shape[0], 1, 1)
        pred2=model_mu([y1, f,dom, mask])


    return torch.real(pred2).numpy()+1J*torch.imag(pred2).numpy()
   

def NN( F, X, Y, dom,mask):
    
    int_points=np.vstack([X,Y]).T
 

    

  

    with torch.no_grad():
       
        y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
        f=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        tensor_to_repeat= torch.tensor(dom, dtype=torch.float32)
        dom= tensor_to_repeat.unsqueeze(0).repeat(y1.shape[0], 1, 1)
        tensor_to_repeat= torch.tensor(mask, dtype=torch.float32)
        mask= tensor_to_repeat.unsqueeze(0).repeat(y1.shape[0], 1, 1)
        pred2=model([y1, f,dom, mask])
        # pred1=model.model1.branch2(submodel(dom,dom,dom, mask).squeeze(-1))[0]
        # plt.plot(pred1)
        # plt.show()
        # sys.exit()

    return torch.real(pred2).numpy()+1J*torch.imag(pred2).numpy()
   

A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
# A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()

A,f_ref,f,dom,mask, X,Y, valid_indices, d_super, d=generate_rect2(28)

x0=(f+1J*f)*0.001
# A = pyamg.gallery.poisson((30,30), format='csr')/((1/31)**2)
# A=A+1*scipy.sparse.identity(A.shape[0])
# x, exitCode = scipy.sparse.linalg.gmres(A, f,x0*100, tol=1e-13, maxiter=10)
# print(np.linalg.norm(A@x-f)/np.linalg.norm(f))


sol=scipy.sparse.linalg.spsolve(A, f)
# NN(f_ref,X,Y,dom, mask)


# l,v=scipy.sparse.linalg.eigs(A+A.conjugate().T, k=2,which='SR')
# x, exitCode = scipy.sparse.linalg.gmres(A, f,x0, tol=1e-13, maxiter=100)
# print(np.linalg.norm(A@x-f)/np.linalg.norm(f))


b=f
err=[]
color=[]
def hints(A,b,x0):
    for k in range(3000):
        if (k+1)%120==0:
            f_real=(-A@x0+b).real
            f_imag=(-A@x0+b).imag
            
            func_real=interpolation_2D(d_super.X,d_super.Y,f_real)
            func_imag=interpolation_2D(d_super.X,d_super.Y,f_imag)
            f_real=np.array(func_real(d.X,d.Y))
            f_imag=np.array(func_imag(d.X,d.Y))
            
            mu_real=np.mean(f_real)
            s_real=np.std(f_real)
            mu_imag=np.mean(f_imag)
            s_imag=np.std(f_imag)
            f_ref_real=np.zeros(Constants.n**2)
            f_ref_imag=np.zeros(Constants.n**2)
            
            
            f_ref_real[valid_indices]=(f_real-mu_real)/s_real
            f_ref_imag[valid_indices]=(f_imag-mu_imag)/s_imag
            corr_real=(NN(f_ref_real,X,Y, dom, mask)+mu_real*NN2(f_ref_real*0+1,X,Y,dom,mask)/s_real)*s_real
            corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+mu_imag*NN2(f_ref_real*0+1,X,Y,dom,mask)/s_imag)*s_imag
            # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
            # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
            corr=corr_real+1J*corr_imag
            x0=x0+corr
            color.append('red')
            
        else:
            x0=Gauss_zeidel(A.todense(),b,x0,theta=1)[0]
            # x0,exitcode=scipy.sparse.linalg.gmres(A, b, x0,tol=1e-2, maxiter=1)
            color.append('black')
        if k %20 ==0:    
            print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
            print(k)
        err.append(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
        if err[-1]<1e-14:
            break
    return err, color    
err, color=hints(A,b,x0)     
# torch.save([err,color], Constants.outputs_path+'output1.pt')
# torch.save({'X':X, 'Y':Y, 'err':err}, Constants.outputs_path+'output1.pt')

