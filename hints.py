for v in dir():
    exec('del '+ v)
    del v

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
from utils import  grf, evaluate_model

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D, gs_new

from two_d_model import  deeponet
from test_deeponet import domain
from main import generate_f_g
from df_polygon import generate_example, generate_rect, generate_rect2, generate_example_2, generate_obstacle


    
model=deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.05.16.19.26.50best_model.pth')
model.load_state_dict(best_model['model_state_dict'])

model_mu=deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
# 2024.05.09.07.51.45best_model.pth
best_model=torch.load(Constants.path+'runs/'+'2024.05.15.02.19.55best_model.pth')
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
   

# A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
# A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()

# A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect2(8)
A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
# A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle()
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
# x0=evaluate_model(b,valid_indices,NN,NN2,X,Y, dom,mask)
# print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
err=[]
color=[]
# l,v=gs_new(A.todense())

# print((1/np.linalg.norm(l[0]**(J+1)))*np.linalg.norm(evaluate_model(v[:,0]*l[0]**J,valid_indices,d,d_super,NN,NN2,X,Y, dom,mask)))
def hints(A,b,x0, J, alpha):
    for k in range(3000):
        if (k+1)%J==0:
            # corr=evaluate_model(-A@x0+b,valid_indices,NN,NN2,X,Y, dom,mask)
            
            f_real=(-A@x0+b).real
            f_imag=(-A@x0+b).imag
            
            func_real=interpolation_2D(X,Y,f_real)
            func_imag=interpolation_2D(X,Y,f_imag)
            f_real=np.array(func_real(X_ref,Y_ref))
            f_imag=np.array(func_imag(X_ref,Y_ref))
            

            s_real=np.std(f_real)/alpha
            s_imag=np.std(f_imag)/alpha
            f_ref_real=np.zeros(Constants.n**2)
            f_ref_imag=np.zeros(Constants.n**2)
            
            
            f_ref_real[valid_indices]=(f_real)/s_real
            f_ref_imag[valid_indices]=(f_imag)/s_imag
        
            corr_real=(NN(f_ref_real,X,Y, dom, mask))*s_real
            corr_imag=(NN(f_ref_imag,X,Y, dom, mask))*s_imag
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
        if err[-1]<1e-13:
            break
    return err, color, J, alpha    
err, color, J, alpha, =hints(A,b,x0,J=5, alpha=1)     
torch.save({'X':X, 'Y':Y, 'err':err,'J':J, 'alpha':alpha}, Constants.outputs_path+'output3.pt')


def eval(f, alpha,X,Y,dom,mask, X_ref=None, Y_ref=None):
    
    f_real=f.real
    f_imag=f.imag
    # try:
    func_real=interpolation_2D(X,Y,f_real)
    func_imag=interpolation_2D(X,Y,f_imag)
    f_real=np.array(func_real(X_ref,Y_ref))
    f_imag=np.array(func_imag(X_ref,Y_ref))
    # except:
    #     pass    
    

    s_real=np.std(f_real)/alpha
    s_imag=np.std(f_imag)/alpha
    f_ref_real=np.zeros(Constants.n**2)
    f_ref_imag=np.zeros(Constants.n**2)
    
    
    f_ref_real[valid_indices]=(f_real)/s_real
    f_ref_imag[valid_indices]=(f_imag)/s_imag

    corr_real=(NN(f_ref_real,X,Y, dom, mask))*s_real
    corr_imag=(NN(f_ref_imag,X,Y, dom, mask))*s_imag
    # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
    # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
    corr=corr_real+1J*corr_imag


    return corr   
   
# A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
# A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect2(8)
# X_ref=None
# Y_ref=None
# A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
# l,v=gs_new(A.todense())
# fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
# V0=[]
# V1=[]
# V2=[]
# # print(v.shape)
# for j in [10,20,30,40,50,60]:
#     V0.append(np.linalg.norm(eval(l[0]**j*v[:,0],0.5,X,Y,dom,mask,X_ref, Y_ref)/v[:,0]/l[0]**(j+1)))
#     V1.append(np.linalg.norm(eval(l[1]**j*v[:,1],0.5,X,Y,dom,mask, X_ref,Y_ref)/v[:,1]/l[1]**(j+1)))
#     # V2.append(np.linalg.norm(eval(l[2]**j*v[:,2],0.5,X,Y,dom,mask)/l[2]**j))

# ax1.plot(V0, label='0')    
# ax1.plot(V1, label='1')  
# # # ax1.plot(V2, label='2')  
# print(abs(l[0]))
# print(abs(l[1]))
# print(abs(l[2]))
# # print(abs(l[-1]))
# plt.legend()
# plt.show()