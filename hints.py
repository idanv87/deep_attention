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


from utils import upsample, fft
from constants import Constants
from utils import  grf, evaluate_model, generate_grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D, gs_new

from two_d_model import  deeponet, deeponet_van
from test_deeponet import domain
from main import generate_f_g
from df_polygon import generate_example, generate_rect, generate_rect2, generate_example_2, generate_obstacle, generate_obstacle2, generate_two_obstacles

# 2024.06.06.12.21.27best_model.pth several domains
# 2024.06.07.09.57.24best_model.pth single domain 

# 2024.06.05.12.50.00best_model.pth small domain  
# 2024.05.16.19.26.50best_model.pth with dom

model=deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.06.06.12.21.27best_model.pth')
# best_model=torch.load(Constants.path+'runs/'+'2024.06.07.09.57.24best_model.pth')
model.load_state_dict(best_model['model_state_dict'])


   

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
# A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
# A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle()
# x0=(f+1J*f)*0.001
# sol=scipy.sparse.linalg.spsolve(A, f)
# NN(f_ref,X,Y,dom, mask)


# l,v=scipy.sparse.linalg.eigs(A+A.conjugate().T, k=2,which='SR')
# x, exitCode = scipy.sparse.linalg.gmres(A, f,x0, tol=1e-13, maxiter=100)
# print(np.linalg.norm(A@x-f)/np.linalg.norm(f))


# b=f
# x0=evaluate_model(b,valid_indices,NN,NN2,X,Y, dom,mask)
# print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
err=[]
color=[]
# l,v=gs_new(A.todense())
def single_hints(f, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices):
            f_real=f.real
            f_imag=f.imag
            s_real=np.std(f_real)
            s_imag=np.std(f_imag)
            f_ref_real=np.zeros(Constants.n**2)
            f_ref_imag=np.zeros(Constants.n**2)
            
            
            f_ref_real[valid_indices]=(f_real)
            f_ref_imag[valid_indices]=(f_imag)
        
            corr_real=(NN(f_ref_real,X,Y, dom, mask))
            corr_imag=(NN(f_ref_imag,X,Y, dom, mask))
            # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
            # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
            
            
            return corr_real+1J*corr_imag

# print((1/np.linalg.norm(l[0]**(J+1)))*np.linalg.norm(evaluate_model(v[:,0]*l[0]**J,valid_indices,d,d_super,NN,NN2,X,Y, dom,mask)))
def hints(A,b,x0, J, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices):
    err=[]
    spec1=[]
    spec2=[]
    for k in range(6000):
        if (k+1)%J==0:
        # if True:    
            # corr=evaluate_model(-A@x0+b,valid_indices,NN,NN2,X,Y, dom,mask)
            
            f_real=(-A@x0+b).real
            f_imag=(-A@x0+b).imag
            
            # func_real=interpolation_2D(X,Y,f_real)
            # func_imag=interpolation_2D(X,Y,f_imag)
            # f_real=np.array(func_real(X_ref,Y_ref))
            # f_imag=np.array(func_imag(X_ref,Y_ref))
            

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
        if k %10 ==0:    
            print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
            print(k)
        try:
            spectrum, freqx, freqy=fft((abs((A@x0-b)/b)).reshape((8,15)),8,15)
            spec1.append(abs(spectrum[4,7]))
            spec2.append(abs(spectrum[0,0]))
        except:
            pass
        err.append(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
        if err[-1]<1e-13 or err[-1]>100:
            break
    return err, color, J, alpha,k, spec1, spec2    
    # return err, color, J, alpha,k 


def exp1(sigma=0.1,l=0.2,mean=0):
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
    F=generate_grf(X,Y,20)
    all_k=[]
    for i in range(20):
        b=F[i]
        f_ref[valid_indices]=b
        x0=(b+1J*b)*0.001
        err, color, J, alpha, k=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output11.pt')    




def exp2(sigma=0.1,l=0.1,mean=0):
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect2(8)
    F=generate_grf(X,Y,20)
    all_k=[]
    for i in range(20):
        b=F[i]
        x0=(b+1J*b)*0.001
        f_ref[valid_indices]=b
        err, color, J, alpha, k=hints(A,b,x0,J=20, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output13.pt') 



def exp3(sigma=0.1,l=0.2,mean=0):
    A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
    F=generate_grf(X,Y,20)
    all_k=[]
    for i in range(20):
        b=F[i]
        func=interpolation_2D(X,Y,b)
        f_ref[valid_indices]=func(X_ref,Y_ref)
        x0=(b+1J*b)*0.001
        
        err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output14.pt')     
    


def exp4(sigma=0.4,l=0.4,mean=1):
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
    F=generate_grf(X,Y,20, sigma,l,mean)
    all_k=[]
    for i in range(20):
        b=F[i]
        f_ref[valid_indices]=b
        x0=(b+1J*b)*0.001
        err, color, J, alpha, k=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output15.pt')    

def exp5(sigma=0.4,l=0.4,mean=1):
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect2(8)
    F=generate_grf(X,Y,20, sigma,l,mean)
    all_k=[]
    for i in range(20):
        b=F[i]
        x0=(b+1J*b)*0.001
        f_ref[valid_indices]=b
        err, color, J, alpha, k=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output16.pt') 

def exp6(sigma=1,l=0.7,mean=1):
    A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
    F=generate_grf(X,Y,20, sigma,l,mean)
    all_k=[]
    for i in range(20):
        b=F[i]
        func=interpolation_2D(X,Y,b)
        f_ref[valid_indices]=func(X_ref,Y_ref)
        x0=(b+1J*b)*0.001
        
        err, color, J, alpha, k=hints(A,b,x0,J=5, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output17.pt')  
    
def exp7(sigma=1,l=0.7,mean=1):
    A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle2(29)
    F=generate_grf(X,Y,20, sigma,l,mean)
    all_k=[]
    for i in range(20):
        b=F[i]
        func=interpolation_2D(X,Y,b)
        f_ref[valid_indices]=func(X_ref,Y_ref)
        x0=(b+1J*b)*0.001
        
        err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    # torch.save({'X':X, 'Y':Y,'J':J,'all_k':all_k}, Constants.outputs_path+'output18.pt')     

def exp8(sigma=1,l=0.7,mean=1):
    A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle2(29)
    F=generate_grf(X,Y,20, sigma,l,mean)
    all_k=[]
    for i in range(20):
        b=F[i]
        func=interpolation_2D(X,Y,b)
        f_ref[valid_indices]=func(X_ref,Y_ref)
        x0=(b+1J*b)*0.001
        
        err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'J':J,'all_k':all_k}, Constants.outputs_path+'output19.pt')  

def exp9(sigma=1,l=0.7,mean=1):
    A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_two_obstacles()
    F=generate_grf(X,Y,20, sigma,l,mean)
    all_k=[]
    for i in range(20):
        b=F[i]
        func=interpolation_2D(X,Y,b)
        f_ref[valid_indices]=func(X_ref,Y_ref)
        x0=(b+1J*b)*0.001
        
        err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'J':J,'all_k':all_k}, Constants.outputs_path+'output20.pt')  

def exp10(sigma=0.1,l=0.1,mean=0):
    err=[]
    alpha=1
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
    # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
    X_ref=X
    Y_ref=Y
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    # F=generate_grf(d_ref.X, d_ref.Y, n_samples=1, seed=1)[0]
    # F=F[valid_indices]
    F=generate_grf(X, Y, n_samples=50, seed=10)
    for i in range(50):
        sol=scipy.sparse.linalg.spsolve(A, F[i])
        err.append(np.linalg.norm(single_hints(F[i], alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices)-sol)/np.linalg.norm(sol))
        # print(err[-1])
    print(np.mean(err))
    print(np.std(err) )
    # spectrum, freqx, freqy=fft(err.reshape((8,15)),8,15)
    # print(abs(spectrum[0,1]))
    # print(abs(spectrum[4,7]))
    
    
   
 
def exp11(sigma=0.1,l=0.1,mean=0):
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
    # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
    F=generate_grf(X,Y,l=0.1,n_samples=1, seed=10)
    all_k=[]
    for i in range(1):
        b=F[i]
        f_ref[valid_indices]=b
        x0=(b+1J*b)*0.001
        err, color, J, alpha, k,spec1,spec2=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
        torch.save({'low':spec1,'high':spec2}, Constants.outputs_path+'spec_mult.pt')  
        all_k.append(k)
exp11()        
# spec_single=torch.load(Constants.outputs_path+'spec_single.pt')        
# spec_mult=torch.load(Constants.outputs_path+'spec_mult.pt')

# plt.plot(spec_single['low'][:40])  
# plt.plot(spec_mult['low'][:40],color='red')
# plt.show()

def exp12(sigma=0.1,l=0.2,mean=0):
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
    F=generate_grf(X,Y,20)
    
    all_k=[]
    for i in range(20):
        b=F[i]
        u=scipy.sparse.linalg.spsolve(A, b)
        f_ref[valid_indices]=b
        x0=(b+1J*b)*0.001
        err, color, J, alpha, k=hints(A,b,x0,J=5, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
        all_k.append(k)
    torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output20.pt') 

def exp13(sigma=0.1,l=0.2,mean=0):
    alpha=1
    A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
    # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
    F=generate_grf(X,Y,1,seed=2)
    
    
    all_k=[]
    for i in range(20):

        u=F[i]*0
        u[32:35]=F[i][32:35]
        u[32:35]=F[i][47:50]
        u[32:35]=F[i][62:65]
        u[40:42]=F[i][40:42]
        u[55:57]=F[i][55:57]
        u[70:72]=F[i][70:72]
        u[85:87]=F[i][85:87]
        u[146:149]=F[i][146:149]
        u[154:157]=F[i][154:157]
        u=F[i]
        b=A@(u/480)
        

        sol=scipy.sparse.linalg.spsolve(A, b)
        X_ref=X
        Y_ref=Y
        err=single_hints(b, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices)-sol
        # print(np.linalg.norm(err))
        print(np.linalg.norm(err)/np.linalg.norm(sol))



# data=torch.load(Constants.outputs_path+'output20.pt')
# print(np.mean(data['all_k']))    
# print(np.std(data['all_k']))  
# def eval(f, alpha,X,Y,dom,mask, X_ref=None, Y_ref=None):
    
#     f_real=f.real
#     f_imag=f.imag
#     # try:
#     func_real=interpolation_2D(X,Y,f_real)
#     func_imag=interpolation_2D(X,Y,f_imag)
#     f_real=np.array(func_real(X_ref,Y_ref))
#     f_imag=np.array(func_imag(X_ref,Y_ref))
#     # except:
#     #     pass    
    

#     s_real=np.std(f_real)/alpha*0+1
#     s_imag=np.std(f_imag)/alpha*0+1
#     f_ref_real=np.zeros(Constants.n**2)
#     f_ref_imag=np.zeros(Constants.n**2)
    
    
#     f_ref_real[valid_indices]=(f_real)/s_real
#     f_ref_imag[valid_indices]=(f_imag)/s_imag

#     corr_real=(NN(f_ref_real,X,Y, dom, mask))*s_real
#     corr_imag=(NN(f_ref_imag,X,Y, dom, mask))*s_imag
#     # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
#     # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
#     corr=corr_real+1J*corr_imag


#     return corr   
   
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