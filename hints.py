import os
import sys
import math
from matplotlib.ticker import ScalarFormatter

import time

from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf

from utils import upsample
from constants import Constants
from utils import  grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D

from two_d_model import  deeponet
from test_deeponet import domain
from main import generate_f_g

def NN( F, X, Y):
    
    x_domain=X.flatten()
    y_domain=Y.flatten()
    int_points=np.vstack([x_domain,y_domain]).T
 
    n=15
    model=deeponet(dim=2,f_shape=n**2, domain_shape=2, p=80) 
    best_model=torch.load(Constants.path+'runs/'+'2024.05.01.21.20.36best_model.pth')
    model.load_state_dict(best_model['model_state_dict'])
  

    with torch.no_grad():
       
        y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
        f=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        tensor_to_repeat= torch.tensor(np.hstack((d.X.reshape(-1, 1), d.Y.reshape(-1, 1))),dtype=torch.float32)
        dom= tensor_to_repeat.unsqueeze(0).repeat(y1.shape[0], 1, 1)
        pred2=model([y1, f,dom])

    return torch.real(pred2).numpy()+1J*torch.imag(pred2).numpy()
   
n=15
x=np.linspace(0,1,n)
y=np.linspace(0,1,n)
d=domain(x,y)
xx,yy=np.meshgrid(x,y,indexing='ij')
f,ga,gb,gc,gd=generate_f_g(n, 400,500, 400)
A,b=d.solver(f.reshape((n,n)),[ga,gb,gc,gd])
sol=scipy.sparse.linalg.spsolve(A, b)
NN(f,xx,yy)
x0=(f+1J*f)*0.0001

def hints(A,b,x0):
    for k in range(1000):
        if (k+1)%3==0:
            f_real=(-A@x0+b).real
            f_imag=(-A@x0+b).imag
            mu_real=np.mean(f_real)
            s_real=np.std(f_real)
            mu_imag=np.mean(f_imag)
            s_imag=np.std(f_imag)
            
            corr_real=(NN((f_real-mu_real)/s_real,xx,yy)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
            corr_imag=(NN((f_imag-mu_imag)/s_imag,xx,yy)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
            corr=corr_real+1J*corr_imag
            x0=x0+corr
            
        else:
            x0=Gauss_zeidel(A.todense(),b,x0,theta=1)[0]
        print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
        
hints(A,b,x0)     
