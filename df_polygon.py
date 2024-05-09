import numpy as np
import matplotlib.pyplot as plt
from test_deeponet import domain
from utils import *
from scipy.sparse import csr_matrix, kron, identity
def generate_domains(i1,i2,j1,j2):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    x_ref=d_ref.x
    y_ref=d_ref.y
    return domain(x_ref[i1:i2], y_ref[j1:j2])

def generate_f_g(shape, seedf):

        f=generate_random_matrix(shape,seed=seedf)
        
        # f=(f-np.mean(f))/np.std(f)
        f=f/np.std(f)
       
        return f+1

def masking_coordinates(X,Y):
        xx,yy=np.meshgrid(np.linspace(0,1, Constants.n),np.linspace(0,1, Constants.n),indexing='ij')
        X0=xx.flatten()
        Y0=yy.flatten()
        original_points=[(X0[i],Y0[i]) for i in range(len(X0))]
        points=np.array([(X[i],Y[i]) for i in range(len(X))])
        valid_indices=[]
        masked_indices=[]
        for j,p in enumerate(original_points):
            dist=[np.linalg.norm(np.array(p)-points[i]) for i in range(points.shape[0])]
            if np.min(dist)<1e-14:
                valid_indices.append(j)
            else:
                masked_indices.append(j)    
        return valid_indices, masked_indices     
        
def generate_example():  
    x1=np.linspace(0,1/2,8) 
    y1=np.linspace(0,1,15)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(8/14,1,7) 
    y2=np.linspace(0,1/2,8)
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D1=d1.D.todense()
    D2=d2.D.todense()
    D=block_matrix(D1,D2)

    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    

    intersection_indices_l=[105,106,107,108,109,110,111,112]
    l_jump=-15
    r_jump=15

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[105,105]=-4/dx/dx-2/dx*Constants.l
    D[105,90]=1/dx/dx
    D[105,120]=1/dx/dx
    D[105,106]=2/dx/dx

    intersection_indices_r=[120,121,122,123,124,125,126,127]
    l_jump=-15
    r_jump=8

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[120,120]=-4/dx/dx-2/dx*Constants.l
    D[120,12+l_jump]=1/dx/dx
    D[120,120+r_jump]=1/dx/dx
    D[120,121]=2/dx/dx

    D[127,127]=-4/dx/dx-2/dx*Constants.l
    D[120,12+l_jump]=1/dx/dx
    D[120,120+r_jump]=1/dx/dx
    D[120,119]=2/dx/dx

        
    xx,yy=np.meshgrid(np.linspace(0,1, Constants.n),np.linspace(0,1, Constants.n),indexing='ij')
    X0=xx.flatten()
    Y0=yy.flatten()    
    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    from main import generate_f_g
    f=generate_f_g(len(X), 1)

    f_ref[valid_indices]=f
    # f_ref=torch.tensor(f_ref, dtype=torch.float32)
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),f_ref,f,dom,mask, X,Y, valid_indices
def generate_rect():
    
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    d=generate_domains(8,15, 2,8)
    f=generate_f_g(d.nx*d.ny, 500)
    f_ref[d.valid_indices]=f
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, d.non_valid_indices] = float('-inf')  
    A,G=d.solver(f.reshape((d.nx,d.ny)))
    dom=np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1)))
    return A, f_ref,f,dom,mask, d.X, d.Y, d.valid_indices
        
# D,f,dom,mask=generate_example()

