import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'
x0=np.linspace(0,1,15)
y0=np.linspace(0,1,15)
X0,Y0=np.meshgrid(x0,y0,indexing='ij')
X0=X0.flatten()
Y0=Y0.flatten()

x=np.linspace(0,1,15)
X,Y=np.meshgrid(x,x, indexing='ij')
def plot_Lshape(ax2):
    x1=np.linspace(0,1/2,8)
    y1=np.linspace(0,1,15)
    x2=np.linspace(8/14,1,7)
    y2=np.linspace(0,1/2,8)
    X1,Y1=np.meshgrid(x1,y1, indexing='ij')
    X2,Y2=np.meshgrid(x2,y2, indexing='ij')
    ax2.scatter(X1.flatten(), Y1.flatten(), color='black',s=1)
    ax2.scatter(X2.flatten(), Y2.flatten(), color='black',s=1)
    plt.xticks([])
    plt.yticks([])

def fig1():
    # Create a figure and two subplots
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # Adjust figsize as needed

    # Plot data on the first subplot
    ax.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax.set_title('Train domain')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(path+'fig1.eps', format='eps', bbox_inches='tight', dpi=600) # Save the figure as an image file

    # Show the plots
    plt.show()


def fig2():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    data=torch.load(Constants.outputs_path+'output1.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    plt.tight_layout() 
    plt.savefig(path+'fig2.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 

def fig3():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    data=torch.load(Constants.outputs_path+'output2.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='red',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    plt.tight_layout() 
    plt.savefig(path+'fig3.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 
fig1()

