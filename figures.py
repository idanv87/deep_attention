import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

    # Plot data on the first subplot
    ax1.scatter(X.flatten(), Y.flatten(),color='black')
    ax1.set_title('Train domain')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')


    # Plot data on the second subplot
    x1=np.linspace(0,1/2,8)
    y1=np.linspace(0,1,15)
    x2=np.linspace(8/14,1,7)
    y2=np.linspace(0,1/2,8)
    X1,Y1=np.meshgrid(x1,y1, indexing='ij')
    X2,Y2=np.meshgrid(x2,y2, indexing='ij')
    ax2.scatter(X1.flatten(), Y1.flatten(), color='black')
    ax2.scatter(X2.flatten(), Y2.flatten(), color='black')
    ax2.set_title('Test domain')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')


    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(path+'fig1.eps', format='eps', bbox_inches='tight')  # Save the figure as an image file

    # Show the plots
    plt.show()
def fig2():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    err,c=torch.load(Constants.outputs_path+'output1.pt')
    for i in range(1,len(err)):
        ax1.scatter(i,np.log(err[i]),color='black', s=1)
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])    
    plot_Lshape(ax2)
    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    plt.tight_layout() 
    plt.savefig(path+'fig2.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show()    
fig2()


