"""
Created on Mon Jul  4 2022
@author: Anton D. Lautrup

This script holds examples of how the evo_opt heuristic is run
"""

### imports 
import numpy as np
import matplotlib.pyplot as plt

import time

from evo_opt import EvoOpt

#import scipy as sp
 
### functions 
def f(X):
    x,y = X[0],X[1]
    res = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20
    return res 

def main(): 
    """Launcher.""" 
    # init the GUI or anything else 
    X = np.arange(-5, 5, 0.01)
    Y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(X, Y)
       
    fig, ax = plt.subplots(figsize=(8, 7))
    c = ax.pcolormesh(X, Y, f([X,Y]), cmap='RdBu',shading='auto')
    ax.set_title('Ackley function landscape',fontsize=16)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)
    
    EA = EvoOpt(pop_size=10)
    EA.set_dims([[-5,5]]*2)
    EA.set_fitness(f)
    EA.mutation_chance = 0.15
    #EA.percentage_to_select = 0.20
    #EA.percentage_outsiders = 0.2
    
    t = time.time()
    x, s, h = EA.Execute_SPORE2(num_cycles=10)
    print(time.time() - t) 

    print('Best Fitness found at %10s, with score of %d' % (x, s))
    hist = np.array(EA.pos_hist).T
    plt.plot(hist[0],hist[1],'m.',markersize=10)
    plt.plot(x[0],x[1],'r.',markersize=18)  
    print(h)
    plt.show()        
     
    plt.figure()
    plt.plot(h)

    plt.show()   
    
    pass 
 
if __name__ == "__main__": 
	main()