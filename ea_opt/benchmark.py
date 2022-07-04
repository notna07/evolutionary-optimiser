"""
Created on Mon Jul  4 2022
@author: Anton D. Lautrup

Script for asserting the optimization efficiency of the evo_opt heuristic.
"""

### imports 
import numpy as np
import matplotlib.pyplot as plt

import time
import math

from evo_opt import EvoOpt
 
### functions 
def f(X):
    x,y = X[0],X[1]
    res = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20
    return res 

def AssertPerformance(EA, num_attempts,num_cycles,epsilon,r_sol):
    times =[]
    hists = []
    sols = []
    
    for attempt in range(num_attempts):
        EA.StartOver()
        t = time.time()
        x, s, h = EA.Execute_SPORE2(num_cycles)
        times.append(time.time() - t)
        hists.append(h)
        sols.append(x)
    
    M_new_vec = np.array(np.median(hists,axis=0))
    lower_bound = np.quantile(hists, 0.25,axis=0)
    upper_bound = np.quantile(hists, 0.75,axis=0)
    
    plt.figure(figsize=(5,5))
    
    plt.title('Median and quantules of best fitness',fontsize=16)
    plt.plot(np.arange(len(M_new_vec)),M_new_vec,color='deepskyblue',linewidth=2)
    plt.fill_between(np.arange(len(M_new_vec)), lower_bound, upper_bound,color='lightskyblue', alpha=.3)
    plt.plot(np.arange(len(M_new_vec)),lower_bound,color='lightskyblue',linestyle='--')
    plt.plot(np.arange(len(M_new_vec)),upper_bound,color='lightskyblue',linestyle='--')
    plt.xlabel('number of cycles',fontsize=14)
    plt.ylabel('fitness',fontsize=14)
    plt.grid(linewidth=0.5)
    plt.yscale("log")
    plt.show()
    
    cor_count=0
    for x in sols:
        cor_count += epsilon > math.dist(r_sol,x)
    
    print('Average time per attempt; ', np.mean(times))
    print(cor_count,' agents came within ',epsilon, ' units of target')
    pass

def main(): 
    """Launcher.""" 
    
    EA = EvoOpt(pop_size=10)
    EA.set_dims([[-5,5]]*2)
    EA.set_fitness(f)
    EA.mutation_chance = 0.15
    #EA.percentage_to_select = 0.20
    #EA.percentage_outsiders = 0.2
    
    AssertPerformance(EA, num_attempts=25, num_cycles=250, epsilon=0.1, r_sol=[0,0])
    
    pass 
 
if __name__ == "__main__": 
	main() 