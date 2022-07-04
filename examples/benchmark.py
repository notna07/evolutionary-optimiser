"""
Created on Thu Feb  3 12:29:44 2022
@author: Anton Lautrup
"""
import sys
import time
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

#In Strats
from Strats import OptiStrat


ABC = OptiStrat.ArtBeeCol()
FuPSO = OptiStrat.FParticle()
Genetic = OptiStrat.GeneticOpt()
Genetic.O.percentage_outsiders = 0.2

Optimizers = [FuPSO, Genetic]#,ABC]

# -5 to 5, res: f(0,0)=0
def Ackley(X):
    x=X[0];y=X[1]
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20

# -100 to 100, res: f(0.0)=0
def Schaffer(X):
    x=X[0];y=X[1]
    return 0.5+(np.sin(x**2-y**2)**2-0.5)/((1+0.001*(x**2+y**2))**2)

#-1.2 to 1.2, res: f(1,1)=0
def Leon(X):
    x=X[0];y=X[1]
    return 100*(y-x**2)**2+(1-x)**2

# -10 to 10, res: f(various)=-2.06261
def Cross(X):
    x=X[0];y=X[1]
    return -0.0001*(abs(np.sin(x)*np.sin(y)*np.exp(abs(100-(np.sqrt(x**2+y**2))/np.pi)))+1)**0.1

#-2 to 2, res: f(0,-1)=3
def Goldstein(X):
    x=X[0];y=X[1]
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))

#-5 to 5, res: f(various)=0
def Himmelblau(X):
    x=X[0];y=X[1]
    return (x**2+y-11)**2+(x+y**2-7)**2

#-512 to 512, res: f(512,404)=-959
def Eggholder(X):
    x=X[0];y=X[1]
    return -(y+47)*np.sin(np.sqrt(np.abs(x/2+(y+47))))-x*np.sin(np.sqrt(np.abs(x-(y+47))))

functions = [Ackley, Schaffer, Leon, Cross, Goldstein, Himmelblau, Eggholder]

dims_lst = [[[-5,5]],[[-100,100]],[[-1.2,1.2]],[[-10,10]],[[-2,2]],[[-5,5]],[[-512,512]]]

Times1, Hists1 = [],[]

# cycles = 25
# n_agents = 300
# reps = 5

# for O in Optimizers:    
#     temp_t, temp_hist = [], []
#     for f,d in zip(functions,dims_lst):
#         loop_t, loop_h, loop_f = [],[],[]
#         for i in range(reps):
#             t = time.time()
#             O.reset()
#             O.set_search_space(d*2)
#             O.num_agents(n_agents)
#             O.set_fitness(f)
#             bestpos, bestf, hist = O.solve(cycles)
#             T = time.time() - t
#             loop_f.append(bestf)
#             loop_t.append(T)
#             loop_h.append(hist)
#         temp_t.append(np.mean(loop_t))
#         temp_hist.append(np.mean(loop_h,axis=0))
#         print(O.__class__.__name__ + ' completed ' + f.__name__ + '. Minimum found of ' + str(np.round(np.mean(loop_f),3)))
    
#     Times1.append(temp_t)
#     Hists1.append(temp_hist)

# for i in range(len(functions)):
#     plt.figure()
#     plt.title(functions[i].__name__)
#     plt.grid()
#     for O in range(len(Optimizers)):    
#         plt.plot(Hists1[O][i],label=Optimizers[O].__class__.__name__ + ', Time: ' + str(np.round(Times1[O][i],3)))
#     plt.legend()
#     plt.show()        
        
### Next is the stress test
# -5.12 to 5.12, res: f(0,...,0)=0
def Rastrigin(X,args):
    n = args["n"]   
    f = 10*n
    for i in range(n):
        f += X[i]**2-10*np.cos(2*np.pi*X[i])
    return  f

#-inf to inf, res: f(1,...,1)=0
def Rosenbrock(X,args):
    n = args["n"]
    f=0
    for i in np.arange(1,n-1):
        f += 100.0*(X[i+1]-X[i]**2)**2+(1-X[i])**2
    return f

ns = np.arange(6,30,4)

colors = ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:pink']

d2 = [[[-5.12,5.12]]]#,[[-1000,1000]]]
fs = [Rastrigin]#,Rosenbrock]
cycles = 100
n_agents = 100
reps = 100

for f,d in zip(fs,d2):
    Times2, Hists2 = [],[]
    for O in Optimizers:
        fig, axes = plt.subplots(figsize=(11,5),nrows=1, ncols=2)
        plt.rc('font',family='Bahnschrift')
        temp_t, temp_h = [],[]
        temp_l, temp_u = [],[]
        # for n in ns:
        #     t = time.time()
        #     O.reset()
        #     O.set_search_space(d*n)
        #     O.num_agents(n_agents)
        #     O.set_fitness(f,arguments={'n':n})
        #     bestpos, bestf, hist = O.solve(cycles)
        #     T = time.time() - t
        #     temp_t.append(T)
        #     temp_h.append(hist)
        for n in ns:
            times =[]
            hists = []
            sols = []
            #O.reset()
            O.set_search_space(d*n)    
            O.set_fitness(f,arguments={'n':n})
        
            for rep in range(reps):
                O.reset()
                O.num_agents(n_agents)
                t = time.time()
                x, s, h = O.solve(cycles)
                times.append(time.time() - t)
                hists.append(h)
                sols.append(x)
                
            temp_t.append(np.mean(times))
            temp_h.append(np.array(np.median(hists,axis=0)))
            temp_l.append(np.quantile(hists, 0.25,axis=0))
            temp_u.append(np.quantile(hists, 0.75,axis=0))
    
        Times2.append(temp_t)
        Hists2.append(temp_h)
        plt.suptitle(O.__class__.__name__ + ' ' + f.__name__)
        ax1 = plt.subplot(1,2,1)
        for i in range(len(ns)):
            #ax1.plot(temp_h[i], label='n= ' + str(ns[i]))
            ax1.plot(temp_h[i],linewidth=2,color=colors[i],label='n= ' + str(ns[i]))
            ax1.fill_between(np.arange(len(temp_l[i])),temp_l[i], temp_u[i],color=colors[i], alpha=.3)
            #ax1.plot(temp_l[i],linestyle='--')
            #ax1.plot(temp_u[i],linestyle='--')
        ax1.set_yscale('log')
        ax1.yaxis.grid(linewidth=0.5)
        ax1.set_xlabel('number of cycles')
        ax1.set_ylabel('fitness')
        ax1.legend()
        ax2 = plt.subplot(1,2,2)
        ax2.plot(ns,temp_t)
    plt.show()
    
        
    
    







