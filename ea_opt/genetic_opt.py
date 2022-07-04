"""
Created on Thu Oct 21 12:21:32 2021
@author: Anton Danholt Lautrup

This is a genetic optimizer, 
"""

import numpy as np
import itertools

class Agent(object):
    """The Agent class, don't touch!"""
    id_iter = itertools.count()
    
    def __init__(self, pos):
        self.ID = next(Agent.id_iter)
        self.pos = pos
        self.score = np.inf
        pass

class Genetic():
    """The Optimizer class
    
    Args:
        pop_size: int - how many agents to use for optimization
        species_name: str - custom tag, to keep track of multiple optimizers
        
    Attributes:
        mutation_chance: float - choose chance in ]0,1[ for mutation to happen
        percentage_to_select: float - choose amount in ]0,1[ for crossover
        selection_type: keyword - choose either 'random' or 'sexual'
        percentage_outsiders: float - unstable feature, choose amount of new 
                                        individuals to be randomly built to add
                                        new DNA to genepool.
        do_parallel: bool - advanced feature that requires your code to do some
                                of the dirty work of parallellization.
                                
    Methods:
        set_dims(limits: list):
            Set dimension and range of search space.
            
        set_fitness(fitness_func: function, arguments: dict)
            Set the function to use for fitness assesment.
            
        Execute_SPORE2(n_cycles: int)
            Main function for starting optimisation. Requires to use set_dims
            and set_fitness first.
        
        StartOver()
            Method for use to reset the algorithm.
    """
    
    def __init__(self,pop_size=100,species_name='Agentosaurus'):
        self.day = 0
        
        self.pop_size = pop_size

        self.mutation_chance = 0.1
        self.percentage_to_select = 0.2
        self.selection_type = 'random'  # 'sexual' or 'random' 
        

        self.percentage_outsiders = 0
        
        self.Agents = []
        self.pos_hist = []
        self.sco_hist = []
        self.name = species_name
        
        self.do_parallel = False
        pass
    
    def set_dims(self, limits):
        """Sets the boundaries of the search space.
		
        Args:
			limA,limB: it can be either a 2D list or 2D array, shape = (dimensions, 2).
                if you have D=3 variables in the real interval [-1,1]
			    you can provide the boundaries as: [[-1,1], [-1,1], [-1,1]].
		        The dimensions of the problem are automatically determined 
		        according to the length of 'lim'.
		"""
        self.dim = len(limits)
        self.bounds = list(np.array(limits).T)
        pass
        
    def set_fitness(self,fitness,arguments=None,skip_test=True):
        """Sets the fitness function used to evaluate the score of the agents of species A.
			
        Args:
			fitness : a user-defined function. The function must accept
    			a vector of real-values as input (i.e., a candidate solution)
                return a real value (i.e., the corresponding fitness value)
			arguments : a dictionary containing the arguments to be passed to the fitness function.
		"""
        if skip_test:
            self.FITNESS = fitness
            self._FITNESS_ARGS = arguments
            return
        
        self.FITNESS = fitness
        self._FITNESS_ARGS = arguments
        
        if self.bounds == []:
            test_agent = [[1e-10]]*self.dim
        else:
            test_agent = np.random.uniform(self.bounds[0],self.bounds[1],self.dim)
        self._call_fitness(test_agent,self._FITNESS_ARGS)
        print('Fitness function has been tested, and approved.')
        pass
        
    def _call_fitness(self,data,arguments=None):
        """Utility function for allowing the fitness function with arguments 
        to be callable from within the Genetic() class
        
        Args:
            data: vector of values to evaluate using the fitness function.
            arguments: a dictionary containing the arguments to be passed to the fitness function.
        """
        if self.FITNESS == None: raise Exception("ERROR: fitness function not valid!")
        if arguments==None:
            return self.FITNESS(data)
        else:
            return self.FITNESS(data, arguments)
    
    def _setup_agents(self):
        del self.Agents[:]
        print('Agents are being initialized')
        if self.do_parallel:
            Ts = np.random.uniform(self.bounds[0],self.bounds[1],(self.pop_size,self.dim))
            costs = self._call_fitness(Ts, self._FITNESS_ARGS)
            
        for i in range(self.pop_size):
            x = np.random.uniform(self.bounds[0],self.bounds[1],self.dim)
            agent = Agent(x)     #Create Species A agents
            # plt.plot(x[0],x[1],'w.',markersize=18, markeredgecolor='k',zorder=9)
            
            if self.do_parallel:
                agent.score = costs[i]
            else:
                agent.score = self._call_fitness(x, self._FITNESS_ARGS)
            self.Agents.append(agent)
        
        self.Agents.sort(key=lambda x: x.score)
        pass
    
    def _crossover(self,agents):
        offspring = []
        if self.selection_type == 'sexual': 
            s_lst = [agent.score for agent in agents]
            w_lst = s_lst/sum(s_lst)
            
        for _ in range((self.pop_size - (len(agents)+int(self.percentage_outsiders*self.pop_size))) // 2):
            if self.selection_type == 'sexual':
                parent1 = np.random.choice(agents)
                parent2 = np.random.choice(agents,p=w_lst)    
            else: #selection_type 'random'
                parent1 = np.random.choice(agents)
                parent2 = np.random.choice(agents)
            
            r = np.random.choice((0,1),len(parent1.pos))
            
            #The offspring are twins 
            # child1 = Agent(parent1.pos*r+parent2.pos*(1-r))
            # child2 = Agent(parent1.pos*r+parent2.pos*(1-r))
    
            #The offspring are opposites
            child1 = Agent(parent1.pos*r+parent2.pos*(1-r))
            child2 = Agent(parent2.pos*r+parent1.pos*(1-r))
            
            offspring.append(child1)
            offspring.append(child2)
            
        # for agent in offspring:
        #     plt.plot(agent.pos[0],agent.pos[1],'cyan',marker=".",markersize=18,markeredgecolor='k',zorder=7)
        
        agents.extend(offspring)
        return agents
    
    def _mutation(self,agents):
        for agent in agents:
            if np.random.uniform(0.0, 1.0) <= self.mutation_chance:
                randint = np.random.randint(0,len(agent.pos))
                #plt.plot(agent.pos[0],agent.pos[1],'lime',marker="x",markersize=10,markeredgecolor='k',zorder=8)
                agent.pos[randint] = np.random.uniform(self.bounds[0][randint],self.bounds[1][randint])
                #plt.plot(agent.pos[0],agent.pos[1],'lime',marker=".",markersize=18,markeredgecolor='k',zorder=8)
        return agents

    def _outsiders(self,agents):
        outsiders = []
        for _ in range(int(self.percentage_outsiders*self.pop_size)):
            outsiders.append(Agent(np.random.uniform(self.bounds[0],self.bounds[1],(self.dim))))
        agents.extend(outsiders)
        return agents
    
    def StartOver(self):
        """Use this to reset the optimiser"""
        self.pop_size = len(self.Agents)
        del self.Agents[:]
        self.pos_hist = []
        self.sco_hist = []
        self.day = 0
        print('Attention! The population has been purged!')

    def Execute_SPORE2(self,n_cycles):
        """This is the main function that runs the optimization
        
        Args:
            n_cycles: int - how many iterations to optimise for
        """
        if self.day == 0:   #Initilazition of the Agents of the first day.
            self._setup_agents()
            
        for c in range(n_cycles):
            #Do selection of fit individuals
            Selected = self.Agents[:int(self.percentage_to_select * len(self.Agents))]
            
            # for agent in self.Agents[int(self.percentage_to_select * len(self.Agents)):]:
            #     plt.plot(agent.pos[0],agent.pos[1],'rx',markersize=10,zorder=10)
            
            NextGen = self._crossover(Selected)
            
            Mutated = self._mutation(NextGen)
            
            self.Agents = self._outsiders(Mutated)

            if self.do_parallel:
                Ts = []
                for agent in self.Agents:
                    Ts.append(agent.pos)

                res = self._call_fitness(Ts,self._FITNESS_ARGS)
                for i in range(self.pop_size):
                    self.Agents[i].score = res[i]
            else:
                for agent in self.Agents:
                    agent.score = self._call_fitness(agent.pos,self._FITNESS_ARGS)
            
            #Finally we log the progress the colony have made.
            self.Agents.sort(key=lambda x: x.score)
            
            self.pos_hist.append(self.Agents[0].pos.copy())
            self.sco_hist.append(self.Agents[0].score.copy())
            print(self.name + ' %d was best adapted with fitness %.2f (day %d)' % (self.Agents[0].ID,self.Agents[0].score,self.day))
            self.day += 1
            #plt.plot(self.Agents[0].pos[0],self.Agents[0].pos[1],'y*',markersize=10,zorder=10)
        return self.Agents[0].pos, self.Agents[0].score, self.sco_hist

import math
import seaborn as sns
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
    
    plt.title('Median and quantules of best fitness',fontsize=16,fontname='Bahnschrift')
    plt.plot(np.arange(len(M_new_vec)),M_new_vec,color='deepskyblue',linewidth=2)
    plt.fill_between(np.arange(len(M_new_vec)), lower_bound, upper_bound,color='lightskyblue', alpha=.3)
    plt.plot(np.arange(len(M_new_vec)),lower_bound,color='lightskyblue',linestyle='--')
    plt.plot(np.arange(len(M_new_vec)),upper_bound,color='lightskyblue',linestyle='--')
    plt.xlabel('number of cycles',fontsize=14,fontname='Bahnschrift')
    plt.ylabel('fitness',fontsize=14,fontname='Bahnschrift')
    plt.grid(linewidth=0.5)
    plt.yscale("log")
    plt.show()
    
    cor_count=0
    for x in sols:
        cor_count += epsilon > math.dist(r_sol,x)
    
    print('Average time per attempt; ', np.mean(times))
    print(cor_count,' agents came within ',epsilon, ' units of target')
    pass
                                                            
                                                           
if __name__ == '__main__':
    #raise Exception("ERROR: Please call the code from another script")
    
    #Here is an example of how the code is run
    def f(X):
        x,y = X[0],X[1]
        res = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20
        #res = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
        #res = (x**2+y-11)**2+(x+y**2-7)**2
        #res = -(y+47)*np.sin(np.sqrt(np.abs(x/2+(y+47))))-x*np.sin(np.sqrt(np.abs(x-(y+47))))
        return res 
    
    # k_hist = []
    # def Callback(Xi,convergence):
    #     k_hist.append(f(Xi))
    #     pass
    
    X = np.arange(-5, 5, 0.01)
    Y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(X, Y)
    
    import matplotlib.pyplot as plt
    import time 
    import scipy as sp
    
    fig, ax = plt.subplots(figsize=(8, 7))
    c = ax.pcolormesh(X, Y, f([X,Y]), cmap='RdBu',shading='auto')
    ax.set_title('Ackley function landscape',fontsize=16,fontname='Bahnschrift')
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)
    
    GA = Genetic(50)
    GA.set_dims([[-5,5]]*2)
    GA.set_fitness(f)
    GA.mutation_chance = 0.15
    #GA.percentage_to_select = 0.20
    #GA.percentage_outsiders = 0.2
    t = time.time()
    x, s, h = GA.Execute_SPORE2(1)
    print(time.time() - t) 
    
    #K = sp.optimize.differential_evolution(f,[[-512,512]]*2,callback=Callback,maxiter=100,popsize=100,mutation=0.15,recombination=0.2)
        
    print('Best Fitness found at %10s, with score of %d' % (x, s))
    hist = np.array(GA.pos_hist).T
    plt.plot(hist[0],hist[1],'m.',markersize=10)
    plt.plot(x[0],x[1],'r.',markersize=18)  
    #print(h)
    plt.show()        
     
    plt.figure()
    plt.plot(h)
    #plt.plot(k_hist) 
    plt.show()      

            
            








