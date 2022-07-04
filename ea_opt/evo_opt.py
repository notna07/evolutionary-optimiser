"""
Created on Thu Oct 21 2021
@author: Anton Danholt Lautrup

This is an evolutionary algorithm optimizer, 
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

class EvoOpt():
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
        #self.pop_size = len(self.Agents)
        del self.Agents[:]
        self.pos_hist = []
        self.sco_hist = []
        self.day = 0
        print('Attention! The population has been purged!')

    def Execute_SPORE2(self,num_cycles):
        """This is the main function that runs the optimization
        
        Args:
            n_cycles: int - how many iterations to optimise for
        """
        if self.day == 0:   #Initilazition of the Agents of the first day.
            self._setup_agents()
            
        for c in range(num_cycles):
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
                                                           
if __name__ == '__main__':
    raise Exception("ERROR: Please call the code from another script")   

            
            








