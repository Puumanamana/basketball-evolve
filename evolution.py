#!/usr/bin/env python

from itertools import combinations
from time import time

import numpy as np
from scipy.stats import cauchy,norm
import matplotlib.pyplot as plt

from player import Player

np.random.seed(1234)

# Location of the hoop
TARGET = np.array([1000,300])

def timer(fun):
    def fun_wrapper(x):
        t0 = time()
        fun(x)
        print('{0} - Elapsed time: {1:.3f}'.format(fun.__name__,time()-t0))
    return fun_wrapper

class Evolution:

    def __init__(self,N):
        self.pop_size = N
        self.populations = {i: Player(i) for i in range(N)}
        # 50% of the population replaced with children
        self.recombine_prob = 0.5
        # 10% of the population gets mutated
        self.mutation_rate = 0.1
        # Record the fitnesses for each generation (to display it in the end)
        self.fitnesses = []

    def calc_fitnesses(self):
        fitnesses = []
        for player in self.populations.values():
            player.evaluate(TARGET)
            fitnesses.append(player.fitness)
        self.fitnesses.append(fitnesses)

    def select_next_gen(self):
        fitnesses = np.array([ player.fitness for player in self.populations.values() ])
        # fitness proportional selection
        recombination_probs = fitnesses / np.sum(fitnesses)

        parents1 = np.random.choice(range(self.pop_size),
                                   int(self.recombine_prob*self.pop_size),
                                   p=recombination_probs,
                                   replace=False)
        parents2 = np.random.choice(range(self.pop_size),
                                    int(self.recombine_prob*self.pop_size),
                                    p=recombination_probs,
                                    replace=False)

        sorted_fitnesses = sorted([(i,p) for (i,p) in enumerate(fitnesses)],
                                  key=lambda x: x[1])
        # Replace the worst N individuals with children
        removed = [ i for i,f in sorted_fitnesses ]

        return zip(parents1,parents2),removed[0:min(len(removed),len(parents1))]

    def mutate(self,player):
        player.angle += norm.rvs(loc=0,scale=5)
        player.v0 += cauchy.rvs(loc=0,scale=20)
        player.angle = min(max(1,player.angle),89)
        player.v0 = max(1,player.v0)

    def mutate_generation(self):
        for player in self.populations.values():
            if np.random.uniform() < self.mutation_rate:
                self.mutate(player)

    def recombine(self,parents):
        '''
        Recombination = weighted mean of genes
        (using relative fitness as weights)
        '''

        child = Player()
        weights = np.array([p.fitness for p in parents])
        weights /= weights.sum()

        child.angle = np.sum([p.angle*wi
                              for p,wi in zip(parents,weights)])
        child.v0 = np.sum([p.v0*wi
                           for p,wi in zip(parents,weights)])
        return child

    def make_next_generation(self):
        selected,removed = self.select_next_gen()

        children = []
        for ids in selected:
            parents = [self.populations[n] for n in ids]
            children.append(self.recombine(parents))

        # Replace "bad" individuals with children
        for i,child in enumerate(children):
            self.populations[removed[i]] = child

    def cycle(self):
        self.calc_fitnesses()
        self.mutate_generation()
        self.make_next_generation()

    def cycles(self,n_gen,n_plots=6):
        fig,ax = plt.subplots(2,n_plots/2)
        plt.tight_layout()
        i_plot = 0

        for n in range(n_gen):
            print('Generation {}/{}'.format(n,n_gen))
            self.cycle()

            if n % (1+(n_gen/n_plots)) == 0:
                title = 'Generation {}/{}'.format(n,n_gen)
                i,j = (i_plot/ax.shape[1],
                       i_plot%ax.shape[1])
                self.demo(ax=ax[i,j],title=title)
                i_plot +=1
        return self

    def demo(self,ax=None,title=None):
        fitnesses = sorted(self.populations.items(),
                           key=lambda x: x[1].fitness,
                           reverse=True)
        best_player = fitnesses[0][1]

        best_player.plot_shoot(target=TARGET,ax=ax,title=title)


    def display_fitness(self):
        fig,ax = plt.subplots()
        for i,v in enumerate(self.fitnesses):
            ax.scatter([i]*len(v),v,s=1,c='b')
        ax.plot(range(len(ev.fitnesses)),list(map(np.mean,ev.fitnesses)),label='Population mean',c='k')
        ax.plot(range(len(ev.fitnesses)),list(map(np.max,ev.fitnesses)),label='Best player',c='g')
        ax.plot(range(len(ev.fitnesses)),list(map(np.min,ev.fitnesses)),label='Worst player',c='r')
        ax.set_xlabel('Generation',fontsize=20)
        ax.set_ylabel('Fitness',fontsize=20)
        plt.title('Evolution of population fitness over generations',
                  fontweight='bold',fontsize=24)
        plt.legend(fontsize=12)
        plt.show()


if __name__ == '__main__':

    ev = Evolution(50)
    ev.cycles(50)

    ev.display_fitness()
