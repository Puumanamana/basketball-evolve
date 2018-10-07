import numpy as np
from math import cos,sin,pi
import matplotlib.pyplot as plt

np.random.seed(1234)

class Player:

    def __init__(self,angle=None,v0=None):
        self.angle = angle
        self.v0 = v0
        self.fitness = 0
        self.size = 180 # Player height in cm
        self.initialize()

    def initialize(self):
        self.angle = np.random.randint(1,90)
        self.v0 = np.random.randint(1,10)

    def shoot(self):
        def traj(t):
            v0_x,v0_y = self.v0 * np.array([cos(pi*self.angle/180),sin(pi*self.angle/180)])
            x_t = v0_x * t
            y_t = -(0.5 * 9.8 * t**2) + (v0_y * t) + self.size
            if type(t) in [float,np.float64]:
                y_t = max(0,y_t)
            else:
                y_t = y_t*map(int,y_t>0)
            return x_t,y_t
        return traj

    def evaluate(self,target,plot=False):
        traj = self.shoot()

        v0_x,v0_y = self.v0 * np.array([cos(pi*self.angle/180),sin(pi*self.angle/180)])

        t_f = target[0] / v0_x
        points = traj(np.arange(t_f-2,t_f+2,0.1))

        dists = [np.sum((np.array(M)-target)**2) for M in zip(*points)]

        x_f,y_f = traj(t_f)
        derivative = (v0_y/v0_x) - (9.8/v0_x/v0_x)*x_f

        if plot:
            self.plot_shoot(target)

        self.fitness = 1/(0.001+np.min(dists))

        if derivative > 0 or self.fitness == 0:
            self.fitness = 1e-5

    def plot_shoot(self,target=None,ax=None,title=None):
        if ax is None:
            fig,ax = plt.figure()

        if target is not None:
            ax.scatter(*target,c='r',marker='^',s=50)
            t = np.arange(0,int(target[0]/self.v0/cos(pi*self.angle/180))+2,0.1)
        else:
            t = range(20)
        traj = self.shoot()
        pos = [ traj(ti) for ti in t ]

        ax.scatter(*zip(*pos),s=5)
        if title is None:
            title = 'Fitness: {}'.format(self.fitness)
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Height (cm)')
        ax.set_title(title,fontweight='bold')
