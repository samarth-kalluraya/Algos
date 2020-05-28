#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:08:37 2020

@author: samarth
"""
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import patches as ptc



class Node:
    def __init__(self, location, cost=0, parent=None):
        self.location = location
        self.cost = cost
        self.parent = parent

    def set_parent(self, parent):
        self.parent = parent

    def update_cost(self, cost):
        self.cost = cost

class RRTstar:
    def __init__(self, start, goal, limits, obstacle_limits, params):
        self.start=start
        self.goal=goal
        self.dims = len(self.start)
        self.limits = limits
        self.obstacle_limits=obstacle_limits
        
        self.steer_dist=params[0]
        self.max_iterations=params[1]
        self.bubble=params[2]
        self.plot_cost_graph=params[3]
        self.plot_graph_build=params[4]
        
        self.root=Node(start,cost=0)
        self.last_node=None
        
        self.node_list=[self.root]
        self.node_coords=np.array([self.start])
        
        self.path=[]
        self.path_cost_array=[]
        
        
    def sample(self):
        location=[]
        for i in range(self.dims):
            r=random.random()
            scale=self.limits[i][1]-self.limits[i][0]
            location.append(r*scale+self.limits[i][0])
        return location
        
    def get_dist(self,loc1,loc2):
        vec=np.array(loc1)-np.array(loc2)
        return np.sqrt(np.dot(vec,vec))
        
    def steer(self,start,end):
        dist=self.get_dist(start,end)
        if dist<self.steer_dist:
            return end
        else:
            unit_vec=(np.array(end)-np.array(start))/dist
            new=unit_vec*self.steer_dist+np.array(start)
            return tuple(new)
    
    def is_in_collision(self,loc):
        for i in range(len(self.obstacle_limits)):
            check=0
            for j in range(len(self.obstacle_limits[i])):
                if(self.obstacle_limits[i][j][0] <= loc[j] <= self.obstacle_limits[i][j][1]):
                    check=check+1
            if check==len(self.obstacle_limits[0]):
                return True
        return False
    
    def is_path_collision(self,loc1,loc2):
        divisions=11
        x=np.linspace(loc1[0],loc2[0],divisions)
        y=np.linspace(loc1[1],loc2[1],divisions)
        for i in range(1,divisions):
            check_coll=self.is_in_collision(tuple((x[i],y[i])))
            if(check_coll):
                return True
        return False
    
    def nearest(self,loc):
        min_dist=np.inf
        near_node=None
        for node in self.node_list:
            dist=self.get_dist(loc,node.location)
            if dist<min_dist:
                min_dist=dist
                near_node=node
        return near_node, min_dist
    
    def nearest_vectorized(self,loc):
        a=self.node_coords.copy()
        b=np.array([loc])
        diff=a-b
        dist_array=np.sqrt(np.sum(diff**2, axis=1))
        arg=np.argmin(dist_array)
        return self.node_list[arg],dist_array[arg]
            
    
    def nearest_nodes(self,loc,bubble):
        nearest_node_list=[]
        distances=[]
        for node in self.node_list:
            dist=self.get_dist(loc,node.location)
            if dist<bubble:
                nearest_node_list.append(node)
                distances.append(dist)
        return nearest_node_list, distances
    
    def nearest_nodes_vectorized(self,loc,bubble):
        nearest_node_list=[]
        distances=[]
        a=self.node_coords.copy()
        b=np.array([loc])
        diff=a-b
        dist_array=np.sqrt(np.sum(diff**2, axis=1))
        arg = np.where(dist_array<bubble)[0]
        for i in arg:
            nearest_node_list.append(self.node_list[i])
            distances.append(dist_array[i])
        return nearest_node_list, distances
    
    def connect(self,new_node, nearest_node_list, distances):
        for i in range(len(nearest_node_list)):
            if not self.is_path_collision(new_node.location, nearest_node_list[i].location):
                new_cost=nearest_node_list[i].cost+distances[i]
                if new_node.cost > new_cost:
                    new_node.cost=new_cost
                    new_node.parent=nearest_node_list[i]
    
    def rewire(self,new_node, nearest_node_list, distances):
        for i in range(len(nearest_node_list)):
            if not self.is_path_collision(new_node.location, nearest_node_list[i].location):
                new_cost=new_node.cost+distances[i]
                if nearest_node_list[i].cost > new_cost:
                    nearest_node_list[i].cost=new_cost
                    nearest_node_list[i].parent=new_node
        
    
    def build_path(self):
        found_path=False
        iterations=0
        while iterations<self.max_iterations:
            iterations += 1


            if random.random()<0.1 and not found_path:
                x=self.goal
            else:
                while True:
                    x=self.sample()
                    if not self.is_in_collision(x):
                        break
            near_node, near_d=self.nearest_vectorized(x)
            x_new=self.steer(near_node.location,x)
            if self.is_path_collision(near_node.location, x_new):
                continue
            new_node=Node(x_new,cost=near_node.cost+near_d,parent=near_node)
            
            # self.bubble = 2                                                      #edit bubble formula      
            nearest_node_list, distances=self.nearest_nodes_vectorized(x_new, self.bubble)
            self.connect(new_node, nearest_node_list, distances)
            if iterations%100==0 and found_path:
                self.get_path(True)
                self.path_cost_array.append((self.last_node.cost,iterations))
                # self.path_cost_array.append((self.get_path_cost(),iterations))
                if iterations%300==0 and plot_graph_build:
                    self.draw_graph(iterations,True)
                print(iterations)
            self.node_list.append(new_node)
            self.node_coords=np.append(self.node_coords,[new_node.location],axis=0)
            
            self.rewire(new_node, nearest_node_list, distances) 
            
            if  new_node.location==self.goal:
                found_path=True
                print("found path::")
                print(iterations)
                self.last_node=new_node
        
        return found_path
    
    def get_path(self,found_path):
        if not found_path:
            self.last_node=self.nearest_vectorized(self.goal)
        self.path=[]
        current_node=self.last_node
        while current_node is not None:
            self.path.append(current_node)
            current_node=current_node.parent
        return self.path
    
    def get_path_cost(self):
        path_cost=0
        for n in self.path:
            c=n.location
            if n.parent is not None:
                p = n.parent.location
                path_cost+=self.get_dist(c,p)
        return path_cost
    
    def draw_graph(self,count,is_path=False):
        p=plt.figure(1)
        ax=p.add_subplot(111)
        ax.cla()
        ax.plot(self.start[0], self.start[1], 'xr')
        print("plot")
        for n in self.node_list:
            c = n.location
            if n.parent is not None:
                p = n.parent.location
                ax.plot([c[0], p[0]], [c[1], p[1]], 'b', linewidth=0.2)
        if is_path:
            for n in self.path:
                c = n.location
                if n.parent is not None:
                    p = n.parent.location
                    ax.plot([c[0], p[0]], [c[1], p[1]], 'r')
        rect1=plt.Rectangle((4,5),6,1,color='k')
        rect2=plt.Rectangle((14,6),1,13,color='k')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        if is_path:
            plt.savefig('frames/'+str(count)+'final.png', dpi=300)
        plt.pause(1)
                
    
                
def pad_obstacle(obstacle_limits, limits, padding, pad_limits=False):
    for i in range(len(obstacle_limits)):
        for j in range(len(obstacle_limits[i])):
            obstacle_limits[i][j][0]=obstacle_limits[i][j][0]-padding
            obstacle_limits[i][j][1]=obstacle_limits[i][j][1]+padding
    if pad_limits:
        for i in range(len(limits)):
            limits[i][0]=limits[i][0]+padding
            limits[i][1]=limits[i][1]-padding
    

def plot_cost():
    a=np.array(obj.path_cost_array)
    p = plt.figure(2)
    ax = p.add_subplot(111)
    ax.plot(a[:,1],a[:,0], label="cost") 
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.title('Cost vs Iteration')
    plt.legend(loc="upper right") 
    plt.show()
    p.savefig('cost_plot', format='svg', dpi=1200)
   
"""PARAMS"""
steer_distance_threshold = 1
obstacle_padding=0.6
rewire_bubble=2
max_iterations=10000
plot_cost_graph=True
plot_graph_build=True

params=[steer_distance_threshold, max_iterations, rewire_bubble, plot_cost_graph, plot_graph_build]
"""end PARAMS"""

    
if __name__=="__main__":
    start=(10,10)
    goal=(18,18)        
    limits=[[0,20],[0,20]]      #[[x1,x2],[y1,y2],[z1,z2]...]
    obstacle_limits=[[[4,10],[5,6]],[[14,15],[6,19]]]      #[[[x1,x2],[y1,y2],[z1,z2]...], [[x1,x2],[y1,y2],[z1,z2]...]]
    pad_obstacle(obstacle_limits, limits, obstacle_padding, pad_limits=True)
    
    obj = RRTstar(start,goal,limits,obstacle_limits,params)
    found_path = obj.build_path()
    path = obj.get_path(found_path)
    obj.draw_graph(0,is_path=True)
    if plot_cost_graph:
        plot_cost()

    