# -*- coding: utf-8 -*-
"""
Performs UAV positioning and calculations through GA

@author: Jose Matamoros
"""
import numpy as np
import Mutators
import Fitness

class Child:
    '''
    Each class must be created for each child. 
    Each child will have multiple chromosomes within, which will allow it to mutate.
    '''
    def __init__(self, current, centroids, drone_limit):
        '''
        Parameters
        ----------
        current : (int)
            The current iteration of the experiment
            
        centroids : []
            If current == 0 then it must be a set of cluster centroids
            else
            It must be a set of chromosomes
        '''
        self.born = current
        self.current = current
        if(current==0):
            self.chromosomes = Mutators.initial_build(centroids, drone_limit)
        else:
            self.chromosomes = centroids
    
    def mutate(self,current):
        '''
        Mutates and produces new children from the Child's chromosomes.
        
        Parameters
        ----------
        current : (int)
            The current iteration of the experiment
            
        '''
        self.current = current
        new_childs = []
        
        for new_ch in Mutators.mutator_in_chromo(self.chromosomes): new_childs.append(new_ch)
        #for new_ch in Mutators.mutator_in_chromo(self.chromosomes): new_childs.append(new_ch)
        if(len(self.chromosomes)>2):
            #for new_ch in Mutators.mutator_cross_chromo(self.chromosomes): new_childs.append(new_ch)
            #for new_ch in Mutators.mutator_cross_chromo(self.chromosomes): new_childs.append(new_ch)
            for new_ch in Mutators.mutator_cross_chromo(self.chromosomes): new_childs.append(new_ch)
        
        return(new_childs)
    
    def fitness(self,unserviced_set, threshold):
        '''
        Parameters
        ----------
        unserviced_set : [UE]
            A set of UE that are in need of service
        
        threshold : (float)
            The threshold to meassure service
        
        Returns
        -------
        [float,float,float]
            Return he total distance of the specimen's paths, the angle ratio of all paths, and a ratio of how many did it serviced
        
        '''
        
        distance = Fitness.total_distance(self.chromosomes)
        angle_ratio = Fitness.angle_ratio(self.chromosomes)
        intersections = Fitness.line_cross(self.chromosomes)
        #service = Fitness.service_ratio(self.chromosomes, unserviced_set.copy(), threshold)
        
        #return([distance, angle_ratio, intersections, service])
        return([distance, angle_ratio, intersections])

if __name__ == "__main__":
	##Code below used solely for testing
    from ClusterCentroid import centroids
    #plt.plot(chromo[0][0], chromo[0][1], 'k-')
    
    k=centroids([x[0] for x in exp1.unserviced[0]],[x[1] for x in exp1.unserviced[0]])
    child1 = Child(0,k)    
    exp_chromo = initial_build(k, 5)
    chromosomes = exp_chromo[0]