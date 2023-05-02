# -*- coding: utf-8 -*-
"""
Main code where all experiments will be carried
@author: Jose Matamoros
"""
from ClusterCentroid import centroids
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from AssignedBS import assignedBS
from CapacityCal import capacity
import UAV

class Experiment:
    '''
    Each object will be a different experiment, comprised of:
    -A set of UE point sets
    -A set of working BS
    -An genesis parent from which children will be created
    -Multiple Childs
    
    The object will carry the results of the expermients within it's local variables.
    
    '''
    def __init__(self, UEset, BSset, capacity_threshold):
        '''
        Requires a set of MB and UE
        '''
        self.capacity_threshold = capacity_threshold
        self.BSset = BSset
        self.UEset = UEset
        self.capacities = []
        self.unserviced = []
        self.results = []
        # create parent based on given clusters from a set
        # create first generation of childs from parent
        self.initial_assigned_BS = assignedBS([[x,y] for x,y in zip(UEset.xue,UEset.yue)],[[x,y] for x,y in zip(BSset.xbs,BSset.ybs)])
        ue_set = list(zip([x for x in UEset.xue], [y for y in UEset.yue]))
        bs_set = list(zip([x for x in BSset.xbs], [y for y in BSset.ybs]))
        self.initial_capacity = capacity(ue_set, bs_set,self.initial_assigned_BS, capacity_threshold)
    
    def process_loss(self):
        '''
            A process to determine BS loss and all variable sets needed for further experiments.
        '''
        ue_set = list(zip([x for x in self.UEset.xue], [y for y in self.UEset.yue]))
        loss_set = [i*0.1 for i in range(1,10)]
        tot_l = len(self.BSset.xbs)
        for loss in loss_set:
            BStempX = self.BSset.xbs[:round(tot_l*loss)]
            BStempY = self.BSset.ybs[:round(tot_l*loss)]
            bs_set_temp = list(zip([x for x in BStempX], [y for y in BStempY]))
            temp_assignedBS = assignedBS([[x,y] for x,y in zip(self.UEset.xue,self.UEset.yue)],[[x,y] for x,y in zip(BStempX,BStempY)])
            self.capacities.append(capacity(ue_set, bs_set_temp, temp_assignedBS, self.capacity_threshold))
            temp_un=[]
            for user,cap in zip(ue_set, self.capacities[-1]):
                if(cap[1] == 0):
                    temp_un.append(user)
            self.unserviced.append(temp_un)
    
    def process_uav(self, specimen_population):
        '''
        Executes a set of mutation funcions and thorugh fitness functions retreives the best candidates.
        
        Parameters
        ----------
		specimen_population : int
			The number of best secimens that are selected to keep mutating between generations.
        
        Returns
        -------
        
        '''
        self.unserviced_centroids = []
        for loss in range(0,9):
            unserviced_tmp = self.unserviced[loss][:]
            tmp_unserviced_centroids = centroids([x[0] for x in unserviced_tmp],[x[1] for x in unserviced_tmp])
            self.unserviced_centroids.append(tmp_unserviced_centroids)
            self.uav_set = [UAVset(0,0,i,tmp_unserviced_centroids) for i in range(1,6)]
#            self.uav_set = [UAVset(0,0,i,tmp_unserviced_centroids) for i in range(1,2)]
            for n,uav_option in enumerate(self.uav_set):
                best_score = 0
                candidates = []
                fit_counter = 0
                limit = 40
                while(fit_counter<limit):
                    fit_counter += 1
                    uav_option.mutate()
                    uav_option.fit(unserviced_tmp,200000,5, len(tmp_unserviced_centroids), specimen_population)
                    best = uav_option.best_uav
                    if(best[0] > best_score):
                        best_score = best[0]
                        candidates.append({'specimen':best[1],'score':best_score})
                        fit_counter = 0
                print('Calculated drone %d out of %d for loss %d out of %d' %(n+1,len(self.uav_set),loss+1, 9))
                result = {'loss':(9 - loss)*10, 'allowed_drones':(n+1), 'candidates':candidates, 'best_score':best_score, 'best_specimen':candidates[-1]}
                self.results.append(result)
    
    def showResults(self):
        '''
		Displays the initial results for the given experiment.
		
		Parameters
        ----------
        
        Returns
        -------
		
		'''
        #show initial serviced UE
        fig1 = plt.figure(1) 
        plt.hist([x[1] for x in self.initial_capacity])
        plt.xlabel('Service')
        plt.ylabel('Ammount')
        plt.title('Number of UE that have service before Loss')
        fig1.show()
        
        #show % of serviced UE in relation to BS loss
        prt = []
        for cap in self.capacities:
            [a,b,c] = plt.hist([x[1] for x in cap],2)
            prt.append(a[0]*100/(a[0]+a[1]))
        
        fig2 = plt.figure(2)
        plt.plot(([100-x for x in prt[::-1]]),'--r')
        plt.xlabel('BS loss [%]')
        plt.ylabel('UE serviced [%]')
        plt.title('SERVICE TO UE ACCORDING TO NUMBER OF BS')
        plt.grid(True)
        fig2.show()
        
class UserEquipmentSet:
    '''
    Each object will have a set of points where the UE are, and a set of determinated clusters according to those points.
    '''
    def __init__(self,density, distribution,area):
        from math import sqrt
        quantity = density * area
        if(distribution == 'p'):
            numue = np.random.poisson(quantity, 1) # lambda = quantity
        elif(distribution == 'n'):
            numue = np.int32(np.abs(np.round(np.random.normal(quantity, 10, 1))))# mu = quantity, sigma = 10
        elif(distribution == 'u'):
            numue = np.int32(np.abs(np.round(np.random.uniform(1,quantity,1))))
            
        self.xue = np.random.uniform(0,area,numue)
        self.yue = np.random.uniform(0,area,numue)
        self.clusters = centroids(self.xue, self.yue)

class MacroBaseSet:
    '''
    Each object will have a set of points that establishes where will working base stations be stablished within an experiment.
    '''
    def __init__(self,density, distribution, area):
        from math import sqrt
        quantity = density * area
        if(distribution == 'p'):
            numbs = np.random.poisson(quantity, 1) # lambda = quantity
        elif(distribution == 'n'):
            numbs = np.int32(np.abs(np.round(np.random.normal(quantity, 10, 1))))# mu = quantity, sigma = 1
        elif(distribution == 'u'):
            numbs = np.int32(np.abs(np.round(np.random.uniform(1,quantity,1))))
            
        self.xbs = np.random.uniform(0,area,numbs)
        self.ybs = np.random.uniform(0,area,numbs)

class UAVset:
    '''
    Each class must be created for each child. 
    Each child will have mustiple chromosomes within, which will allow it to mutate.
    '''
    def __init__(self, born, current, drone_limit, centroids):
        self.current = 0
        self.drone_number= drone_limit
        self.uavs = [UAV.Child(current,centroids, drone_limit) for x in range(0,50)]
    
    def mutate(self):
        '''
        Generates new generations of childs based based on mutators.
        At the end of the process, the UAV object will have more Childs (combinations)
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        self.current += 1
        tmp=[]
        for uav in self.uavs:
            for n_uav in uav.mutate(self.current): tmp.append(n_uav)
            #for n_uav in UAV.Child(self.current,uav.mutate(self.current)): self.uavs.append(n_uav)
        for uav in tmp: self.uavs.append(UAV.Child(self.current,uav,self.drone_number))
    
#    def getKey(item):
#        return item[0]
    
    def fit(self,unservice_set, threshold, drone_limit, k, sample):
        from operator import itemgetter
        from functools import reduce
        import numpy as np
        '''
        Performs a fitness procedure on the UAV set.
        It will choose the best 50 Children to stay, and eliminate the rest objects.
        
        Parameters
        ----------
        unservice_set : [UE_set]
            The current iteration of the experiment
        threshold : (float)
            The threshold used to measure the fitness of the drones
        drone_limit : (int)
            The MAX number of chromosomes (paths) a child may have.
        k: (int)
            The number of clusters
        Returns
        -------
        
        '''
        
        distances, angle_ratios, intersections_set, services = [],[],[],[]
        distance, angle_ratio, intersections, service= 0,0,0,0
        for i,uav in enumerate(self.uavs):
            #[distance, angle_ratio, intersections, service] = uav.fitness(unservice_set, threshold)
            [distance, angle_ratio, intersections] = uav.fitness(unservice_set, threshold)
            distances.append(distance)
            intersections_set.append(intersections)
            angle_ratios.append(angle_ratio)
            #services.append(service)
            #print('--> Ratio: ',str(i*100/len(self.uavs)),' - ',i,' - ',len(self.uavs))
            #print(distance, angle_ratio, service)
        
        min_intersections = min(intersections_set)
        min_distance = min(distances)
        
        self.uav_list = []
        ###print("\n-->Eliminating extra childs")
        ##ponderate and sort all uav childs by best fit
        
        #for distance_total,angle_ratio, path_intersections,service_ratio,uav in zip(distances,angle_ratios,services, intersections_set,self.uavs): 
        for distance_total,angle_ratio, path_intersections,uav in zip(distances,angle_ratios, intersections_set,self.uavs):     
            if(path_intersections != 0):
                intersection_ratio = min_intersections/path_intersections
            else:
                intersection_ratio = 1
                
            distance_ratio = min_distance/distance_total
            #grade = service_ratio*0.5 + distance_ratio*0.25 + angle_ratio*0.25
            score = distance_ratio*0.3 + angle_ratio*0.4 + intersection_ratio*0.3
            l1=[]
            for path in [x for x in uav.chromosomes]:
                for point in path:
                    l1.append(point)
            tot_k = reduce(lambda x,y: x+y, [len(x) for x in uav.chromosomes])
            #Filter children with more drones than expected
            if((len(uav.chromosomes)>drone_limit) or (tot_k != k) or (len(l1) != len(np.unique(l1,axis=0)))):
                self.uav_list.append([0,uav])
            else:
                self.uav_list.append([score,uav])
        
        #delete not wanted childs [object wise]
        
        self.uav_list = sorted(self.uav_list, key=itemgetter(0), reverse=True)
        for uav in self.uav_list[sample:]:
            del(uav[1])
        
        self.uav_list = list(filter(lambda x: len(x)>1, self.uav_list))
        self.uavs = [uav[1] for uav in self.uav_list]
        self.best_uav = self.uav_list[0]
    
def graph_path(specimen_candidates, unserviced_points, unserviced_centroids, loss, drones):
    '''
    Creates graphs from the evolution of candidates
    
    Parameters
    ----------
    specimen_candidates: {candidates, score}
        A dictionary that contains candidates and their scores
    
    unserviced_points: [point]
        A set of (points) with unserviced UE
    
    loss: (int)
        BS percentage loss for the set
        
    drones: (int)
        Number of drones allowed for he set
    
    Returns
    ----------
    None
        
    '''
    pic_p = 1
    fig_p = 1
    for i,drone in enumerate(specimen_candidates):
        fig1 = plt.figure(fig_p)
        plt.subplot(2,2,pic_p)
        pic_p += 1
        if(pic_p > 4):
            pic_p = 1
            fig_p += 1
        plt.plot([x[0] for x in unserviced_centroids],[x[1] for x in unserviced_centroids],'g*',markersize=10)
        plt.plot([x[0] for x in unserviced_points],[x[1] for x in unserviced_points],'r.',markersize=0.5)
        path = drone['specimen'].chromosomes
        co=['y--','c--','m--','b--','k--','y-','c-','m-','b-','k-']
        for pa in path:
            x = [x[0] for x in pa]
            x.insert(0,x[-1])
            y = [x[1] for x in pa]
            y.insert(0,y[-1])
            plt.plot(x,y,co.pop(),linewidth=1)
        plt.xlabel('Km')
        plt.ylabel('Km')
        plt.title('Specimen %d with a score of %g %s, with a %d %s of BS loss, and %d drone(s) allowed' % (i, drone['score']*100,'%', loss,'%', drones))
    fig1.show()

if __name__ == "__main__":
    #Create set experiment objects
    uep = UserEquipmentSet(100,'p',25)
    uen = UserEquipmentSet(100,'n',25)
    ueu = UserEquipmentSet(100,'u',25)
    bsp = MacroBaseSet(10,'p',25)
    bsn = MacroBaseSet(10,'n',25)
    bsu = MacroBaseSet(10,'u',25)
    
    
    exp_pp = Experiment(uep, bsp, 200000)
    exp_pp.process_loss()
    exp_pp.process_uav(100)
    print('PP')
    
    exp_pn = Experiment(uep, bsn, 200000)
    exp_pn.process_loss()
    exp_pn.process_uav(100)
    print('PN')
    
    exp_pu = Experiment(uep, bsu, 200000)
    exp_pu.process_loss()
    exp_pu.process_uav(100)
    print('PU')
    
    exp_nn = Experiment(uen, bsn, 200000)
    exp_nn.process_loss()
    exp_nn.process_uav(100)
    print('NP')
    
    exp_np = Experiment(uen, bsp, 200000)
    exp_np.process_loss()
    exp_np.process_uav(100)
    print('NN')
    
    exp_nu = Experiment(uen, bsu, 200000)
    exp_nu.process_loss()
    exp_nu.process_uav(100)
    print('NU')
    
    exp_uu = Experiment(ueu, bsu, 200000)
    exp_uu.process_loss()
    exp_uu.process_uav(100)
    print('UP')
    
    exp_up = Experiment(ueu, bsp, 200000)
    exp_up.process_loss()
    exp_up.process_uav(100)
    print('UN')

    exp_un = Experiment(ueu, bsn, 200000)
    exp_un.process_loss()
    exp_un.process_uav(100)
    print('UU')

	##Code below used solely for testing
    #vor = Voronoi(list(zip([x for x in bs1.xbs], [y for y in bs1.ybs])))
    #fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='black',line_width=1, line_alpha=1, point_size=2)
    #plt.plot(ue1.xue,ue1.yue,'r.',bs1.xbs, bs1.ybs,'b^')
    #plt.show()
    
    element = 3
    loss1 = 0
    
#    graph_path(candidates, exp_pp.unserviced[element])
    graph_path(exp_uu.results[element]['candidates'],exp_uu.unserviced[loss1],exp_uu.unserviced_centroids[loss1], exp_uu.results[element]['loss'], exp_uu.results[element]['allowed_drones'])
    print('Ready!')
    #used to test Mutators.py
    #uav = uav_set1.uavs[0]
    #m_chromo = uav.chromosomes
    #chromo = m_chromo[0]
    
    #used for ploting
    #plt.plot([x[0] for x in exp1.unserviced[0]],[x[1] for x in exp1.unserviced[0]],'r.',[x[0] for x in k],[x[1] for x in k],'g*')
    