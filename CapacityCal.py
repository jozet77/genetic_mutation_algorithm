# -*- coding: utf-8 -*-
"""
Calculates the capacity (bps) of a UE based on SNR

@author: Jose Matamoros
"""

from ReceivedPower import receivedPower
from AssignedBS import distance
from math import log
from math import sqrt

def db_to_w(power_db):
    return(10**(power_db/10))

def capacity(ue_set, bs_set, assigned_bs_list, capacity_threshold):
    '''
    Calculate each UE capacity based on the Rx from all BS
    
    Parameters
    ----------
    ue_set : float[]
        UE set object that includes all positions
    bs_set : float
        BS set object that includes all positions
    assigned_bs_list : float[]
        BS index and power assigne in order of each UE
    capacity_threshold : float
        The capacity threshold in bps
    
    Returns
    -------
    [float,int]
        a 2xn matrix with [0] as the capacity in bps, and [1] as a BOOLEAN whether is within the threshold
    '''
    nloss = -102
    bw=1.4e6
    ue_capacity = []
    for ue,a_bs in zip(ue_set,assigned_bs_list):
        noise = 0
        for i,bs in enumerate(bs_set):
            if(i!=a_bs[0]):
                #with open('test.txt', 'a') as of: of.write("poins: " + str(ue) +" - "+ str(bs)+'\n')
                try:
                    noise += db_to_w(receivedPower(30,distance(ue,bs),20))**2
                except:
                    print('points: ',ue,bs)
#        noise = noise - nloss
        snr = abs(db_to_w(a_bs[1])/sqrt(noise/(len(bs_set)-1)))
        c= bw*log(1+snr,10)
        if ((c>= capacity_threshold)and(a_bs[1] > -102)):
            ue_capacity.append([c,1])
        else:
            ue_capacity.append([c,0])
    return(ue_capacity)

def capacityBS_from_UE(ue_set, bs_set, assigned_ue_list, capacity_threshold):
    ##Currently unsed !
    '''
    Calculate each UE capacity based on the Rx from all BS
    
    Parameters
    ----------
    ue_set : float[]
        UE set object that includes all positions
    bs_set : float
        BS set object that includes all positions
    assigned_ue_list : float[]
        BS index and power assigne in order wit evey UE that it services
    capacity_threshold : float
        The capacity threshold in bps
    
    Returns
    -------/
    [float,int]
        a 2xn matrix with [0] as the capacity in bps for each user, and [1] as a BOOLEAN whether is within the threshold
        
    '''
    bw=1.4e6
    bs_to_ue_capacity = [[] for i in range(0,len(ue_set))]
    for bs,bs_pos in zip(assigned_ue_list,bs_set):
        for i,ue in enumerate(bs):
            if(len(bs)!=1):
                noise=0
            else:
                noise=1
            for j,ue_extra in enumerate(bs):
                if(i!=j):
                   noise += db_to_w(receivedPower(-10,distance(ue_extra,bs_pos),20))**2
            if(len(bs)!=1):
                snr = abs(db_to_w(ue[1])/sqrt(noise/(len(bs)-1)))
            else:
                noise = -112
                snr = abs(db_to_w(ue[1])/noise)
            c= bw*log(1+snr,10)
            if ((c>= capacity_threshold)and(ue[1] > -102)):
                bs_to_ue_capacity[ue[0]] = [c,1]
            else:
                bs_to_ue_capacity[ue[0]] = [c,0]
    return(bs_to_ue_capacity)


if __name__ == "__main__":
	##Code below used solely for testing
    from AssignedBS import assignedBS
    from AssignedBS import assignedUE
    #capacity(zip([x for x in ue1.xue], [y for y in ue1.yue]),zip([x for x in bs1.xbs], [y for y in bs1.ybs]),assi,500e3)
    ue_set = list(zip([x for x in ue1.xue], [y for y in ue1.yue]))
    bs_set = list(zip([x for x in bs1.xbs], [y for y in bs1.ybs]))
    #bs_set = bs_set[:round(len(bs_set)*0.1)]
    #plt.plot([x[0] for x in ue_set],[x[1] for x in ue_set],'r.',[x[0] for x in bs_set],[x[1] for x in bs_set],'b^')
    assigned_bs_list=assignedBS(ue_set,bs_set)
    assigned_ue_list=assignedUE(ue_set,bs_set)
    bw=1.4e6
    capacity_threshold = 200000
    ue_capacity = []
    for ue,a_bs in zip(ue_set,assigned_bs_list):
        noise = 0
        for i,bs in enumerate(bs_set):
            if(i!=a_bs[0]):
                noise += db_to_w(receivedPower(30,distance(ue,bs),20))**2
        snr = abs(db_to_w(a_bs[1])/sqrt(noise/(len(bs_set)-1)))
        c= bw*log(1+snr,10)
        if ((c>= capacity_threshold)and(a_bs[1] > -102)):
            ue_capacity.append([c,1])
        else:
            ue_capacity.append([c,0])
            