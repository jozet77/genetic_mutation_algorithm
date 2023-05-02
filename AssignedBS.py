# -*- coding: utf-8 -*-
"""
Claculates each UE assigned BS

@author: Jose Matamoros
"""
from math import sqrt
from ReceivedPower import receivedPower

def distance(a,b):
    '''
    Calculate the distance between points a and b
    
    Parameters
    ----------
    a : [float,float]
        A set of points (x,y)

    b : [float,float]
        A set of points (x,y)

    Returns
    -------
    float
        Distance between a and b

    '''
    c = ((a[0]-b[0]) ** 2) + ((a[1]-b[1])**2)
    d = sqrt(c)
    if (d<0.001): d = 0.001
    return(d)

def distanceUAV(a,b):
    '''
    Calculate the distance between a and an Aerial Base Station b
    
    Parameters
    ----------
    a : [float,float]
        A set of points (x,y)

    b : [float,float]
        A set of points (x,y)

    Returns
    -------
    float
        Distance between a and b

    '''
    c = ((a[0]-b[0]) ** 2) + ((a[1]-b[1])**2) + ((0 - 0.05)**2)
    return(sqrt(c))

def assignedBS(ue_set,bs_set):
    '''
    Calculate the assigned BS fpr each UE
    
    Parameters
    ----------
    ue_set : float
        UE set object that includes all positions
    bs_set : float
        BS set object that includes all positions
        
    Returns
    -------
    [int,float]
        An oredered vector. Where the first value will be for the first UE, 
        and its value will be the vector position of the BS
        [2,    BS2    [UE1,R_PWR]
        4,     BS4    [UE2,R_PWR]
        3,     BS3    [UE3,R_PWR]
        2]     BS2    [UE4,R_PWR]

    '''
    assigned = []
    for ue in ue_set:
        as_i=0
        Pt = 30
        as_pow=-1000
        for i,bs in enumerate(bs_set):
            temp_power = receivedPower(Pt,distance(ue,bs),20)
            if(temp_power>as_pow):
                as_pow = temp_power
                as_i = i
        assigned.append([as_i,as_pow])
    return(assigned)

def assignedUAV(ue_set,bs_set):
    '''
    Calculate the assigned BS fpr each UAV
    
    Parameters
    ----------
    ue_set : float
        UE set object that includes all positions
    bs_set : float
        UAV BS set object that includes all positions
        
    Returns
    -------
    [int,float]
        An oredered vector. Where the first value will be for the first UE, 
        and its value will be the vector position of the BS
        [2,    BS2    [UE1,R_PWR]
        4,     BS4    [UE2,R_PWR]
        3,     BS3    [UE3,R_PWR]
        2]     BS2    [UE4,R_PWR]

    '''
    assigned = []
    for ue in ue_set:
        as_i=0
        Pt = 20
        as_pow=-1000
        for i,bs in enumerate(bs_set):
            temp_power = receivedPower(Pt,distanceUAV(ue,bs),20)
            if(temp_power>as_pow):
                as_pow = temp_power
                as_i = i
        assigned.append([as_i,as_pow])
    return(assigned)

def assignedUE(ue_set,bs_set):
    '''
    Calculate the assigned UE for each BS
    
    Parameters
    ----------
    ue_set : float
        UE set object that includes all positions
    bs_set : float
        BS set object that includes all positions
        
    Returns
    -------
    [int,float]
        An oredered matrix, where the position represents he BS, and within there will be a list that represents all UE assigned to it.
        
        [[1,8,32]            BS1    [UE1,R_PWR],[UE8,R_PWR],[UE32,R_PWR]
        [2,5,10,13,14],      BS2    [UE2,R_PWR],[UE5,R_PWR],[UE10,R_PWR],[UE13,R_PWR],[UE14,R_PWR]
        [3,4,16,21]]         BS3    [UE3,R_PWR],[UE4,R_PWR],[UE16,R_PWR],[UE21,R_PWR]

    '''
    assigned = [[] for i in range(0,len(bs_set))]
    for u,ue in enumerate(ue_set):
        as_i=0
        as_pow=-1000
        Pt = 30
        for i,bs in enumerate(bs_set):
            temp_power = receivedPower(Pt,distanceUAV(ue,bs),20)
            if(temp_power>as_pow):
                as_pow = temp_power
                as_i = i
        assigned[as_i].append([u,as_pow])
    return(assigned)

if __name__ == "__main__":
	##Code below used solely for testing
    ue_set = list(zip([x for x in ue1.xue], [y for y in ue1.yue]))
    bs_set = list(zip([x for x in bs1.xbs], [y for y in bs1.ybs]))
    UE_ass = assignedUE(ue_set,bs_set)