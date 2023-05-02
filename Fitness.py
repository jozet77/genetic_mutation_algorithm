# -*- coding: utf-8 -*-
"""
Fitness Functions

@author: Jose Matamoros
"""
from AssignedBS import distance

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def line_cross(chromosome_set):
    '''
    The ratio of intersected paths the specimen has
    
    Parameters
    ----------
    chromosome_set : [chromosome]
        A set of chromosomes
        
    Returns
    -------
    (int)
        the number of intersections between all paths within the specimen
    '''
    
    paths = []
    for path in chromosome_set[:]:
        tmp_lines = path[:]
        tmp_lines.append(path[0])
        paths.append(tmp_lines)                
    
    segments=[]
    for path in paths:
        aa = [point for point in path][:-1]
        bb = [point for point in path][1:]
        tmp_segments = []
        for point_a,point_b in zip(aa,bb):
            tmp_segments.append([point_a,point_b])
        segments.append(tmp_segments)
    
    intersections = 0
    for i,lines in enumerate(segments):
        for j,lines2 in enumerate(segments):
            if(j>i):
                for line in lines:
                    for line2 in lines2:
                        if(intersect(line[0],line[1],line2[0],line2[1])):
                            intersections += 1
    return(intersections)

def total_distance(chromosome_set):
    '''
    The total distance of all the paths for a given child (set of chromosomes)
    
    Parameters
    ----------
    chromosome_set : [chromosome]
        A set of chromosomes
        
    Returns
    -------
    (float)
        The sum of all the distances of all paths (chromosomes)
    '''
    import numpy as np
    
    suma=0
    for path in chromosome_set.copy():
        aa = [point for point in path][:-1]
        bb = [point for point in path][1:]
        for point_a, point_b in zip(aa,bb):
            suma += distance(point_a,point_b)
        suma += distance(aa[0],bb[-1])
    return(suma)

def angle(a,b,c):
    '''
    Returns the angle ABC
    
    Parameters
    ----------
    a,b,c : [point,poin,point]
        Point (X,Y)
        
    Returns
    -------
    (float)
        the angle ABC
    '''
    from math import acos
    
    ab = distance(a,b)
    bc = distance(b,c)
    ca = distance(c,a)
    return( acos((ab**2 + bc**2 - ca**2)/(2*ab*bc)) ) 

def angle_ratio(chromosome_set):
    '''
    Returns a ratio of how much does the shape is similar to a perfect polygon
    with the ame number of angles
    
    Parameters
    ----------
    chromosome_set : [chromosome]
        The current iteration of the experiment
        
    Returns
    -------
    (float)
        The total calculated ratio of angles from all individual angles
        compared to an angle from a perfect polygon of n vertex
    '''
    from math import pi
    from functools import reduce
    
    ratio = []
    for path in chromosome_set.copy():
        points = [point for point in path]
        n = len(points)
        perfect_angle = (n-2)*pi/n
        points.insert(0,points[-1])
        points.append(points[0])
        suma = 0
        for i in range(1,n):
            tmp_angle = angle(points[i-1],points[i],points[i+1])
            if(tmp_angle>pi): tmp_angle = pi - tmp_angle
            suma += abs((tmp_angle - pi)/pi)
        ratio.append(1 - suma/n)
    return (reduce((lambda x,y: x+y), ratio)/len(chromosome_set))

def service_ratio(chromosome_set, unserviced_set, threshold):
    '''
    Returns a ratio which provides an idea of how users are serviced.
    In addition, it modifies the unserviced_set list provided by name bonding
    
    Parameters
    ----------
    chromosome_set : [chromosome]
        The current iteration of the experiment
    unserviced_set: [points]
        A set of unserviced UE
    threshold: (float)
        The usage threshold from the experiment
        
    Returns
    -------
    (float)
        A ratio of temporarly serviced UE with the solution
    '''
    from AssignedBS import distance
    from CapacityCal import capacity
    from AssignedBS import assignedUAV
    
    unserviced_num = len(unserviced_set)
    
    diameter = 3
    points = []
    for path_i in chromosome_set:
        ##Calculate new points in path with the given radius
        path = path_i.copy()
        n=len(path)
        path.append(path[0])
        for i in range(0,n): #each line
            magnitude = distance(path[i],path[i+1])
            points.append(path[i])
            div = round(magnitude/diameter - 0.5)
            if(div > 3):
                for divisor in range(1,round(magnitude - 0.5),diameter):
                    t = divisor/magnitude
                    xt,yt = [(1-t)*path[i][0]+t*path[i+1][0],(1-t)*path[i][1]+t*path[i+1][1]]
                    points.append([xt,yt])
            points.append(path[-1])
            
            ##Calculate if UE is being serviced during the path
            capacities = []
            capacities = capacity(unserviced_set,points,assignedUAV(unserviced_set,points),threshold)
            for i,cap in enumerate(capacities):
                if(cap[1]>0):
                    unserviced_set.pop(i)
            
    ##HERE YOU MUST CALCULATE HOW MANY WERE LEFT WITHOUT SERVICE
    return(len(unserviced_set)/unserviced_num)
    
if __name__ == "__main__":
	##Code below used solely for testing
    next
    #plt.plot([x[0] for x in points],[x[1] for x in points],'r.')
    #fig1.show()
    dd = uv_list[0][1].chromosomes
    paths = []
    for path in dd[:]:
        tmp_lines = path[:]
        tmp_lines.append(path[0])
        paths.append(tmp_lines)                
    
    segments=[]
    for path in paths:
        aa = [point for point in path][:-1]
        bb = [point for point in path][1:]
        tmp_segments = []
        for point_a,point_b in zip(aa,bb):
            tmp_segments.append([point_a,point_b])
        segments.append(tmp_segments)
    
    intersections = 0
    for i,lines in enumerate(segments):
        for j,lines2 in enumerate(segments):
            if(j>i):
                for line in lines:
                    for line2 in lines2:
                        if(intersect(line[0],line[1],line2[0],line2[1])):
                            intersections += 1