# -*- coding: utf-8 -*-
"""
Functions for child mutators

@author: Jose Matamoros
"""
import numpy as np

def initial_build(centroids,drone_limit):
    '''
    Generates a vecotr of 3 Vertice, with a graph 3+-2.
    A Graph(V,E) will now be called chromosome
    
    Parameters
    ----------
    centroids : [float,float]
        A position vector of centroids
        
    Returns
    -------
    [[int]]
        A matriz, of 1xn, where each n element is a chromosome 
    '''
    from sklearn.utils import shuffle
    centroids = shuffle(centroids, random_state=0)
    div = round(len(centroids)/drone_limit + 0.0001 - 0.5)
    if(div<3):
        div = 3
    n=round(len(centroids)/div + 0.0001 - 0.5)
    chromosome = [[] for x in range(0,n)]
    loop = 0
    for point in centroids:
        chromosome[loop].append(point)
        loop+=1
        if(loop>=n):loop=0
    return(chromosome)

def mutator_in_chromo(chromosomes):
    '''
    Generates 3 chromosome sets from a given chromosome through 3 In-Chromosome methods
    
    Parameters
    ----------
    chromosomes : [chromosome]
        A set of chromosomes (Graph(V,E))
    
    Returns
    -------
    3x[chromosomes]
        Where each element will be the provided set minus n selected chromosomes,
        and atached will be n mutatd chromosomes
    '''
    new_chromo=[]
    
    #First In-Chromosome mutator
    #Swap a section of the chromosome
    tmp_chromosome = chromosomes.copy()
    nl = len(tmp_chromosome)
    s = int(round(np.random.uniform(0,nl-1,1)[0])-0.5)
    chromosome = tmp_chromosome[s] 
    n=len(chromosome)
    ini = int(round(np.random.uniform(0,n-3,1)[0]))
    end = int(round(np.random.uniform(ini,n-1,1)[0]))
    chromo1 = chromosome.copy()
    chromo1[ini:end] = chromosome[ini:end][::-1]
    #tmp_chromosome.tolist()
    tmp_chromosome.pop(s)
    tmp_chromosome.append(chromo1)
    new_chromo.append(tmp_chromosome)
    
    #Second In-Chromosome mutator
    #Swaps the positions of two chromosomes
    tmp_chromosome = chromosomes.copy()
    nl = len(tmp_chromosome)
    s = int(round(np.random.uniform(0,nl-1,1)[0])-0.5)
    chromosome = tmp_chromosome[s] 
    n=len(chromosome)
    pos = [int(round(x)) for x in np.random.uniform(0,n-1,2)]
    chromo2 = chromosome.copy()
    chromo2[pos[0]],chromo2[pos[1]] = chromosome[pos[1]],chromosome[pos[0]]
    #tmp_chromosome.tolist()
    tmp_chromosome.pop(s)
    tmp_chromosome.append(chromo2)
    new_chromo.append(tmp_chromosome)
    
    #Third In-Chromosome mutator
    #Inserts a chromosome in a different place
    tmp_chromosome = chromosomes.copy()
    nl = len(tmp_chromosome)
    s = int(round(np.random.uniform(0,nl-1,1)[0])-0.5)
    chromosome = tmp_chromosome[s]
    n=len(chromosome)
    pos = [int(round(x)) for x in np.random.uniform(0,n-1,2)]
    chromo3 = chromosome.copy()
    chromo3.pop(pos[0])
    chromo3.insert(pos[1],chromosome[pos[0]])
    #tmp_chromosome.tolist()
    tmp_chromosome.pop(s)
    tmp_chromosome.append(chromo3)
    new_chromo.append(tmp_chromosome)
    
    return([x for x in new_chromo])

def mutator_cross_chromo(chromosomes):
    '''
    Generates 4 new childs from a given chromosome thorugh 3 Cross-Chromosome methods
    
    Parameters
    ----------
    chromosomea : [chromosome]
        A set of chromosomes (Graph(V,E))
    
    Returns
    -------
    kx[chromosomes]
        A vector of size up to 4 new set of chromosomes
    '''
    
    new_chromo=[]
    #First Cross-Chromo mutator
    #Vertex swapping between chromosomes
    
    tmp_chromosome = chromosomes.copy()
    n = len(chromosomes)
    pos = [int(round(np.random.uniform(0,n-1,1)[0])-0.5),int(round(np.random.uniform(0,n-1,1)[0])-0.5)]
    while(pos[0] == pos[1]): pos[1] = int(round(np.random.uniform(0,n-1,1)[0]))
    chromo1, chromo2 = chromosomes[pos[0]],chromosomes[pos[1]]
    n1,n2 = len(chromo1),len(chromo2)
    ini1 = int(round(np.random.uniform(0,n1-2,1)[0]))
    end1 = int(round(np.random.uniform(ini1+1,n1-1,1)[0]))
    ini2 = int(round(np.random.uniform(0,n2-2,1)[0]))
    end2 = int(round(np.random.uniform(ini2+1,n2-1,1)[0]))
    tmp1 = []
    tmp1.extend(chromo1[:ini1])
    tmp2 = []
    tmp2.extend(chromo2[:ini2])
    aux1=chromo1[ini1:end1]
    aux2=chromo2[ini2:end2]
    for x in aux2: tmp1.append(x)
    for x in aux1: tmp2.append(x)
    for x in chromo1[end1:]: tmp1.append(x)
    for x in chromo2[end2:]: tmp2.append(x)
    if((len(tmp1)>2) and (len(tmp2)>2)):
        #tmp_chromosome.tolist()
        tmp_chromosome.pop(pos[0])
        if(pos[0]<pos[1]): 
            pos[1]-=1
        tmp_chromosome.pop(pos[1])
        tmp_chromosome.append(tmp1)
        new_chromo.append(tmp_chromosome)
        tmp_chromosome.append(tmp2)
        new_chromo.append(tmp_chromosome)
    
    '''
    for i,chromo1 in enumerate(chromosomes):
        n1 = len(chromo1)
        for j,chromo2 in enumerate(chromosomes):
            if(j!=i):
                n2 = len(chromo2)
                ini1 = int(round(np.random.uniform(0,n1-1,1)[0]))
                end1 = int(round(np.random.uniform(ini1,n1-1,1)[0]))
                ini2 = int(round(np.random.uniform(0,n2-1,1)[0]))
                end2 = int(round(np.random.uniform(ini2,n2-1,1)[0]))
                tmp1 = []
                tmp1.extend(chromo1[:ini1])
                tmp2 = []
                tmp2.extend(chromo2[:ini2])
                aux1=chromo1[ini1:end1]
                aux2=chromo2[ini2:end2]
                for x in aux2: tmp1.append(x)
                for x in aux1: tmp2.append(x)
                for x in chromo1[end1:]: tmp1.append(x)
                for x in chromo2[end2:]: tmp2.append(x)
                new_chromo.append(tmp1)
                new_chromo.append(tmp2)
    '''
                
    #Second Cross-Chromo mutator
    #Joines two chromosomes
    tmp_chromosome = chromosomes.copy()
    n = len(chromosomes)
    if(n > 1):
        f = int(round(np.random.uniform(0,n-1,1)[0])-0.5)
        e = f
        while(e == f): e = int(round(np.random.uniform(0,n-1,1)[0])-0.5)
        tmp = tmp_chromosome[f][:]
        for x in tmp_chromosome[e]: tmp.append(x)
        #tmp_chromosome.tolist()
        tmp_chromosome.pop(e)
        tmp_chromosome.pop(f)
        tmp_chromosome.append(tmp)
        new_chromo.append(tmp_chromosome)
        
    #Third Cross-Chromo mutator
    #Separates a chromosome into two chromosomes
    tmp_chromosome = chromosomes.copy()
    s = int(round(np.random.uniform(0,n-1,1)[0])-0.5)
    chromo = chromosomes[s]
    n1 = len(chromo)
    if(n1>=6):
        ini = int(round(np.random.uniform(0,n1-4,1)[0]))
        end = int(round(np.random.uniform(ini,n1-1,1)[0]))
        while(end == ini or end<ini+3): end = int(round(np.random.uniform(ini,n1-1,1)[0]))
        tmp1 = chromo[ini:end]
        tmp2 = chromo[:ini]
        for x in chromo[end:]: tmp2.append(x)
        #tmp_chromosome.tolist()
        if(len(tmp1)>2 and len(tmp2)>2):
            tmp_chromosome.pop(s)
            tmp_chromosome.append(tmp1)
            tmp_chromosome.append(tmp2)
            new_chromo.append(tmp_chromosome)
    
    return(new_chromo)
    
if __name__ == "__main__":
	##Code below used solely for testing
    next
    #chromosomes = m_chromo.copy()
    #new_chromo = []
    
    
