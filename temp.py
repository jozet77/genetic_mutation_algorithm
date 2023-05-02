# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
for n in range(0,len(exp1.unserviced)):
    k=centroids([x[0] for x in exp1.unserviced[n]],[x[1] for x in exp1.unserviced[n]])
    fg1 = plt.plot([x[0] for x in exp1.unserviced[n]],[x[1] for x in exp1.unserviced[n]],'r.',markersize=2)
    fg1 = plt.plot([x[0] for x in k],[x[1] for x in k],'g*',markersize=12)
    tot_l = len(bs1.xbs)
    BStempX = bs1.xbs[:round(tot_l*0.1*(n+1))]
    BStempY = bs1.ybs[:round(tot_l*0.1*(n+1))]
    fg1 = plt.plot(BStempX,BStempY,'b^',markersize=8)
    plt.show(fg1)
    
'''

'''



'''
from functools import reduce
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from Fitness import *

#plt.hist([len(x['best_specimen']['specimen'].chromosomes) for x in exp_pp.results])


#plt.hist([[len(x['best_specimen']['specimen'].chromosomes) for x in exp_pp.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_pn.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_pu.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_np.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_nn.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_nu.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_up.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_un.results],
#           [len(x['best_specimen']['specimen'].chromosomes) for x in exp_uu.results],
#           ], label=['UBPP', 'UBPN', 'UBPU','UBNP', 'UBNN', 'UBNU','UBUP', 'UBUN', 'UBUU'])
#plt.xticks(np.arange(5), ('1', '2', '2', '3', '4'))
#plt.autoscale(tight=True)
#plt.grid('False')
#plt.legend(loc='upper right')
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xlabel('NUMBER OF PATHS SELECTED FROM THE BEST SPECIMEN', fontsize='18')
#plt.ylabel('NUMBER OF EXPERIMENTS', fontsize='18')
#plt.title('NUMBER OF DETERMINED PATHS FOR ALL EXPERIMENTS FOR ALL DISTRIBUIONS FOR UE AND BS', fontsize='18')

#-------------------------------------------------

#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pp.results]))]) for k in range(10,100,10)], sigma=1),'-*b',markersize=12, label='UBPP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pn.results]))]) for k in range(10,100,10)], sigma=1),'-og',markersize=12, label='UBPN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pu.results]))]) for k in range(10,100,10)], sigma=1),'-^r',markersize=12, label='UBPU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_np.results]))]) for k in range(10,100,10)], sigma=1),'-vc',markersize=12, label='UBNP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_nn.results]))]) for k in range(10,100,10)], sigma=1),'-Dm',markersize=12, label='UBNN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_nu.results]))]) for k in range(10,100,10)], sigma=1),'-sy',markersize=12, label='UBNU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_up.results]))]) for k in range(10,100,10)], sigma=1),'-Pk',markersize=12, label='UBUP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_un.results]))]) for k in range(10,100,10)], sigma=1),'-pb',markersize=12, label='UBUN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_score'] for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_uu.results]))]) for k in range(10,100,10)], sigma=1),'-Xg',markersize=12, label='UBUU')
#
#plt.legend()
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xlabel('BS loss [%]', fontsize='18')
#plt.ylabel('MAXIMUM SCORE OVER 1', fontsize='18')
#plt.title('MAXIMUM SCORE PER BS LOSS PERCENTAGE EXPERIMENTS WITH POISSON DISTRIBUION FOR UE AND BS', fontsize='18')

#-------------------------------------------------

#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_pp.results]))]) for k in range(10,100,10)], sigma=1),'-*b',markersize=12, label='UBPP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_pn.results]))]) for k in range(10,100,10)], sigma=1),'-og',markersize=12, label='UBPN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_pu.results]))]) for k in range(10,100,10)], sigma=1),'-^r',markersize=12, label='UBPU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_np.results]))]) for k in range(10,100,10)], sigma=1),'-vc',markersize=12, label='UBNP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_nn.results]))]) for k in range(10,100,10)], sigma=1),'-Dm',markersize=12, label='UBNN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_nu.results]))]) for k in range(10,100,10)], sigma=1),'-sy',markersize=12, label='UBNU')
#    
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_up.results]))]) for k in range(10,100,10)], sigma=1),'-Pk',markersize=12, label='UBUP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_un.results]))]) for k in range(10,100,10)], sigma=1),'-pb',markersize=12, label='UBUN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([len(y['candidates']) for y in list(filter(lambda x: x['loss']==k,
#         [x for x in exp_uu.results]))]) for k in range(10,100,10)], sigma=1),'-Xg',markersize=12, label='UBUU')
#    
#plt.legend()
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xlabel('BS loss [%]', fontsize='18')
#plt.grid('False')
#plt.ylabel('NUMBER OF CANDIDATES FOR BEST SPECIMEN', fontsize='18')
#plt.title('NUMBER OF CANDIDATES NEEDED TO FIND THE BEST SPECIMEN FOR EXPERIMENTS WITH POISSON DISTRIBUION FOR UE AND BS', fontsize='18')

#-------------------------------------------------

#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pp.results]))]) for k in range(10,100,10)], sigma=1),'-*b',markersize=12, label='UBPP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pn.results]))]) for k in range(10,100,10)], sigma=1),'-og',markersize=12, label='UBPN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pu.results]))]) for k in range(10,100,10)], sigma=1),'-^r',markersize=12, label='UBPU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_np.results]))]) for k in range(10,100,10)], sigma=1),'-vc',markersize=12, label='UBNP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_nn.results]))]) for k in range(10,100,10)], sigma=1),'-Dm',markersize=12, label='UBNN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_nu.results]))]) for k in range(10,100,10)], sigma=1),'-sy',markersize=12, label='UBNU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_up.results]))]) for k in range(10,100,10)], sigma=1),'-Pk',markersize=12, label='UBUP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_un.results]))]) for k in range(10,100,10)], sigma=1),'-pb',markersize=12, label='UBUN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([max([y['best_specimen']['specimen'].born for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_uu.results]))]) for k in range(10,100,10)], sigma=1),'-Xg',markersize=12, label='UBUU')
#
#plt.legend()
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xlabel('BS loss [%]', fontsize='18')
#plt.ylabel('Generations', fontsize='18')
#plt.title('MAXIMUM NUMBER OF GENERATIONS NEEDED TO FIND BEST SPECIMEN WITH POISSON DISTRIBUION FOR UE AND BS', fontsize='18')

#-------------------------------------------------
#
#vor = Voronoi(list(zip([x for x in bsn.xbs], [y for y in bsn.ybs])))
#fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='black',line_width=0.75, line_alpha=0.75, point_size=1)
#plt.plot(uen.xue,uen.yue,'r.',label='UE')
#plt.plot(bsn.xbs, bsn.ybs,'b^', label='BS',markersize=7)
#plt.xlabel('Km', fontsize='18')
#plt.ylabel('Km', fontsize='18')
#plt.legend()
#plt.legend(loc='upper right')
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.title('Point placement and Voronoi diagram for UBNN', fontsize='18')
#plt.show()
#
#-------------------------------------------------
#tot_l = len(exp_pp.BSset.xbs)
#plt.plot(uep.xue, uep.yue, 'gP', label='With Service')
#plt.plot([x[0] for x in exp_pp.unserviced[0]],[x[1] for x in exp_pp.unserviced[0]],'ro', label='No Service')
###plt.plot(bsn.xbs[:round(tot_l*0.1)], bsn.ybs[:round(tot_l*0.1)], 'b^')
#plt.xlabel('Km', fontsize='18')
#plt.ylabel('Km', fontsize='18')
#plt.legend(loc='upper right')
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.title('UBPP with 80% BS loss', fontsize='18')
#plt.show()

#-------------------------------------------------

#element = 1
#graph_path(exp_pp.results[element]['candidates'],exp_pp.unserviced[element],exp_pp.unserviced_centroids[element], exp_pp.results[element]['loss'], exp_pp.results[element]['allowed_drones'])
   
#exp_pu.showResults()

#-------------------------------------------------
#col = ['b','g','r','c','m','y','k','b','g']
#ha = ('/','o','+','','v','D','p','P','//')
#plt.hist([x[1] for x in exp_pp.initial_capacity],hatch='/', label='UBPP')

#plt.hist([[x[1] for x in exp_pp.initial_capacity],
#          [x[1] for x in exp_pn.initial_capacity],
#          [x[1] for x in exp_pu.initial_capacity],
#          [x[1] for x in exp_np.initial_capacity],
#          [x[1] for x in exp_nn.initial_capacity],
#          [x[1] for x in exp_nu.initial_capacity],
#          [x[1] for x in exp_up.initial_capacity],
#          [x[1] for x in exp_un.initial_capacity],
#          [x[1] for x in exp_uu.initial_capacity],
#          ], label=['UBPP', 'UBPN', 'UBPU','UBNP', 'UBNN', 'UBNU','UBUP', 'UBUN', 'UBUU'])
#
#plt.autoscale(tight=True)
#plt.legend(loc='upper left')
#plt.grid(color='k', linestyle='-', linewidth=0.5)
#plt.xlabel('Service', fontsize='18')
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xticks([0,1],['No Service','With Service'], fontsize='18')
#plt.ylabel('Ammount of UE', fontsize='18')
#plt.title('Number of UE that have service in Initial Conditions', fontsize='18')

#-------------------------------------------------

fig1 = plt.figure(1)

col = ['-*b','-og','-^r','-vc','-Dm','-sy','-Pk','-pb','-Xg']
lab = ['UBPP','UBPN','UBPU','UBNP','UBNN','UBNU','UBUP','UBUN','UBUU']

for j,k in enumerate((exp_pp,exp_pn,exp_pu,exp_np,exp_nn,exp_nu,exp_up,exp_un,exp_uu)):
    prt = []
    for cap in k.capacities:
        [a,b,c] = plt.hist([x[1] for x in cap],2)
        prt.append(a[0]*100/(a[0]+a[1]))
    fig2 = plt.figure(2)
    plt.plot([x for x in range(10,100,10)],[100-x for x in prt[::-1]],col[j], label=lab[j])
    fig1 = plt.figure(1)


fig2 = plt.figure(2)
plt.legend()
plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
plt.xlabel('BS loss [%]', fontsize='18')
plt.ylabel('UE serviced [%]', fontsize='18')
plt.title('UE WITH SERVICE AFTER BS LOSS', fontsize='18')
plt.grid(True)
fig2.show()

#-------------------------------------------------

#tot = []
#
#col = ['-*b','-og','-^r','-vc','-Dm','-sy','-Pk','-pb','-Xg']
#lab = ['UBPP','UBPN','UBPU','UBNP','UBNN','UBNU','UBUP','UBUN','UBUU']
#
#for dis in (exp_pp,exp_pn,exp_pu,exp_np,exp_nn,exp_nu,exp_up,exp_un,exp_uu):
#    cov_dis = []
#    for loss in range(10,100,10):
#        cov = []
#        for itt in [x for x in dis.results]:
#            if(itt['loss'] == loss):
#                cov.append(service_ratio(itt['best_specimen']['specimen'].chromosomes,dis.unserviced[int((loss)/10)-1],200000))
#        print('The coverage ratio is %g with a loss of %d'%(min(cov), loss))
#        cov_dis.append(min(cov))
#    plt.plot([i for i in range(10,100,10)],cov_dis)
#    tot.append(cov_dis)
#
#plt.hist(tot, label=['UBPP', 'UBPN', 'UBPU','UBNP', 'UBNN', 'UBNU','UBUP', 'UBUN', 'UBUU'])
#plt.legend()
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xlabel('BS loss [%]', fontsize='18')
#plt.ylabel('UE serviced [%]', fontsize='18')
#plt.title('UE WITH SERVICE AFTER BS LOSS WITH UABS IMPLEMENTATION', fontsize='18')
#plt.grid(True)

#print(service_ratio(itt['best_specimen']['specimen'].chromosomes,exp_pp.unserviced[1],200000))
#-------------------------------------------------

#fig1 = plt.figure(1)
#
#col = ['-*b','-og','-^r','-vc','-Dm','-sy','-Pk','-pb','-Xg']
#lab = ['UBPP','UBPN','UBPU','UBNP','UBNN','UBNU','UBUP','UBUN','UBUU']
#
#for j,k in enumerate((exp_pp,exp_pn,exp_pu,exp_np,exp_nn,exp_nu,exp_up,exp_un,exp_uu)):
#    prt = []
#    for cap in k.capacities:
#        [a,b,c] = plt.hist([x[1] for x in cap],2)
#        prt.append((a[1])*100/(a[0]+a[1]))
#    fig2 = plt.figure(2)
#    plt.plot([x for x in range(10,100,10)],[100-x for x in prt[::-1]],col[j],markersize=9, label=lab[j])
#    fig1 = plt.figure(1)
#
#fig2 = plt.figure(2)
#plt.legend()
#plt.xlabel('BS loss [%]', fontsize='18')
#plt.ylabel('UE serviced after UABS Implementation [%]', fontsize='18')
#plt.title('SERVICE DIFFERENCE FOR UE BETWEEN INITIAL LOSS AND AFTER UABS IMPLEMENTATION', fontsize='18')
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.grid(True)
#fig2.show()

#print(service_ratio(exp_pp.results[5]['best_specimen']['specimen'].chromosomes,exp_pp.unserviced[1],200000))
#print([list(filter(lambda x: x==90,[x['loss'] for x in k.results])) for k in (exp_pp, exp_pn, exp_pu)])    

#-------------------------------------------------

#def avg_distance(chromosome_set):
#    '''
#    The avg distance of all the paths for a given child (set of chromosomes)
#    
#    Parameters
#    ----------
#    chromosome_set : [chromosome]
#        A set of chromosomes
#        
#    Returns
#    -------
#    (float)
#        The sum of all the distances of all paths (chromosomes) divided by the number of chromosomes
#    '''
#    import numpy as np
#    
#    suma=0
#    for path in chromosome_set.copy():
#        aa = [point for point in path][:-1]
#        bb = [point for point in path][1:]
#        for point_a, point_b in zip(aa,bb):
#            suma += distance(point_a,point_b)
#        suma += distance(aa[0],bb[-1])
#    return(suma/len(chromosome_set))
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pp.results]))]) for k in range(10,100,10)], sigma=1),'-*b',markersize=12, label='UBPP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pn.results]))]) for k in range(10,100,10)], sigma=1),'-og',markersize=12, label='UBPN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_pu.results]))]) for k in range(10,100,10)], sigma=1),'-^r',markersize=12, label='UBPU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_np.results]))]) for k in range(10,100,10)], sigma=1),'-vc',markersize=12, label='UBNP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_nn.results]))]) for k in range(10,100,10)], sigma=1),'-Dm',markersize=12, label='UBNN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_nu.results]))]) for k in range(10,100,10)], sigma=1),'-sy',markersize=12, label='UBNU')
#
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_up.results]))]) for k in range(10,100,10)], sigma=1),'-Pk',markersize=12, label='UBUP')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_un.results]))]) for k in range(10,100,10)], sigma=1),'-pb',markersize=12, label='UBUN')
#plt.plot([i for i in range(10,100,10)],gaussian_filter1d([min([avg_distance(y['best_specimen']['specimen'].chromosomes)*60/50 for y in list(filter(lambda x: x['loss']==k,
#          [x for x in exp_uu.results]))]) for k in range(10,100,10)], sigma=1),'-Xg',markersize=12, label='UBUU')    
#
#plt.legend()
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
#plt.xlabel('BS LOSS', fontsize='18')
#plt.ylabel('TIME [minutes]', fontsize='18')
#plt.title('AVERAGE ROUND-TRIP TIME NEEDED PER BS LOSS', fontsize='18')
#
