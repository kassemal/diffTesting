"""
For every $epsilon in $P_VALUES, read the related distances then plot their CDF. 
Also plot the max. diatance w.r.t epsilon
"""
import pdb
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
#
from utils import methods

NAME = 'adult' #internet'
D_TAG = 'EMD' #distances to consider: EMD for Earth Mover Distance 
NB_classes = 13  #make sure to use the correct number of classes. Maximum number for adult: 2, internet: 9 
SIZE = 1000    #size of the dataset to consider
NB_distinct = 619   

d_method = 'd_per_coordinate' #'avg_pred' 'd_per_att'
ITERATIONS_NB = 1000  
P_VALUES = [0.1, 0.5, 1.0]   
P_VALUES_L = [0.1, 0.5, 1.0]   
COLOR_LIST = {0.1:'k', 0.5:'r', 1.0:'g'} 

if __name__ == '__main__':

    #initiate distance list
    distances = []
    max_d = []
    #read distances
    for p, epsilon in enumerate(P_VALUES):  
        distances_p = []
        filename = 'results/distances/%s/S%s/DP/%s/N%d/distances_%s_DP_p%s_N%d_%s'%(NAME, str(SIZE), d_method, ITERATIONS_NB, NAME, str(epsilon), ITERATIONS_NB, D_TAG)
        f = open(filename, 'r')   
        for c in range(NB_distinct): #line in f: 
            line = f.readline()
            line = line.strip()
            distances_p.append(float(line))
        f.close()
        distances.append(distances_p)
        max_d.append(max(distances_p))

    #plot max. distances
    max_dfilename = 'results/figures/%s/S%s/DP/%s/N%d/max_d_%s_DP_N%d_%s.pdf'%(NAME, str(SIZE), d_method, ITERATIONS_NB, NAME, ITERATIONS_NB, D_TAG)
    fig = plt.figure()  
    plt.plot(range(len(P_VALUES)), max_d, '-x')
    plt.xticks(range(len(P_VALUES)), P_VALUES_L)
    plt.ylabel(r'$\delta$', fontsize=26)
    plt.xlabel(r'$\epsilon$', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    #fig = plt.gcf()
    methods.file_create(max_dfilename)
    fig.savefig(max_dfilename, bbox_inches='tight', format='pdf')#(max_dfilename, figsize=(8, 6), bbox_inches='tight')
    plt.close(fig)

    #plot CDF of the distances 
    curves, legend = [], []
    fig = plt.figure()  
    curves = [0 for x in range(len(P_VALUES))]
    legend = [r'$\epsilon$ = ' + str(x) for x in P_VALUES_L]
    for i, d_list in enumerate(distances):
        size = len(d_list)
        yvals = np.arange(1,  size+1)/float(size)
        curves[i],  = plt.semilogx(np.sort(d_list), yvals, COLOR_LIST[P_VALUES[i]], label=P_VALUES[i])
    plt.legend(curves, legend, loc=0, fontsize=16, frameon=False)
    #
    label_x = r'$\mathit{d}^i$' 
    #plt.minorticks_off()
    plt.xlabel('%s'%label_x, fontsize=26)
    plt.ylabel('CDF', fontsize=26) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.title('')
    fig = plt.gcf()
    filename = 'results/figures/%s/S%s/DP/%s/N%d/cdf_%s_DP_N%d_%s.pdf'%(NAME, str(SIZE), d_method, ITERATIONS_NB, NAME, ITERATIONS_NB, D_TAG)
    methods.file_create(filename)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print 'Done!'