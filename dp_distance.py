"""
For every $p_value in $P_VALUES, read the related predictions, compute and plot the corresponding distances.
"""
import pdb
import matplotlib.pyplot as plt
import numpy as np
#
from utils import methods

DISTANCE_TAGS = ['EMD', 'm_ratio'] #distances to consider: EMD for Earth Mover Distance, and m_ratio for Maximal ratio
NAME = 'adult'  #name of the dataset: 'adult', 'internet'
NB_classes = 13 #make sure to use the correct number of classes. Maximum number for adult: 14, internet: 9 
SIZE = 1000    #size of the dataset to consider
ITERATIONS_NB = 500 
#
P_VALUES = [0.1, 0.3, 0.5, 1.0]
COLOR_LIST  = {0.1:'m', 0.3:'g', 0.5:'r', 1.0:'b'} 
#
y_label = {'EMD':'EMD distance', 'm_ratio':'Ratio'}
x_label = {'EMD':'Earth Mover\'s Distance', 'm_ratio':'Maximal Ratio'}

if __name__ == '__main__':

    #initiate distance dictionary
    distances_dict = dict()
    max_d = dict()
    for d_tag in DISTANCE_TAGS:
        distances_dict[d_tag] = []
        max_d[d_tag] = []
    #Read prediction distributions
    for p_value in P_VALUES:
        predicted_distributions, predicted_distributions_w = [], []
        filename_predictions = 'results/predictions/%s/S%s/DP/N%d/predictions_%s_DP_p%s'%(NAME, str(SIZE), ITERATIONS_NB, NAME, str(p_value))
        f = open(filename_predictions, 'r')   
        for i,  line in enumerate(f):
            prob, prob_w = [], []
            line = line.split(' ')
            predicted_distributions.append([float(line[j]) for j in range(NB_classes)])
            #
            predicted_distributions_w.append([float(line[j]) for j in range(NB_classes, 2*NB_classes)])
        f.close()
        #
        #Compute distances
        for d_tag in DISTANCE_TAGS:
            distances = []
            #for every distinct record, compute the distance between the corresponding prediction distributions
            for i in range(len(predicted_distributions)):
                distances.append(methods.distance_compute(predicted_distributions[i], predicted_distributions_w[i], NB_classes, d_tag))
            distances_dict[d_tag].append(distances)
            max_d[d_tag].append(max(distances))
            filename_d = 'results/distances/%s/S%s/DP/N%d/distances_%s_DP_p%s_N%d_%s'%(NAME, str(SIZE), ITERATIONS_NB, NAME, str(p_value), ITERATIONS_NB, d_tag)
            methods.file_create(filename_d)
            f = open(filename_d, 'w')
            for line in distances:
                f.write(str(line) + '\n')
            f.close()

    #Plot  
    for d_tag in DISTANCE_TAGS:
        #plot max. distances
        filename_max_d = 'results/figures/%s/S%s/DP/N%d/max_d_%s_DP_N%d_%s.pdf'%(NAME, str(SIZE), ITERATIONS_NB, NAME, ITERATIONS_NB, d_tag)
        fig = plt.figure()  
        plt.plot(range(len(P_VALUES)), max_d[d_tag], '-x')
        plt.xticks(range(len(P_VALUES)), P_VALUES)
        plt.ylabel('Maximum %s'%y_label[d_tag], fontsize=16)
        plt.xlabel('eps', fontsize=16)
        plt.show()
        #fig = plt.gcf()
        methods.file_create(filename_max_d)
        fig.savefig(filename_max_d, bbox_inches='tight')
        plt.close(fig)
        #plot crf
        curves, legend = [], []
        fig = plt.figure()  
        curves = [0 for x in range(len(P_VALUES))]
        legend = ['eps = ' + str(x) for x in P_VALUES]
        #
        for i, d_list in enumerate(distances_dict[d_tag]):
            size = len(d_list)
            yvals = np.arange(1,  size+1)/float(size)
            curves[i],  = plt.semilogx(np.sort(d_list), yvals, COLOR_LIST[P_VALUES[i]], label=P_VALUES[i])
        plt.legend(curves, legend, loc=0, fontsize=12, frameon=False)
        #plt.xlim()
        #plt.xticks()
        plt.xlabel('%s'%x_label[d_tag], fontsize=16)
        plt.ylabel('Cumulative Relative Frequency', fontsize=16)  
        #plt.title('')
        fig = plt.gcf()
        filename_crf = 'results/figures/%s/S%s/DP/N%d/crf_%s_DP_N%d_%s.pdf'%(NAME, str(SIZE), ITERATIONS_NB, NAME, ITERATIONS_NB, d_tag)
        methods.file_create(filename_crf)
        fig.savefig(filename_crf, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    print 'Done!'