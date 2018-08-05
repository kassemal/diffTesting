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
NB_classes = 13 #max: adult: 14, internet: 9
SIZE = 1000     #size of the dataset to consider
#
ANON_TECH = 'k_anonymity' #anonymization technique 'k_anonymity', 'l_diversity', 'DP'
MLA = 'BNB' # 'GNB'
ENC_TECH = 'TF' #encoding technique 'TF' (True-False), 'OH' (one-hot) 
P_VALUES = [1, 2, 3, 5, 8] # Parameter used for anonymization
COLOR_LIST  = {1:'b', 2:'r', 3:'g', 5:'y', 8:'m'}  
#
x_label = {'k_anonymity':'K', 'l_diversity':'L'}
xx_label = {'EMD':'Earth Mover\'s Distance', 'm_ratio':'Maximal Ratio'}
y_label = {'EMD':'EMD Distance', 'm_ratio':'Ratio'}

if __name__ == '__main__':

    #initiate distance dictionary
    distances_dict = dict()
    max_d = dict()
    for d_tag in DISTANCE_TAGS:
        distances_dict[d_tag] = []
        max_d[d_tag] = []
    #Read exact predictions and predictions distributions
    for p_value in P_VALUES:
        predicted_distributions, predicted_distributions_w = [], []
        filename_predictions = 'results/predictions/%s/S%s/%s/%s/%s/predictions_%s_%s_p%s'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, NAME, ANON_TECH, str(p_value))
        f = open(filename_predictions, 'r')   
        for i,  line in enumerate(f):
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
            filename_d = 'results/distances/%s/S%s/%s/%s/%s/%s/distances_%s_%s_p%s_%s'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, str(p_value), d_tag)
            methods.file_create(filename_d)
            f = open(filename_d, 'w')
            for line in distances:
                f.write(str(line) + '\n')
            f.close()

    #Plot 
    for d_tag in DISTANCE_TAGS:
        #plot max. distances
        filename_max_d = 'results/figures/%s/S%s/%s/%s/%s/%s/max_d_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, d_tag)
        fig = plt.figure()  
        plt.plot(range(len(P_VALUES)), max_d[d_tag], '-x')
        plt.xticks(range(len(P_VALUES)), P_VALUES)
        plt.ylabel('Maximum %s'%y_label[d_tag], fontsize=16)
        plt.xlabel('%s'%x_label[ANON_TECH], fontsize=16)
        plt.show()
        #fig = plt.gcf()
        methods.file_create(filename_max_d)
        fig.savefig(filename_max_d, bbox_inches='tight')
        plt.close(fig)
        #
        #plot cdf
        curves, legend = [], []
        fig = plt.figure()  
        curves = [0 for x in range(len(P_VALUES))]
        legend = [x_label[ANON_TECH] + ' = ' + str(x) for x in P_VALUES]
        #
        for i, d_list in enumerate(distances_dict[d_tag]):
            size = len(d_list)
            yvals = np.arange(1,  size+1)/float(size)
            curves[i],  = plt.semilogx(np.sort([x for x in d_list]), yvals, COLOR_LIST[P_VALUES[i]], label=P_VALUES[i])
        plt.legend([x for x in curves], legend, loc=4, fontsize=12, frameon=False)
        #plt.xlim()
        plt.xlabel('%s'%xx_label[d_tag], fontsize=16)
        plt.ylabel('Cumulative Relative Frequency', fontsize=16)  
        #plt.title('')
        fig = plt.gcf()
        filename_cdf = 'results/figures/%s/S%s/%s/%s/%s/%s/cdf_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, d_tag)
        methods.file_create(filename_cdf)
        fig.savefig(filename_cdf, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    print 'Done!'