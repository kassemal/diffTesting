"""
For every $p_value in $P_VALUES, read the related predictions, compute and plot the corresponding distances.
"""
import pdb
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
#
from utils import methods

NAME = 'adult'  #name of the dataset: 'adult', 'internet'
ANON_TECH = 'k_anonymity' #anonymization technique 'k_anonymity', 'l_diversity', 'DP'
#
NB_CLASSES  = {'adult':13, 'internet':9} #Has to be set manually 
NB_classes = NB_CLASSES[NAME] #adult: 14, internet: 9
P_VALUES = [2, 4, 8, 16] # Parameter used for anonymization
COLOR_LIST  = {2:'r', 4:'g', 8:'y', 16:'m'}  
#
# P_VALUES = [1, 2, 3, 4, 5, 6] # Parameter used for anonymization
# COLOR_LIST  = {1:'b', 2:'r', 3:'g', 4:'y', 5:'m', 6:'k'} 

#
DISTANCE_TAGS = ['EMD', 'm_ratio'] #distances to consider: EMD for Earth Mover Distance, and m_ratio for Maximal ratio
SIZE = 1000    #size of the dataset to consider
MLA = 'BNB' # 'GNB'
ENC_TECH = 'TF' #encoding technique 'TF' (True-False), 'OH' (one-hot) 
#
x_label = {'k_anonymity':'k', 'l_diversity':r'$\ell$'}

def max_plot(x_values, y_values, tech, tag, filename):
    fig = plt.figure()  
    plt.plot(range(len(x_values)), y_values, '-x')
    plt.xticks(range(len(x_values)), x_values, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(r'$\delta$', fontsize=26)
    plt.xlabel('%s'%x_label[tech], fontsize=26)
    plt.show()
    #fig = plt.gcf()
    methods.file_create(filename)
    fig.savefig(filename, bbox_inches='tight', format='pdf')
    plt.close(fig)
    return

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
            prob, prob_w = [], []
            lline = line.split(' ')
            for j in range(NB_classes):
                prob.append(float(lline[j]))
            predicted_distributions.append(prob)
            #
            for j in range(NB_classes, 2*NB_classes):
                prob_w.append(float(lline[j]))
            predicted_distributions_w.append(prob_w)
        f.close()
        #
        #Compute distances
        for d_tag in DISTANCE_TAGS:
            distances = []
            #for every distinct record, compute the distance between the corresponding prediction distributions
            for i in range(len(predicted_distributions)):
                d = methods.distance_compute(predicted_distributions[i], predicted_distributions_w[i], NB_classes, d_tag)
                distances.append(d)
            distances_dict[d_tag].append(distances)
            max_d[d_tag].append(max(distances))
            filename = 'results/distances/%s/S%s/%s/%s/%s/%s/distances_%s_%s_p%s_%s'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, str(p_value), d_tag)
            methods.file_create(filename)
            f = open(filename, 'w')
            for line in distances:
                f.write(str(line) + '\n')
            f.close()

    #Plot the relative CDF of the distances  
    for d_tag in DISTANCE_TAGS:
        #plot max. distances
        max_dfile = 'results/figures/%s/S%s/%s/%s/%s/%s/max_d_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, d_tag)
        print max_d[d_tag]
        max_plot(P_VALUES, max_d[d_tag], ANON_TECH, d_tag, max_dfile)
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
        plt.legend([x for x in curves], legend, loc=0, fontsize=16, frameon=False)
        #
        label_x = r"$d^{i}$"
        plt.xlabel('%s'%label_x, fontsize=26)
        plt.ylabel('CDF', fontsize=26) 
    	plt.xticks(fontsize=20)
    	plt.yticks(fontsize=20)
        #plt.title('')
        fig = plt.gcf()
        filename = 'results/figures/%s/S%s/%s/%s/%s/%s/cdf_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, d_tag)
        methods.file_create(filename)
        fig.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    print 'Done!'
