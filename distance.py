"""
For every $p_value in $P_VALUES, read the related predictions, compute and plot the corresponding distances.
"""
import pdb
import matplotlib.pyplot as plt
import numpy as np
#
from utils import methods

DISTANCE_TAGS = ['EMD', 'm_ratio'] #distances to consider: EMD for Earth Mover Distance, and m_ratio for Maximal ratio
NAME = 'adult' #name of the dataset: 'adult', 'internet'
NB_classes = 14 #adult: 14, internet: 9
SIZE = 2000   #size of the dataset to consider
#
ANON_TECH = 'k_anonymity' #anonymization technique 'k_anonymity', 'l_diversity', 'DP'
MLA = 'BNB' # 'GNB'
ENC_TECH = 'TF' #encoding technique 'TF' (True-False), 'OH' (one-hot) 
P_VALUES = [1, 2, 3, 5, 8] # Parameter used for anonymization
COLOR_LIST  = {1:'b', 2:'r', 3:'g', 5:'y', 8:'m'}  
#
#P_VALUES = [0.01, 0.05, 0.1, 0.3, 1.0] 
#COLOR_LIST  = {0.01:'m', 0.05:'c', 0.1:'g', 0.3:'r', 1.0:'b'} 
#
x_label = {'k_anonymity':'K', 'l_diversity':'L', 'DP':'eps'}

def max_plot(x_values, y_values, tech, tag, filename):
    y_label = {'EMD':'EMD distance', 'm_ratio':'Maximal Ratio'}
    fig = plt.figure()  
    plt.plot(range(len(x_values)), y_values, '-x')
    plt.xticks(range(len(x_values)), x_values)
    plt.ylabel('Maximum %s'%y_label[tag], fontsize=16)
    plt.xlabel('%s'%x_label[tech], fontsize=16)
    #plt.show()
    #fig = plt.gcf()
    methods.file_create(filename)
    fig.savefig(filename, bbox_inches='tight')
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
        if ANON_TECH == 'DP':
            filename_predictions = 'results/predictions/%s/S%s/%s/predictions_%s_%s_p%s'%(NAME, str(SIZE), ANON_TECH, NAME, ANON_TECH, str(p_value))
        else:
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
            if ANON_TECH == 'DP':
                filename = 'results/distances/%s/S%s/%s/%s/distances_%s_%s_p%s_%s'%(NAME, str(SIZE), ANON_TECH, d_tag, NAME, ANON_TECH, str(p_value), d_tag)
            else:
                filename = 'results/distances/%s/S%s/%s/%s/%s/%s/distances_%s_%s_p%s_%s'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, str(p_value), d_tag)
            methods.file_create(filename)
            f = open(filename, 'w')
            for line in distances:
                f.write(str(line) + '\n')
            f.close()


    #Plot the relative CDF of the distances  
    for d_tag in DISTANCE_TAGS:

        if ANON_TECH == 'DP':
            filename = 'results/figures/%s/S%s/%s/%s/cdf_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, d_tag, NAME, ANON_TECH, d_tag)
            max_dfile = 'results/figures/%s/S%s/%s/%s/max_d_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, d_tag, NAME, ANON_TECH, d_tag)
        else:
            filename = 'results/figures/%s/S%s/%s/%s/%s/%s/cdf_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, d_tag)
            max_dfile = 'results/figures/%s/S%s/%s/%s/%s/%s/max_d_%s_%s_%s.pdf'%(NAME, str(SIZE), ANON_TECH, MLA, ENC_TECH, d_tag, NAME, ANON_TECH, d_tag)

        #plot max. distances
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
        plt.legend([x for x in curves], legend, loc=4, fontsize=12, frameon=False)
        #
        label_x = ''
        if d_tag == 'EMD':
            label_x = 'Earth Mover\'s Distance'
            #plt.xlim(10**-6,1)
        elif d_tag == 'm_ratio':
            label_x = 'Maximal Ratio'
            #plt.xlim(10**0,2*10**0)
        plt.xlabel('%s'%label_x, fontsize=18)
        plt.ylabel('Cumulative Relative Frequency', fontsize=18)  
        #plt.title('')
        fig = plt.gcf()
        methods.file_create(filename)
        fig.savefig(filename, bbox_inches='tight')
        #plt.show()
        plt.close(fig)
    print 'Done!'