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
NB_classes = 14 #make sure to use the correct number of classes. Maximum number for adult: 14, internet: 9 
SIZE = 10000    #size of the dataset to consider
ITERATIONS_NB = 100 
#
P_VALUES = [0.01, 0.05, 0.1] 
COLOR_LIST  = {0.01:'m', 0.05:'g', 0.1:'b'} 
#
def max_plot(x_values, y_values, tag, filename):
    """
    plot $y_values with respect to $x_values and save the result in $filename. 
    """
    y_label = {'EMD':'EMD distance', 'm_ratio':'Ratio'}
    fig = plt.figure()  
    plt.plot(range(len(x_values)), y_values, '-x')
    plt.xticks(range(len(x_values)), x_values)
    plt.ylabel('Maximum %s'%y_label[tag], fontsize=16)
    plt.xlabel('eps', fontsize=16)
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
    #Read prediction distributions
    for p_value in P_VALUES:
        predicted_distributions, predicted_distributions_w = [], []
        filename_predictions = 'results/predictions/%s/S%s/DP/N%d/predictions_%s_DP_p%s'%(NAME, str(SIZE), ITERATIONS_NB, NAME, str(p_value))
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
            filename = 'results/distances/%s/S%s/DP/N%d/distances_%s_DP_p%s_N%d_%s'%(NAME, str(SIZE), ITERATIONS_NB, NAME, str(p_value), ITERATIONS_NB, d_tag)
            methods.file_create(filename)
            f = open(filename, 'w')
            for line in distances:
                f.write(str(line) + '\n')
            f.close()

    #Plot the relative CDF of the distances  
    for d_tag in DISTANCE_TAGS:
        #plot max. distances
        max_dfile = 'results/figures/%s/S%s/DP/N%d/max_d_%s_DP_N%d_%s.pdf'%(NAME, str(SIZE), ITERATIONS_NB, NAME, ITERATIONS_NB, d_tag)
        max_plot(P_VALUES, max_d[d_tag], d_tag, max_dfile)
        #plot cdf
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
        #
        label_x = ''
        if d_tag == 'EMD':
            label_x = 'Earth Mover\'s Distance'
            #plt.xlim(10**-1, 10**0)
        elif d_tag == 'm_ratio':
            label_x = 'Maximal Ratio'
            #plt.xlim()
            #plt.xticks()
        plt.xlabel('%s'%label_x, fontsize=16)
        plt.ylabel('Cumulative Relative Frequency', fontsize=16)  
        #plt.title('')
        fig = plt.gcf()
        filename = 'results/figures/%s/S%s/DP/N%d/cdf_%s_DP_N%d_%s.pdf'%(NAME, str(SIZE), ITERATIONS_NB, NAME, ITERATIONS_NB, d_tag)
        methods.file_create(filename)
        fig.savefig(filename, bbox_inches='tight')
        #plt.show()
        plt.close(fig)
    print 'Done!'