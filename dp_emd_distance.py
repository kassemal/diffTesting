"""
For every distance tag: 
For every $epsilon in $P_VALUES:
for every record read $ITERATIONS_NB compute 
d_i = 0.5*sum_j distance( distribution of P_i[j]'s, distribution of P_i_w[j]'s' )

"""
import pdb
#
from utils import methods
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

NAME = 'adult'     #'adult', 'internet'
NB_classes = 13    #Number of possible classes for adult: 14, internet: 9 
SIZE = 1000       #size of the dataset to consider
NB_distinct = 619  
#
ITERATIONS_NB = 1000
P_VALUES = [0.1, 0.5, 1.0]


def compute_emd(X,Y):
    """compute EMD distance"""
    return  (1.0/len(X))*sum(abs(np.sort(X) - np.sort(Y)))

if __name__ == '__main__':

    #Read prediction distributions
    for epsilon in P_VALUES:
        print 'Distances for epsilon %s ...'%str(epsilon)
        distances = []
        for i in range(NB_distinct): 
            predicted_distributions_r, predicted_distributions_r_w = [], []
            predictions_filename = 'results/predictions/%s/S%s/DP/E%s/predictions_%s_DP_p%s_r%d'%(NAME, str(SIZE), str(epsilon), NAME, str(epsilon), i+1)
            f = open(predictions_filename, 'r')   
            for c in range(ITERATIONS_NB):#,  line in enumerate(f):
                line = f.readline()
                line = line.strip()
                line = line.split(' ')
                predicted_distributions_r.append([float(x) for x in line[:NB_classes]])
                predicted_distributions_r_w.append([float(x) for x in line[NB_classes:]])
            f.close()
            #compute emd distance for record $i
            d_list = []
            for j in range(NB_classes):
                pred_j = [pred[j] for pred in predicted_distributions_r]
                pred_j_w = [pred_w[j] for pred_w in predicted_distributions_r_w]
                d_list.append(compute_emd(pred_j, pred_j_w))
            distances.append(0.5*sum(d_list))
            #print distances

        #write distances into a file
        filename = 'results/distances/%s/S%s/DP/d_per_coordinate/N%d/distances_%s_DP_p%s_N%d_%s'%(NAME, str(SIZE), ITERATIONS_NB, NAME, str(epsilon), ITERATIONS_NB, 'EMD')
        methods.file_create(filename)
        f = open(filename, 'w')
        for line in distances:
            f.write(str(line) + '\n')
        f.close()
    print 'Done!'
