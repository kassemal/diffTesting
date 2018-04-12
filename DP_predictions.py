"""
This program aims to study the effect of the existence of a certain record inside 
a certain dataset on the predictions that are made by a classifier that is trained using this dataset. 
It considered a part from UCI Adult dataset with only two attributes: 'occupation' that is used as a 
quasi-identifier (this works as an input for the classifier) and 'income_level' that is used as a sensitive 
value (the classifier has to predict the sensitive value for a given quasi-identifier). 
The considered classifier takes as input a quasi-identifier from a set Q and 
makes a prediction from a set of sensitive attributes S. The classifier predicts, for input q, 
the value of s for which the probability Pr[q and s] is maximal for every s in S. 
Note that, that the probabilities are computed after adding Laplacian noise to the counts of (q,s) 
inside the dataset.

More precisely, it does the following: 
1- Read some already selected Adult records from a file or select new $size 
   records from 'data/adult.all' based on the value of the variable $Select. 
2- For every distinct record (q,s), it makes $NB_Iterations predictions while considering the full dataset 
   (i.e. while this record exists). It then obtains a probability distribution P from the predicted 
   sensitive attribute values. After removing this record from the dataset, it repeats the same process in 
   order to obtain a probability distribution P'. Note again that he predictions are obtained based on the maximal 
   probability Pr[q and s'] for every sensitive value s', after adding some noise. 
3- For every record, it computes a distance between the corresponding probability distributions P and P'.
   Steps 2 and 3 can be repeated for more than one distance computation method and more than one noise parameter epsilon. 
4- Plot the cumulative relative frequency of the computed distances for all epsilon values in one graph. 
   A graph is plotted for each distance method. 
""" 
#
# !/usr/bin/env python
# coding=utf-8
#
from utils.methods import import_data, compute_counts, get_att_values
from utils.methods import create_file
import numpy as np
import copy, math
from pyemd import emd
from scipy.stats import entropy
from math import sqrt,  log 
from random import randint
import matplotlib.pyplot as plt
#
def compute_probabilities(counts, epsilon): 
	"""
	This method computes the probabilities Pr[q and s] for all pairs (q,s) whose 
	frequencies are inside dictionary $counts.  
	"""
	mean = 0.0
	decay = 1.0/epsilon
	total = 0.0
	#compute the total number of records (summation of all frequencies inside $counts)
	#this is equal to the size of the dataset ($Size)
	nb_records = 0
	for value in counts.itervalues():
		nb_records += value
	#compute probabilities
	probabilities = dict()
	probabilities = copy.deepcopy(counts)
	for key, value in probabilities.iteritems():
		noise = np.random.laplace(loc=mean, scale=decay)
		term = 1.0 + min(max(0, value + noise), nb_records) #The addition of 1 is just for correction. 
		probabilities[key] = term                                #Also to avoid having total = 0.0
		total += term
	for key, value in probabilities.iteritems():
		probabilities[key] = value/total
	return probabilities
#
def compute_proba_distribution(q, counts, epsilon, N, sa_values): 
	"""
	This method obtains a prediction distribution for quasi-identifier $q after making 
	$N predictions based on the records distribution $counts inside the related dataset. 
	"""
	pred_proba = [0.0 for s in sa_values]
	#obtain $N predictions for $q
	for i in range(N):
		#compute all probabilities Pr[q' and s] for all QI q' and SA s
		probabilities = dict()
		probabilities = compute_probabilities(counts, epsilon)
		#get the maximal probability Pr[$q and s] among all pairs ($q, s) for all SA s
		maxx = 0.0
		p_index = randint(0, len(sa_values)-1) #if all probabilities are equal, then one of them is randomly predicted
		for i, s in enumerate(sa_values):
			if probabilities[q+';'+s] > maxx:
				maxx = probabilities[q+';'+s]
				p_index = i
		pred_proba[p_index] += 1
	return pred_proba #Note that $pred_proba is not normalized
#
def distance_D(pd1, pd2, nb_classes, N, D='EMD'):
	"""
	Computes distance between two probability distributions. 
	$N is the total number of predictions that have been used to obtain pd1 and pd2. 
	$N is used to denote the infinity when a division by 0 is encountered. 
	"""
	distance = 0.0
	if D == 'EMD':
	    #An equal ground distance is taken between any two different attribute values. 
	    matrix = []
	    for i in range(nb_classes):
	        raw = []
	        for j in range(nb_classes):
	            if j == i:
	                raw.append(0.0)
	            else:
	                raw.append(1.0)
	        matrix.append(raw)
	    matrix_d = np.array(matrix) # ground distance matrix for EMD
	    distance = emd(np.array(pd1), np.array(pd2), matrix_d)
	elif D == 'ratio':
		for i in range(nb_classes):
			mr = 0.0   
			if (pd1[i] == 0 and pd2[i] != 0) or (pd1[i] != 0 and pd2[i] == 0):
				mr = 5#N
			elif pd1[i] == 0 and pd2[i] == 0:
				mr = 1
			elif pd1[i] != 0 and pd2[i] != 0:
				mr = max(pd1[i]/pd2[i], pd2[i]/pd1[i])
			#
			if mr > distance:
				distance = mr    
	return distance 
#
#ATT_NAMES_adult = ['age', 'workclass', 'final_weight', 'education',
#             'education_num', 'marital_status', 'occupation', 'relationship',
#             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
#             'native_country', 'income_level']   
QI_ATT = ['occupation'] #quasi-identifier attributes
SA_ATT = ['income_level'] #sensitive attributes
Name = 'adult' #name of the dataset
Size = 1000 #size of the dataset to consider
Select = False #True is to select new data
NB_Iterations = 70 #number of predictions to make for each record
Epsilons = [0.01, 0.05, 0.1, 0.2, 0.3] #[0.01, 0.1, 1.0, 2.0] #noise parameter values to consider
color_list  = {0.01:'c', 0.05:'b', 0.1: 'r', 0.2:'g', 0.3:'k'} 
marker_list = {0.01:'x', 0.05:'o', 0.1: 's', 0.2:'>', 0.3:'*'} 
D = ['EMD', 'ratio'] #distances to consider
#
if __name__ == '__main__':

	#1- Import data
	print 'Read data ...'
	selected_DATA = []
	filename = 'results/selected_DATA/%s/testing_data'%Name #selected_%s_S%s'%(Name,Name,str(Size))
	selected_DATA = import_data(Select, name=Name, size=Size, qi_list=QI_ATT, sa_list=SA_ATT, filename=filename)
	#
	#2- Add noise and compute prediction probability distributions
	QI_num = len(QI_ATT)
	SA_num = len(SA_ATT)
	COUNTS = compute_counts(selected_DATA, QI_num, SA_num)
	att_values = get_att_values(selected_DATA)
	Distances_all_D = dict()
	for xD in D:
		Distances_all_D[xD] = []
	for epsilon in Epsilons:
		print 'Predictions for epsilon %s ...'%str(epsilon)
		predicted_proba, predicted_proba_w = [], []
		for key in COUNTS:
			q = key.split(';')[0]
			#obtain $NB_Iterations with the record
			proba = compute_proba_distribution(q, COUNTS, epsilon, NB_Iterations, att_values[-1])
			predicted_proba.append(proba)
			#obtain $NB_Iterations without the record
			COUNTS[key] -= 1
			proba_w = compute_proba_distribution(q, COUNTS, epsilon, NB_Iterations, att_values[-1])
			predicted_proba_w.append(proba_w) 
			COUNTS[key] += 1
		#
		#3- Compute distances
		for xD in D:
			Distance_pp = []
			for j in range(len(predicted_proba)):
			    d = distance_D(predicted_proba[j], predicted_proba_w[j], len(att_values[-1]), NB_Iterations, xD)
			    Distance_pp.append(d)
			Distances_all_D[xD].append(Distance_pp)
	#4- Plot the relative CDF of the distances  
	for xD in D:
		curves, leg = [], []
		fig = plt.figure()  
		curves = [0 for j in range(len(Epsilons))]
		leg = ['eps = ' + str(x) for x in Epsilons]
		index = 0
		for i in range(len(Distances_all_D[xD])):
		    size = len(Distances_all_D[xD][i])
		    yvals = np.arange(1,  size+1)/float(size)
		    curves[index],  = plt.plot(np.sort([x for x in Distances_all_D[xD][i]]), yvals, color_list[Epsilons[i]], marker=marker_list[Epsilons[i]], label=Epsilons[i])
		    index += 1
		plt.legend([x for x in curves],leg,loc=4,fontsize=12,frameon=False)
		#plt.xlim(min([min(x) for x in Distances_all_D][xD]), max([max(x) for x in Distances_all_D][xD]))
		plt.xlabel('%s'%xD, fontsize=14)
		plt.ylabel('Cumulative Relative Frequency',fontsize=14)  
		plt.title('Size=%dK, N=%dK'%(Size/1000,NB_Iterations/1000))
		fig = plt.gcf()
		filename = 'results/figures/cdf/%s/cdf_DP_%s_S%d_N%d_%s.pdf'%(Name,Name,Size,NB_Iterations,xD)
		create_file(filename)
		fig.savefig(filename, bbox_inches='tight')
		plt.show()
		plt.close(fig)
	print 'Done!'
