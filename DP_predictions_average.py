"""
================================================================================================
Apply Differential Testing on the results of differentially private queries on UCI Adult dataset
================================================================================================

Differential privacy is different from generalization techniques, like k-anonymity and l-diversity, 
in that it does not explicitly publish microdata. Instead, a user makes a query on the data, the 
server then computes the exact answer and adds some noise before replying. 
We consider count queries, that is in the form of "What is the number of individuals in the data that 
satisfy certain conditions". Moreover, we assume histogram queries, that is only a single round of 
querying without any kind of adaptive querying in response to previous results. 

The idea is to use the queries results ("the noised counts") to build a classifier that takes as input 
a tuple of quasi-identifiers and outputs a prediction distribution on the sensitive attribute values. 

The considered classifier is a Naive Bayes one which outputs, for a given quasi-identifier tuple q from 
a set Q, a prediction distribution Pr[S|Q=q] where S is the set of all possible sensitive values.  
As a noise, random values from a Laplacian distribution are considered. 

In more details, do the following: 
1- Read the full UCI Adult dataset from 'data/adult.all', pick up only the attributes in $ATT_QI + 
   $ATT_SA, and randomly select $SIZE records. Moreover, write the selected dataset into $filename.
   Then, quantize attributes in $ATT_CONTINUOUS. 
   Note that, to read an already selected dataset one has to set $IS_SELECT_NEW to False.  

2- For every distinct record (q,s), make $ITERATIONS_NB prediction distributions while considering 
   the entire dataset (i.e. while this record exists). Then take their average in order to obtain 
   a distribution P. After removing this record from the dataset, repeat the same process in order 
   to obtain a probability distribution P_w. 

3- For every record, compute a distance between the corresponding distributions P and P_w.
   Repeat steps 2 and 3 for every distance computation method in $DISTANCE_TAGS 
   and for every noise parameter epsilon in $EPSILON_VALUES. 

4- Plot the cumulative relative frequency of the computed distances for all epsilon values in one graph. 
   A graph is plotted for each distance method. 
""" 

# !/usr/bin/env python
# coding=utf-8

import pdb
import matplotlib.pyplot as plt
import numpy as np
import pyemd
import random
import time 
#
from utils import methods


def conditional_probabilities_compute(quasi_id, counts, epsilon): 
	"""
	Compute the probabilities Pr[s|$quasi_id] for all pairs ($quasi_id,s) whose 
	frequencies are inside dictionary $counts.  

	e.g. 
	conditional_probabilities_compute(
	                                  quasi_id='Female,White', 
	                                  counts={'Female,White;<=50K': 43, 'Female,White;>50K': 26}, 
	                                  epsilon=0.1)
	rType: dictionary of probabilities 
	"""
	mean = 0.0
	decay = 1.0/epsilon
	total_q = 0.0
	#compute the total number of records (summation of all frequencies inside $counts)
	#this is equal to the size of the dataset ($SIZE)
	nb_records_q = 0
	for key, value in counts.iteritems():
		if quasi_id == key.split(';')[0]: 
			nb_records_q += value
	#compute probabilities Pr[s|$quasi_id] for every possible sensitive value s
	probabilities = dict()
	for key, value in counts.iteritems():
		if quasi_id == key.split(';')[0]:
			noise = np.random.laplace(loc=mean, scale=decay)
			term = 1.0 + min(max(0, value + noise), nb_records_q) #The addition of 1 is for correction. 
			probabilities[key] = term 
			total_q += term
	for key, value in probabilities.iteritems():
		probabilities[key] = value/total_q
	return probabilities


def proba_distribution_compute(quasi_id, counts, epsilon, n, sa_values): 
	"""
	Obtain a prediction distribution for quasi-identifier $quasi_id after making 
	$n predictions based on the records distribution $counts inside the related dataset. 
	"""
	distribution = [0.0 for s in sa_values]
	#obtain $n prediction distributions for $quasi_id
	for i in range(n):
		#compute all probabilities Pr[s|$quasi_id] for every sensitive value s
		probabilities = conditional_probabilities_compute(quasi_id, counts, epsilon)
		#add the probabilities to the total summation
		for i, s in enumerate(sa_values):
			distribution[i] += probabilities[quasi_id+';'+s]
	return [x/float(n) for x in distribution] #normalize the summations


def distance_compute(pd1, pd2, classes_nb, d_tag='EMD', infinity=1000):
	"""
	Compute distance between two probability distributions $pd1 and $pd2. 

	e.g. distance_compute([0.45, 0.55], [0.2, 0.8], classes_nb=2)
	rType: float, a distance
	"""
	distance = 0.0
	#Switch cases according to the $d_tag
	if d_tag == 'EMD':
	    #An equal ground distance is taken between any two different attribute values. 
	    ground_matrix = []
	    for i in range(classes_nb):
	        row = []
	        for j in range(classes_nb):
	            if j == i:
	                row.append(0.0)
	            else:
	                row.append(1.0)
	        ground_matrix.append(row)
	    ground_matrix = np.array(ground_matrix) # ground distance matrix for EMD
	    distance = pyemd.emd(np.array(pd1), np.array(pd2), ground_matrix)
	elif d_tag == 'm_ratio':#obtain maximal ratio
		for i in range(classes_nb):
			max_r = 0.0   
			if (pd1[i] == 0 and pd2[i] != 0) or (pd1[i] != 0 and pd2[i] == 0):
				max_r = infinity 
			elif pd1[i] == 0 and pd2[i] == 0:
				max_r = 1.0
			elif pd1[i] != 0 and pd2[i] != 0:
				max_r = max(pd1[i], pd2[i])/min(pd1[i], pd2[i])
			#
			if max_r > distance:
				distance = max_r    
	return distance 


#ATT_NAMES= ['age', 'workclass', 'final_weight', 'education', 
#             'education_num', 'marital_status', 'occupation', 'relationship',
#             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
#             'native_country', 'income_level']   
ATT_QI = ['age', 'education', 'occupation', 'sex'] #quasi-identifier attributes
ATT_SA = 'income_level' #sensitive attributes
ATT_CONTINUOUS = ['age'] 
QUANTILES = [
            [[0,25],[26,50],[51,75],[75,100]]
            ]
NAME = 'adult' #name of the dataset
SIZE = 1000 #size of the dataset to consider
IS_SELECT_NEW = True #True is to select new data
ITERATIONS_NB = 1000 #number of predictions to make for each record
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.3, 1.0] #noise parameter to consider
COLOR_LIST  = {0.01:'c', 0.05:'b', 0.1: 'r', 0.2:'g', 0.3:'k', 1.0:'y'} 
DISTANCE_TAGS = ['EMD', 'm_ratio'] #distances to consider: EMD for Earth Mover Distance, and m_ratio for Maximal ratio


if __name__ == '__main__':
	#pdb.set_trace()
	start_time = time.time()
	#1- Import data
	print 'Read data ...'
	dataset = [] # a dataset is a table
	filename = 'results/selected_DATA/%s/selected_%s_S%s'%(NAME, NAME, str(SIZE))
	#read and filter the dataset
	dataset = methods.data_import(IS_SELECT_NEW, name=NAME, size=SIZE, qi_list=ATT_QI, sa=ATT_SA, filename=filename)
	#quantize attributes in $ATT_CONTINUOUS
	if all(x in ATT_CONTINUOUS for x in ATT_QI):
		methods.data_quantize(dataset, [ATT_QI.index(x) for x in ATT_CONTINUOUS], QUANTILES) #directly modifies $dataset

	#2- Build classifier and make predictions
	#obtain sensitive attribute values
	sa_values = methods.att_values_get(dataset, -1)
	#obtain noised counts
	counts = methods.counts_compute(dataset)
	print 'N=', ITERATIONS_NB
	counter = 0
	for key, value in counts.iteritems():
		if value == 0:
			counter += 1
	print 'counts', counter
	print 'max', max([x for x in counts.itervalues()])
	print 'min', min([x for x in counts.itervalues()])
	#initiate a distances dictionary 
	distances_dict = dict()
	for d_tag in DISTANCE_TAGS:
		distances_dict[d_tag] = []
	#
	for epsilon in EPSILON_VALUES:
		print 'Predictions for epsilon %s ...'%str(epsilon)
		predicted_distributions, predicted_distributions_w = [], []
		#for every distinct quasi-identifiers tuple $quasi_id do: 
		for key, value in counts.iteritems():
			if value != 0: #if the record exist in the dataset 
				quasi_id = key.split(';')[0]
				#while considering the entire dataset, obtain $ITERATIONS_NB prediction distributions and get their average
				distribution = proba_distribution_compute(quasi_id, counts, epsilon, ITERATIONS_NB, sa_values)
				predicted_distributions.append(distribution)
				#remove one record that has $quasi_id, then obtain $ITERATIONS_NB prediction distributions and get their average
				counts[key] -= 1
				distribution_w = proba_distribution_compute(quasi_id, counts, epsilon, ITERATIONS_NB, sa_values)
				predicted_distributions_w.append(distribution_w) 
				counts[key] += 1
		#write the obtained predictions into $predictions_file
		predictions_file = 'results/predictions/DP/%s/predictions_%s_S%s_N%s_eps%s'%(NAME, NAME, str(SIZE), str(ITERATIONS_NB), str(epsilon))
		methods.table_write([predicted_distributions[i]+predicted_distributions_w[i] for i in range(len(predicted_distributions))], predictions_file)
		#
		#3- Compute distances
		for d_tag in DISTANCE_TAGS:
			distances = []
			#for every distinct quasi-identifiers tuple, compute the distance between the corresponding prediction distributions
			for j in range(len(predicted_distributions)):
			    d = distance_compute(predicted_distributions[j], predicted_distributions_w[j], len(sa_values), d_tag)
			    distances.append(d)
			distances_dict[d_tag].append(distances)
	comp_time = time.time() - start_time 
	print 'Computation time in seconds:',  comp_time 
	#4- Plot the relative CDF of the distances  
	for d_tag in DISTANCE_TAGS:
		curves, legend = [], []
		fig = plt.figure()  
		curves = [0 for x in range(len(EPSILON_VALUES))]
		legend = ['eps = ' + str(x) for x in EPSILON_VALUES]
		index = 0
		for i in range(len(distances_dict[d_tag])):
		    size = len(distances_dict[d_tag][i])
		    yvals = np.arange(1,  size+1)/float(size)
		    curves[index],  = plt.plot(np.sort([x for x in distances_dict[d_tag][i]]), yvals, COLOR_LIST[EPSILON_VALUES[i]], label=EPSILON_VALUES[i])
		    index += 1
		plt.legend([x for x in curves], legend, loc=4, fontsize=12, frameon=False)
		#plt.xlim(min([min(x) for x in distances_dict][d_tag]), max([max(x) for x in distances_dict][d_tag]))
		plt.xlabel('%s'%d_tag, fontsize=14)
		plt.ylabel('Cumulative Relative Frequency', fontsize=14)  
		plt.title('Size=%dK, N=%dK'%(SIZE/1000, ITERATIONS_NB/1000))
		fig = plt.gcf()
		filename = 'results/figures/cdf/%s/average/cdf_DP_%s_S%d_N%d_%s_average.pdf'%(NAME, NAME, SIZE, ITERATIONS_NB, d_tag)
		methods.file_create(filename)
		fig.savefig(filename, bbox_inches='tight')
		plt.show()
		plt.close(fig)
	print 'Done!'
