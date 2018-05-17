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
import itertools
import pyemd
import random
import time 
#
from utils import methods


def conditional_probabilities_compute(counts, epsilon): 
	"""
	Compute the probabilities Pr[s|$quasi_id] for all pairs ($quasi_id,s) whose 
	frequencies are inside dictionary $counts.  

	[
	{'a': {'s2': 0.7380952380952381, 's1': 0.5}, 'b': {'s2': 0.2619047619047619, 's1': 0.5}}, 

	{'m': {'s2': 0.5, 's1': 0.7380952380952381}, 'n': {'s2': 0.5, 's1': 0.2619047619047619}}
	]

	"""
	#add noise to counts
	cond_probabilities = [dict() for x_dict in counts]
	total_per_att = [dict() for x_dict in counts]
	for j, dictionary_j in enumerate(counts):
	    for key_qi, dict_counts in dictionary_j.iteritems():
			for key_sa, val_count in dict_counts.iteritems():
				noise = np.random.laplace(loc=0.0, scale=1.0/epsilon)
				proba_numerator = 1.0 + max(0, val_count + np.random.laplace(loc=0.0, scale=1.0/epsilon))
				try:
					cond_probabilities[j][key_qi][key_sa] = proba_numerator
				except KeyError:
					cond_probabilities[j][key_qi]  = {key_sa: proba_numerator}
				#add the value to the related total
				try:
					total_per_att[j][key_sa] += proba_numerator
				except KeyError:
					total_per_att[j][key_sa]  = proba_numerator
	#normalize
	for j, dictionary_j in enumerate(counts):
	    for key_qi, dict_counts in dictionary_j.iteritems():
		    for key_sa, val_count in dict_counts.iteritems():
					cond_probabilities[j][key_qi][key_sa] /= total_per_att[j][key_sa]
	return cond_probabilities


def sa_values_proba_compute(counts, epsilon):
	""" {'s2': 0.5, 's1': 0.5}"""
	sa_probabilities = dict()
	if len(counts) != 0:
		for key_qi, dict_counts in counts[0].iteritems():
			for key_sa, val_count in dict_counts.iteritems():
				try:
					sa_probabilities[key_sa] += val_count
				except KeyError:
					sa_probabilities[key_sa]  = val_count
		#add noise to SA values counts
		total_sa = 0.0
		for key_sa, val_count_sa in sa_probabilities.iteritems():
			sa_probabilities[key_sa] = 1.0 + max(0, val_count_sa + np.random.laplace(loc=0.0, scale=1.0/epsilon))
			total_sa += sa_probabilities[key_sa]
		#normalize probabilities
		for key_sa in sa_probabilities.keys():
			sa_probabilities[key_sa] /= total_sa
	return sa_probabilities


def proba_distributions_compute(distinct_records, counts, epsilon): 
	"""
	Obtain a prediction distribution for quasi-identifier $quasi_id after making 
	$n predictions based on the records distribution $counts inside the related dataset. 
	
	tuples_qi = [['a', 'm'], ['a', 'n'], ['b', 'm']]
	
	[
	{'s2': 0.18452380952380953, 's1': 0.18452380952380953}, 
	{'s2': 0.18452380952380953, 's1': 0.06547619047619048}, 
	{'s2': 0.06547619047619048, 's1': 0.18452380952380953}
	]


	"""
	distributions = [dict() for x in distinct_records]
	#compute all probabilities Pr[s|$quasi_id] for every sensitive value s
	cond_probabilities = conditional_probabilities_compute(counts, epsilon)
	sa_probabilities = sa_values_proba_compute(counts, epsilon)
	#add the probabilities to the total summation
	for i in range(len(distinct_records)):
		total_i = 0.0
		for key_sa, val_Proba_sa in sa_probabilities.iteritems():
			proba_quasi_sa = val_Proba_sa
			for j, val_qi in enumerate(distinct_records[i][:-1]):
				proba_quasi_sa *= cond_probabilities[j][val_qi][key_sa]
			distributions[i][key_sa] = proba_quasi_sa
			total_i += proba_quasi_sa
		#normalize
		for key_sa, val_Proba in distributions[i].iteritems():
			distributions[i][key_sa] = val_Proba/total_i
	return distributions 


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
	    distance = pyemd.emd(np.array([float(x) for x in pd1]), np.array([float(x) for x in pd2]), ground_matrix)
	elif d_tag == 'm_ratio':#obtain maximal ratio
		for i in range(classes_nb):
			max_r = 0.0   
			if (float(pd1[i]) == 0 and float(pd2[i]) != 0) or (float(pd1[i]) != 0 and float(pd2[i]) == 0):
				max_r = infinity 
			elif float(pd1[i]) == 0 and float(pd2[i]) == 0:
				max_r = 1.0
			elif float(pd1[i]) != 0 and float(pd2[i]) != 0:
				max_r = max(float(pd1[i]), float(pd2[i]))/min(float(pd1[i]), float(pd2[i]))
			#
			if max_r > distance:
				distance = max_r    
	return distance 


#ATT_NAMES= ['age', 'workclass', 'final_weight', 'education', 
#             'education_num', 'marital_status', 'occupation', 'relationship',
#             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
#             'native_country', 'income_level']   
# ATT_QI =  ['age', 'education', 'sex', 'native_country'] #quasi-identifier attributes
# ATT_SA = 'occupation' #sensitive attributes
ATT_QI =  ['age', 'education_attainment', 'major_occupation', 'marital_status', 'race'] #quasi-identifier attributes
ATT_SA = 'household_income' #sensitive attributes
ATT_CONTINUOUS = ['age'] 
QUANTILES = [
            [[0,25],[26,50],[51,75],[75,100]] #age
            ]
NAME = 'internet' #name of the dataset
SIZE = 7    #size of the dataset to consider
IS_SELECT_NEW = True #True is to select new data
ITERATIONS_NB = 2    #number of predictions to make for each record
EPSILON_VALUES = [0.1]#[0.01, 0.05, 0.1, 0.3, 1.0] #noise parameter to consider
COLOR_LIST  = {0.01:'c', 0.05:'b', 0.1: 'r', 0.2:'g', 0.3:'k', 1.0:'y'} 
DISTANCE_TAGS = ['EMD', 'm_ratio'] #distances to consider: EMD for Earth Mover Distance, and m_ratio for Maximal ratio


if __name__ == '__main__':

	#pdb.set_trace()
	time_start = time.time()
	#1- Import data
	print 'Read data ...'
	dataset = [] # a dataset is a table
	filename = 'results/selected_DATA/%s/selected_%s_S%s'%(NAME, NAME, str(SIZE))
	#obtain the dataset 
	dataset = methods.data_import(IS_SELECT_NEW, name=NAME, size=SIZE, qi_list=ATT_QI, sa=ATT_SA, filename=filename)
	print dataset
	#quantize attributes in $ATT_CONTINUOUS
	if all(x in ATT_QI for x in ATT_CONTINUOUS):
		methods.data_quantize(dataset, [ATT_QI.index(x) for x in ATT_CONTINUOUS], QUANTILES) #directly modifies $dataset

	#2- Build classifier and make predictions
	#obtain counts for the pairs (qi_j, sa). Note that a record of $dataset has the form (qi_1, ..., qi_m, sa) 
	counts, sa_values = methods.counts_compute(dataset) # $counts is a list of dictionaries of dictionaries: [{{}}]
	#initiate a distances dictionary 
	distances_dict_average = dict()
	distances_dict_modes = dict()
	for d_tag in DISTANCE_TAGS:
	 	distances_dict_average[d_tag] = []
	 	distances_dict_modes[d_tag] = []
	#Obtain the list of distinct quasi-identifier tuples
	dataset.sort()
	distinct_records = list(tup_qi for tup_qi,_ in itertools.groupby(dataset))

	#for every $epsilon obtain the related prediction distributions
	for epsilon in EPSILON_VALUES:
		print 'Predictions for epsilon %s ...'%str(epsilon)
		predicted_distributions_average, predicted_distributions_average_w = [], []
		predicted_distributions_modes, predicted_distributions_modes_w = [], []
		#make $ITERATIONS_NB for every distinct quasi-identifier tuple
		for iteration in range(ITERATIONS_NB):
			#obtain a set of distributions: a distribution for each quasi-identifier tuple
			distributions = proba_distributions_compute(distinct_records, counts, epsilon) 
			#
			for i in range(len(distinct_records)):
				try:
					for key_sa in distributions[i].keys():
						predicted_distributions_average[i][key_sa] += distributions[i][key_sa]
				except IndexError:
					predicted_distributions_average.append(distributions[i])
				#
				#modes ........
				mode_i = max(distributions[i], key=distributions[i].get)
				try:
					predicted_distributions_modes[i][mode_i] += 1.0
				except IndexError:
					predicted_distributions_modes.append(dict.fromkeys(distributions[i], 0.0))
					predicted_distributions_modes[i][mode_i] = 1.0				
		#normalize
		for i in range(len(distinct_records)): 
			for key_sa in predicted_distributions_average[i].keys():
				predicted_distributions_average[i][key_sa] /= ITERATIONS_NB
				#
				predicted_distributions_modes[i][key_sa]  /= ITERATIONS_NB

		#### model M_i
		#
		#
		predicted_distributions_average_w = []
		predicted_distributions_modes_w = []
		for i, tup_qi in enumerate(distinct_records):

			for j, val_qi in enumerate(tup_qi[:-1]): #decrement by 1
				counts[j][val_qi][tup_qi[-1]] -= 1
			for iteration in range(ITERATIONS_NB):
				distribution_tup_qi = proba_distributions_compute([tup_qi], counts, epsilon) 

				try:
					for key_sa in distribution_tup_qi[0].keys():
						predicted_distributions_average_w[i][key_sa] += distribution_tup_qi[0][key_sa]
				except IndexError:
					predicted_distributions_average_w.append(distribution_tup_qi[0])
				#
				#modes ........
				mode_i = max(distribution_tup_qi[0], key=distribution_tup_qi[0].get)
				try:
					predicted_distributions_modes_w[i][mode_i] += 1.0
				except IndexError:
					predicted_distributions_modes_w.append(dict.fromkeys(distribution_tup_qi[0], 0.0))
					predicted_distributions_modes_w[i][mode_i] = 1.0	

			for j, val_qi in enumerate(tup_qi[:-1]): #decrement by 1
				counts[j][val_qi][tup_qi[-1]] += 1
		#normalize
		for i in range(len(distinct_records)): 
			for key_sa in predicted_distributions_average_w[i].keys():
				predicted_distributions_average_w[i][key_sa] /= ITERATIONS_NB
				#
				predicted_distributions_modes_w[i][key_sa]  /= ITERATIONS_NB

		print 'Computation time in seconds:',  time.time() - time_start 


		predictions_average = []
		#write the obtained predictions into $predictions_file
		filename_predictions = 'results/predictions_average/DP/%s/predictions_average_%s_S%s_N%s_eps%s'%(NAME, NAME, str(SIZE), str(ITERATIONS_NB), str(epsilon))
		methods.file_create(filename_predictions)
		f = open(filename_predictions, 'w')
		f.write(' '.join(str(val_sa) for val_sa in sa_values) + '\n')
		for i in range(len(distinct_records)):
			pred_with = ' '.join(str(predicted_distributions_average[i][val_sa]) for val_sa in sa_values) + ' '
			pred_without = ' '.join(str(predicted_distributions_average_w[i][val_sa]) for val_sa in sa_values) + '\n'
			f.write(pred_with+pred_without)
			predictions_average.append((pred_with+pred_without).split())


		predictions_modes = []
		#write the obtained predictions into $predictions_file
		filename_predictions = 'results/predictions_modes/DP/%s/predictions_modes_%s_S%s_N%s_eps%s'%(NAME, NAME, str(SIZE), str(ITERATIONS_NB), str(epsilon))
		methods.file_create(filename_predictions)
		f = open(filename_predictions, 'w')
		f.write(' '.join(str(val_sa) for val_sa in sa_values) + '\n')
		for i in range(len(distinct_records)):
			pred_with = ' '.join(str(predicted_distributions_modes[i][val_sa]) for val_sa in sa_values) + ' '
			pred_without = ' '.join(str(predicted_distributions_modes_w[i][val_sa]) for val_sa in sa_values) + '\n'
			f.write(pred_with+pred_without)
			predictions_modes.append((pred_with+pred_without).split())

		#
		#3- Compute distances
		for d_tag in DISTANCE_TAGS:
			distances = []
			#for every distinct quasi-identifiers tuple, compute the distance between the corresponding prediction distributions
			for j in range(len(predictions_average)):
			    d = distance_compute(predictions_average[j][0:len(sa_values)], predictions_average[j][len(sa_values):], len(sa_values), d_tag)
			    distances.append(d)
			distances_dict_average[d_tag].append(distances)

		for d_tag in DISTANCE_TAGS:
			distances = []
			#for every distinct quasi-identifiers tuple, compute the distance between the corresponding prediction distributions
			for j in range(len(predictions_modes)):
			    d = distance_compute(predictions_modes[j][0:len(sa_values)], predictions_modes[j][len(sa_values):], len(sa_values), d_tag)
			    distances.append(d)
			distances_dict_modes[d_tag].append(distances)

	#4- Plot the relative CDF of the distances  
	for d_tag in DISTANCE_TAGS:
		curves, legend = [], []
		fig = plt.figure()  
		curves = [0 for x in range(len(EPSILON_VALUES))]
		legend = ['eps = ' + str(x) for x in EPSILON_VALUES]
		index = 0
		for i in range(len(distances_dict_average[d_tag])):
		    size = len(distances_dict_average[d_tag][i])
		    yvals = np.arange(1,  size+1)/float(size)
		    curves[index],  = plt.plot(np.sort([x for x in distances_dict_average[d_tag][i]]), yvals, COLOR_LIST[EPSILON_VALUES[i]], label=EPSILON_VALUES[i])
		    index += 1
		plt.legend([x for x in curves], legend, loc=4, fontsize=12, frameon=False)
		#plt.xlim(min([min(x) for x in distances_dict][d_tag]), max([max(x) for x in distances_dict][d_tag]))
		label_x = ''
		if d_tag == 'EMD':
			label_x = 'Earth Mover\'s Distance'
		elif d_tag == 'm_ratio':
			label_x = 'Maximal Ratio'
		plt.xlabel('%s'%label_x, fontsize=14)
		plt.ylabel('Cumulative Relative Frequency', fontsize=14)  
		plt.title('Size=%dK, N=%dK, Average'%(SIZE/1000, ITERATIONS_NB/1000))
		fig = plt.gcf()
		filename = 'results/figures/cdf/%s/average/cdf_DP_%s_S%d_N%d_%s_average.pdf'%(NAME, NAME, SIZE, ITERATIONS_NB, d_tag)
		methods.file_create(filename)
		fig.savefig(filename, bbox_inches='tight')
		#plt.show()
		plt.close(fig)


	for d_tag in DISTANCE_TAGS:
		curves, legend = [], []
		fig = plt.figure()  
		curves = [0 for x in range(len(EPSILON_VALUES))]
		legend = ['eps = ' + str(x) for x in EPSILON_VALUES]
		index = 0
		for i in range(len(distances_dict_modes[d_tag])):
		    size = len(distances_dict_modes[d_tag][i])
		    yvals = np.arange(1,  size+1)/float(size)
		    curves[index],  = plt.plot(np.sort([x for x in distances_dict_modes[d_tag][i]]), yvals, COLOR_LIST[EPSILON_VALUES[i]], label=EPSILON_VALUES[i])
		    index += 1
		plt.legend([x for x in curves], legend, loc=4, fontsize=12, frameon=False)
		#plt.xlim(min([min(x) for x in distances_dict][d_tag]), max([max(x) for x in distances_dict][d_tag]))
		label_x = ''
		if d_tag == 'EMD':
			label_x = 'Earth Mover\'s Distance'
		elif d_tag == 'm_ratio':
			label_x = 'Maximal Ratio'
		plt.xlabel('%s'%label_x, fontsize=14)
		plt.ylabel('Cumulative Relative Frequency', fontsize=14)  
		plt.title('Size=%dK, N=%dK, Modes'%(SIZE/1000, ITERATIONS_NB/1000))
		fig = plt.gcf()
		filename = 'results/figures/cdf/%s/modes/cdf_DP_%s_S%d_N%d_%s_modes.pdf'%(NAME, NAME, SIZE, ITERATIONS_NB, d_tag)
		methods.file_create(filename)
		fig.savefig(filename, bbox_inches='tight')
		#plt.show()
		plt.close(fig)


	print 'Done!'
