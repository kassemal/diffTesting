"""
==============================================================================================================
Apply Differential Inference Testing on the results of differentially private queries (Naive Bayes classifier)
==============================================================================================================

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
1- Read full dataset from 'data/$data_filename', pick up only the attributes in $ATT_QI + 
   $ATT_SA, and randomly select $SIZE records. Moreover, write the selected dataset into $filename.
   Then, quantize attributes in $ATT_CONTINUOUS. 
   Note that, to read an already selected dataset one has to set $IS_SELECT_NEW to False.  

2- For every distinct record (q,s), make $ITERATIONS_NB prediction distributions while considering 
   the entire dataset (i.e. while this record exists). Then take their average in order to obtain 
   a distribution P. After removing this record from the dataset, repeat the same process in order 
   to obtain a probability distribution P_w. Note that, technically speaking, considering the 
   average over $ITERATIONS_NB will result in level of privacy $ITERATIONS_NB*epsilon where epsilon 
   is the exponent used in the Laplacian noise. 

   Repeat steps 2 and 3 for every privacy parameter p_value in $P_VALUES.
""" 

# !/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
#import itertools
import random
import time 
#
from utils import methods


def conditional_probabilities_compute(counts, epsilon): 
	"""
	Compute the probability Pr[q|s] for every pair (q,s) whose 
	frequencies are inside dictionary $counts.  

	Pr[q|s] = C_qs/(sum_q C_qs)
	where C_qs = 1 + max(0, counts[j][q][s] + noise) with j is the index of related attribute 

	Moreover, compute the probability Pr[s] for every SA value s which appears 
 	inside dictionary $counts.  
	"""
	overall_total = 0.0
	pr_SA = dict()
	cond_probabilities = [dict() for x_dict in counts]
	total_per_QI = [dict() for x_dict in counts]
	for j, dictionary_j in enumerate(counts):
	    for key_q, dict_counts in dictionary_j.iteritems():
			for key_s, val_count in dict_counts.iteritems():
				#add noise to the count value of the pair (q,s) and compute C_qs
				#sensitivity = m/epsilon) where m is the number of counts disclosed (equal to the number of QI) 
				C_qs = 1.0 + max(0, val_count + np.random.laplace(loc=0.0, scale=float(len(counts))/epsilon)) 
				#insert the value of C_qs in $cond_probabilities list
				try:
					cond_probabilities[j][key_q][key_s] = C_qs
				except KeyError:
					cond_probabilities[j][key_q]  = {key_s: C_qs}
				#add the value of C_qs to the related QI total
				try:
					total_per_QI[j][key_s] += C_qs
				except KeyError:
					total_per_QI[j][key_s]  = C_qs
				######### Pr[s] ############
				try:
					pr_SA[key_s] += C_qs
				except KeyError:
					pr_SA[key_s]  = C_qs
				overall_total += C_qs
	#normalize the values to obtain the related conditional probabilities 
	for j, dictionary_j in enumerate(cond_probabilities):
	    for key_q, dict_counts in dictionary_j.iteritems():
		    for key_s in dict_counts.iterkeys():
					cond_probabilities[j][key_q][key_s] /= total_per_QI[j][key_s]
	#normalize the values to obtain the related probabilities Pr[s]
	for key_s, value in pr_SA.iteritems():
	        pr_SA[key_s] = value/overall_total
	return cond_probabilities, pr_SA


# def sa_values_proba_compute(counts, epsilon):
# 	"""
# 	Compute the probability Pr[s] for every SA value s which appears 
# 	inside dictionary $counts.  

# 	Pr[s] = C_s/(sum_s C_s)
# 	where C_s = 1 + max(0, count_s + noise) 
# 	"""
# 	probabilities_s = dict()
# 	#compute the counts of SA values by summing up the related entries inside $counts
# 	if len(counts) != 0:
# 		for key_q, dict_counts in counts[0].iteritems():
# 			for key_s, val_count in dict_counts.iteritems():
# 				try:
# 					probabilities_s[key_s] += val_count
# 				except KeyError:
# 					probabilities_s[key_s]  = val_count
# 		##### compute probabilities of sensitive attribute values ##### 
# 		total_s = 0.0
# 		for key_s, val_count_s in probabilities_s.iteritems():
# 			#add noise to SA value counts
# 			probabilities_s[key_s] = 1.0 + max(0, val_count_s + np.random.laplace(loc=0.0, scale=1.0/epsilon))
# 			#add to the total
# 			total_s += probabilities_s[key_s]
# 		#normalize probabilities
# 		for key_s in probabilities_s.keys():
# 			probabilities_s[key_s] /= total_s
# 	return probabilities_s


def proba_distributions_compute(distinct_records, counts, epsilon): 
	"""
	Obtain a prediction distribution on the domain of sensitive attribute for every 
	record inside $distinct_records based on $counts, after adding some Laplacian noise randomly selected from L(0, 1/epsilon). 

	For a given QI tuple (q_1, ..., q_m), a prediction distribution composed of the probabilities Pr[s|(q_1, ..., q_m)]
	for every possible value of sensitive attribute s. 

	Pr[s|(q_1, ..., q_m)] is proportional to (1/C).Pr[s].Pr[q_1|s]. ... .Pr[q_m|s] under the naive conditional 
	independence assumption, where C is a constant equal to the probability of the evidence.  
	"""
	distributions = [dict() for x in distinct_records]
	#compute probability Pr[q|s] for every possible pair (q, s) and Pr[s] for every possible SA value s
	cond_probabilities, sa_probabilities = conditional_probabilities_compute(counts, epsilon) 
	#compute probability Pr[s] for every possible SA value s
	#sa_probabilities = sa_values_proba_compute(counts, epsilon)
	#compute probabilities Pr[s|(q_1, ..., q_m)]
	for i in range(len(distinct_records)):
		total_i = 0.0 #total per record
		for key_s, val_Proba_s in sa_probabilities.iteritems():
			proba_s_quasi = val_Proba_s
			for j, val_q in enumerate(distinct_records[i][:-1]):
				proba_s_quasi *= cond_probabilities[j][val_q][key_s]
			distributions[i][key_s] = proba_s_quasi
			total_i += proba_s_quasi
		#normalize
		for key_s, val_proba in distributions[i].iteritems():
			distributions[i][key_s] = val_proba/total_i
	return distributions 

#
NAME = 'adult' #name of the dataset
#adult 
ATT_QI = ['age', 'education', 'marital_status', 'hours_per_week', 'native_country'] #quasi-identifier attributes
ATT_SA = 'occupation' #sensitive attributes
IS_CAT = [False, True, True, False, True]#specifies which attributes are categorical (True) and which are continue (False). Only required when IS_SELECT_NEW=False
ATT_QUANTIZE = ['age', 'hours_per_week'] 
QUANTILES = [
            [[0,25],[26,40],[41,60],[61,75],[76,90]], #age
            [[0,20],[21,40],[41,60],[61,80],[81,100]]  #hours_per_week
            ]
#internet
#ATT_QI =  ['age', 'education_attainment', 'major_occupation', 'marital_status', 'race'] #quasi-identifier attributes
#ATT_SA = 'household_income' #sensitive attributes
#IS_CAT = [False, True, True, True, True]
#ATT_QUANTIZE = ['age'] 
#QUANTILES = [
#            [[0,20],[21,40],[41,60],[61,80]] #age #internet 
#            ]
SIZE = 10000          #size of the dataset to consider
IS_SELECT_NEW = False #True is to select new data
ITERATIONS_NB = 10000    #number of predictions to make for each record
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0] #noise parameter to consider

if __name__ == '__main__':

	#pdb.set_trace()
	#1- Import data
	print 'Read data ...'
	dataset = [] # a dataset is a table
	filename = 'results/selected_DATA/%s/fixed_selected_%s_S%s'%(NAME, NAME, str(SIZE))
	#obtain the dataset 
	if IS_SELECT_NEW:
	    dataset, _ = methods.data_import(name=NAME, size=SIZE, qi_list=ATT_QI, sa=ATT_SA, filename=filename)
	else:#read already selected data
	    dataset = methods.data_read(filename=filename)
	#quantize attributes in $ATT_QUANTIZE
	if all(x in ATT_QI for x in ATT_QUANTIZE):
		methods.data_quantize(dataset, [ATT_QI.index(x) for x in ATT_QUANTIZE], QUANTILES) #directly modifies $dataset

	#2- Build classifier and make predictions
	#obtain counts for every pair (q_j, s) in the $dataset. Note that a record of $dataset has the form (q_1, ..., q_m, s) 
	counts, sa_values = methods.counts_compute(dataset) # $counts is a list of dictionaries of dictionaries: [{val_q:{val_s:count}} for j=1,..,m]
	#obtain the list of distinct records
	#dataset.sort()
	#distinct_records = list(record for record,_ in itertools.groupby(dataset))
	distinct_records = []
	for i in range(0, len(dataset)):
		if dataset[i] not in distinct_records:
			distinct_records.append(dataset[i])
	#print len(distinct_records)
	#
	#for every $epsilon obtain the related prediction distributions
	for epsilon in EPSILON_VALUES:
		time_start = time.time()
		print 'Predictions for epsilon %s ...'%str(epsilon)
		predicted_distributions_average, predicted_distributions_average_w = [], []
		##### Obtain prediction distributions from the full-data model M for all (distinct) records at the same time #####
		#run $ITERATIONS_NB iterations
		for iteration in range(ITERATIONS_NB):
			#obtain a set of distributions: a distribution for each distinct record
			distributions = proba_distributions_compute(distinct_records, counts, epsilon)
			#add $distributions to the total (in order to obtain the average)
			for i in range(len(distinct_records)):
				try:
					for key_s in distributions[i].keys():
						predicted_distributions_average[i][key_s] += distributions[i][key_s]
				except IndexError:
					predicted_distributions_average.append(distributions[i])			
		#obtain the average  
		for i in range(len(distinct_records)): 
			for key_s in predicted_distributions_average[i].keys():
				predicted_distributions_average[i][key_s] /= ITERATIONS_NB
		##### Obtain prediction distributions from the model M_i based on the dataset after removing the record i #####
		predicted_distributions_average_w = []
		#iterate for every distinct record
		for i, record_i in enumerate(distinct_records):
			#decrement the counts of the related pairs (q,s), each, by 1 -- remove the record
			for j, val_q in enumerate(record_i[:-1]): 
				counts[j][val_q][record_i[-1]] -= 1
			#perform $ITERATIONS_NB for the record $record_i
			for iteration in range(ITERATIONS_NB):
				distribution_record_i = proba_distributions_compute([record_i], counts, epsilon) 
				#add $distribution_record_i to the total
				try:
					for key_s in distribution_record_i[0].keys():
						predicted_distributions_average_w[i][key_s] += distribution_record_i[0][key_s]
				except IndexError:
					predicted_distributions_average_w.append(distribution_record_i[0])
			#increment the related pairs (q,s) to re-obtain the original counts
			for j, val_q in enumerate(record_i[:-1]): 
				counts[j][val_q][record_i[-1]] += 1
		#obtain the average  
		for i in range(len(distinct_records)): 
			for key_s in predicted_distributions_average_w[i].keys():
				predicted_distributions_average_w[i][key_s] /= ITERATIONS_NB
		#print out the computation time in seconds
		print 'Computation time in seconds:',  time.time() - time_start 
		#write obtained prediction into predictions_file, a record per line 
		filename_predictions = 'results/predictions/%s/S%s/DP/N%d/predictions_%s_DP_p%s'%(NAME, str(SIZE), ITERATIONS_NB, NAME, str(epsilon))
		methods.predictions_write(predicted_distributions_average, predicted_distributions_average_w, sa_values, filename_predictions)
	print 'Number of classes:%d'%len(sa_values)
	print 'Done!'
