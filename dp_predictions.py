"""
=====================================================================================
Apply Differential Inference Testing on the results of differentially private queries
=====================================================================================

Differential privacy is different from generalization techniques, like k-anonymity and l-diversity, 
in that it does not explicitly publish microdata. Instead, a user makes a query on the data, the 
server then computes the exact answer and adds some noise before replying. 
We consider count queries, that is in the form of "What is the number of individuals in the data that 
satisfy certain conditions". Moreover, we assume histogram queries, that is only a single round of 
querying without any kind of adaptive querying in response to previous results. 

The idea is to use the queries results ("the noised counts") to build a classifier that takes as input 
a tuple of quasi-identifiers and outputs a prediction distribution on the sensitive attribute values. 

The considered classifier outputs, for a given quasi-identifier tuple q from a set Q, 
a prediction distribution Pr[S|Q=q] where S is the set of all possible sensitive values.  
As a noise, random values from a Laplacian distribution are considered. 

In more details, do the following: 
1- Read full dataset from 'data/$data_filename', pick up only the attributes in $ATT_QI + 
   $ATT_SA, and randomly select $SIZE records. Moreover, write the selected dataset into $filename.
   Then, quantize attributes in $ATT_CONTINUOUS. 
   Note that, to read an already selected dataset one has to set $IS_SELECT_NEW to False.  

2- For every distinct record (q,s), generate $ITERATIONS_NB prediction distributions while considering 
   the entire dataset (i.e. while this record exists). Then take their average in order to obtain 
   a distribution P. After removing this record from the dataset, repeat the same process in order 
   to obtain a probability distribution P_w. Note that, technically speaking, considering the 
   average over $ITERATIONS_NB will result in level of privacy $ITERATIONS_NB*epsilon where epsilon 
   is the exponent used in the Laplacian noise. 
   At the end, write distributions into a file, a file per record

   Repeat steps 2 and 3 for every privacy parameter p_value in $P_VALUES.
""" 

# !/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import copy
import random
import time 
#
from utils import methods


def counts_q_compute(dataset, distinct_records): 
	"""
	Compute the count inside $dataset for every possible pair (q,s). 
	"""
	counts, sa_values = [], []
	#obtain sensitive attribute values 
	sa_values = list(set([record[-1] for record in dataset]))
	#compute counts
	for record in distinct_records:
		q_tuple = record[:-1]
		q_counts = {}
		for val_s in sa_values:
			q_counts[val_s] = dataset.count(q_tuple+[val_s])
		counts.append(q_counts)
	return counts, sa_values


def proba_distribution_compute(count_i, sa_values, epsilon): 
	"""
	Compute the probability Pr[s|q] for every pair (q,s) that appears inside the dictionary $count_i.  
	Pr[s|q] = Pr[s, q] / Pr[q], where Pr[s, q] = count(q, s)/sum_q,s' count(q, s') and Pr[q] = sum_s' Pr[s', q]. 
    Thus, Pr[s|q] = count(q, s) / sum_s' count(q, s'). 
	"""
	pr_cond_s_q = {}
	pr_q = 0.0 #total
	#compute joint prob
	for val_s in sa_values:
		C_qs = 1.0 + max(0, count_i[val_s] + np.random.laplace(loc=0.0, scale=1.0/epsilon))
		pr_cond_s_q[val_s] = C_qs #/sum (omitted)
		pr_q += pr_cond_s_q[val_s]
	#conditional probability
	for val_s in sa_values:
		pr_cond_s_q[val_s] /= pr_q
	return pr_cond_s_q 


def predictions_write(predictions, predictions_w, sa_values, filename, mode='w'):
    """
    Write $predictions and $predictions_w into $filename. 
    A line per record: $predictions[i] + $predictions_w[i]
    """
    if len(predictions) == len(predictions_w):
        methods.file_create(filename)
        f = open(filename, mode)
        for i in range(len(predictions)):
            pred_with = ' '.join(str(predictions[i][val_s]) for val_s in sa_values) + ' '
            pred_without = ' '.join(str(predictions_w[i][val_s]) for val_s in sa_values) + '\n'
            f.write(pred_with+pred_without)
        f.close()
    else:
        print 'The two input lists don\'t have the same length'
    return 


def par_initialize(name):
	# #adult 
	ATT_QI_adult = ['age', 'education', 'marital_status', 'hours_per_week', 'native_country'] #quasi-identifier attributes
	ATT_SA_adult = 'occupation' #sensitive attribute
	IS_CAT_adult = [False, True, True, False, True] #specifies which attributes are categorical (True) and which are continuous (False). Only required when IS_SELECT_NEW=False
	ATT_QUANTIZE_adult = ['age', 'hours_per_week'] 
	QUANTILES_adult = [
	            [[0,25],[26,40],[41,60],[61,75],[76,90]],  #age
	            [[0,20],[21,40],[41,60],[61,80],[81,100]]  #hours_per_week
	            ]
	#internet
	ATT_QI_internet = ['age', 'education_attainment', 'major_occupation', 'marital_status', 'race']  
	ATT_SA_internet = 'household_income'  
	IS_CAT_internet = [False, True, True, True, True]
	ATT_QUANTIZE_internet = ['age'] 
	QUANTILES_internet = [
	            [[0,20],[21,40],[41,60],[61,80]] #age 
	            ]
	#          
	if name == 'adult':
		return ATT_QI_adult, ATT_SA_adult, IS_CAT_adult, ATT_QUANTIZE_adult, QUANTILES_adult
	elif name == 'internet':
		return ATT_QI_internet, ATT_SA_internet, IS_CAT_internet, ATT_QUANTIZE_internet, QUANTILES_internet


#
NAME = 'adult' #'adult' 'internet' #name of the dataset
SIZE = 1000            #size of the dataset to consider
IS_SELECT_NEW  = False  #True is to select new data
ITERATIONS_NB  = 1000  #number of predictions to make for each record
EPSILON_VALUES = [0.1, 0.5, 1.0]  #noise parameter to consider
#


if __name__ == '__main__':

	#pdb.set_trace()
	#1- Import data
	print 'Read data ...'
	ATT_QI, ATT_SA, IS_CAT, ATT_QUANTIZE, QUANTILES = par_initialize(NAME)
	dataset = [] #a dataset is a table
	filename = 'results/selected_DATA/%s/selected_%s_S%s'%(NAME, NAME, str(SIZE))
	#obtain the dataset 
	if IS_SELECT_NEW:
	    dataset, _ = methods.data_import(name=NAME, size=SIZE, qi_list=ATT_QI, sa=ATT_SA, filename=filename)
	else:#read already selected data
	    dataset = methods.data_read(filename=filename)
	#quantize attributes in $ATT_QUANTIZE
	if all(x in ATT_QI for x in ATT_QUANTIZE):
		methods.data_quantize(dataset, [ATT_QI.index(x) for x in ATT_QUANTIZE], QUANTILES) #directly modifies $dataset
	#obtain the list of distinct records
	#distinct_records = list(set(dataset))
	distinct_records = []
	for i in range(0, len(dataset)):
		if dataset[i] not in distinct_records:
			distinct_records.append(dataset[i])
	print 'Number of distinct records: %d'%len(distinct_records)

	#2- Build classifier and make predictions
	#obtain counts for every possible pair (q, s).  
	counts, sa_values = counts_q_compute(dataset, distinct_records) 
	#for every $epsilon, obtain the related prediction distributions
	for epsilon in EPSILON_VALUES:
		time_start = time.time()
		print 'Predictions for epsilon %s ...'%str(epsilon)
		for i, d_record in enumerate(distinct_records): 
			counts_i_w = copy.deepcopy(counts[i])
			counts_i_w[d_record[-1]] -= 1  #decrement count by 1
			prediction_distribution_i, prediction_distribution_i_w = [], []
			#obtain two prediction distributions for current record
			for iteration in range(ITERATIONS_NB):
				#model M, with the record
				distribution_i = proba_distribution_compute(counts[i], sa_values, epsilon)
				for s in sa_values:
					distribution_i[s] = float('%.5f'%distribution_i[s]) 
				prediction_distribution_i.append(distribution_i)
				#model M_w, without the record
				distribution_i_w = proba_distribution_compute(counts_i_w, sa_values, epsilon)
				for s in sa_values:
					distribution_i_w[s] = float('%.5f'%distribution_i_w[s]) 
				prediction_distribution_i_w.append(distribution_i_w)
			#write/append obtained prediction into $predictions_filename
			predictions_filename = 'results/predictions/%s/S%s/DP/E%s/predictions_%s_DP_p%s_r%d'%(NAME, str(SIZE), str(epsilon), NAME, str(epsilon), i+1)
			predictions_write(prediction_distribution_i, prediction_distribution_i_w, sa_values, predictions_filename, mode='a')
			#print('record %d'%(i+1))
		#print out the computation time in seconds
		print 'Computation time in seconds:' +  str(time.time() - time_start) 		
	print 'Number of classes:%d'%len(sa_values)
	print 'Done!'
	