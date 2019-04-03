"""
===================================================================================================
Apply Differential Testing on generalized anonymization techniques like K-anonymity and L-diversity
===================================================================================================

1- Read full dataset from 'data/$data_filename', pick up only the attributes in $ATT_QI + 
   $ATT_SA, and randomly select $SIZE records. Moreover, write the selected dataset into $filename.
   Then, quantize attributes in $ATT_CONTINUOUS. 
   Note that, to read an already selected dataset one has to set $IS_SELECT_NEW to False.

2- Anonymize and encode the full dataset using the binary encoding (TF). Then use it to train a machine learning model (M). 

3- For every distinct record (q,s), use the full-data model (M) to obtain a prediction distribution (P) on the domain 
   of possible values of s given q as an input. Then remove this record from the dataset, and repeat the same process: 
   anonymize and encode the data again, train a new machine learning model (M_w) and then use M_i to obtain a prediction 
   distribution (P_w)on the domain of possible values of s given q as an input.

   Repeat steps 2 and 3 for every privacy parameter p_value in $P_VALUES.

This implementation considers 
Basic Mondrian K-Anonymity: https://github.com/qiyuangong/Basic_Mondrian
and Mondrian L-Diversity: https://github.com/qiyuangong/Mondrian_L_Diversity

"""

# !/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import itertools
import time 
import copy
#
from utils import methods

NAME = 'adult' #name of the dataset: 'adult', 'internet'
#adult 
ATT_QI = ['age', 'education', 'marital_status', 'hours_per_week', 'native_country'] #quasi-identifier attributes
ATT_SA = 'occupation' #sensitive attributes
IS_CAT = [False, True, True, False, True]#specifies which attributes are categorical (True) and which are continue (False). Only required when IS_SELECT_NEW=False
ATT_QUANTIZE = [] #['age', 'hours_per_week'] 
QUANTILES = [
            [[0,30],[31,45],[46,60],[61,75],[76,100]], #age 
            [[0,20],[21,40],[41,60],[61,80],[81,100]]  #hours_per_week
            ]
#internet
# ATT_QI =  ['age', 'education_attainment', 'major_occupation', 'marital_status', 'race'] #quasi-identifier attributes
# ATT_SA = 'household_income' #sensitive attributes
# IS_CAT = [False, True, True, True, True]#specifies which attributes are categorical (True) and which are continue (False). Only required when IS_SELECT_NEW=False
# ATT_QUANTIZE = [] #['age'] 
# QUANTILES = [
#              [[0,20],[21,40],[41,60],[61,80]] #age #internet 
#             ]
#
SIZE = 1000   #size of the dataset to consider
IS_SELECT_NEW = True #True is to select new data
ANON_TECH = 'k_anonymity' #anonymization technique 'k_anonymity', 'l_diversity'
MLA = 'BNB' #Machine Learning algorithm
P_VALUES = [2, 4, 8, 16] # Parameter used for anonymization


if __name__ == '__main__':

	#pdb.set_trace()
	#1- Import data
	print('Read data ...') 
	dataset = [] # a dataset is a table
	filename = 'results/selected_DATA/%s/selected_%s_S%s'%(NAME, NAME, str(SIZE))
	#obtain the dataset 
	if IS_SELECT_NEW:
		dataset, is_cat = methods.data_import(name=NAME, size=SIZE, qi_list=ATT_QI, sa=ATT_SA, filename=filename)
	else:#read already selected data
		dataset = methods.data_read(filename=filename)
		is_cat = IS_CAT
	#quantize attributes in $ATT_QUANTIZE
	if all(x in ATT_QI for x in ATT_QUANTIZE):
		methods.data_quantize(dataset, [ATT_QI.index(x) for x in ATT_QUANTIZE], QUANTILES) #directly modifies $dataset
	#convert $ATT_QUANTIZE status from continuous into categorical 
	for j in range(len(ATT_QI)):
		if ATT_QI[j] in ATT_QUANTIZE:
			is_cat[j] = True
	#obtain the list of distinct records
	dataset.sort()
	distinct_records = list(record for record,_ in itertools.groupby(dataset))
	# distinct_records = []
	# for i in range(0, len(dataset)):
	# 	if dataset[i] not in distinct_records:
	# 		distinct_records.append(dataset[i])

	#Iterate for every $p_value in $P_VALUES
	for p_value in P_VALUES:
		time_start = time.time()
		print('Anonymization parameter: %d'%p_value)
		#2- Anonymize and encode full dataset
		#get the related hierarchy trees
		att_trees, paths_to_leaves = methods.tree_get(ATT_QI, is_cat, prefix = 'data/%s_'%NAME)
		#anonymize full dataset
		temp_dataset = copy.deepcopy(dataset)#method data_anonymize modifies the content of $temp_dataset
		anonymized_dataset, eval_result = methods.data_anonymize(temp_dataset, att_trees, ANON_TECH, p_value)
		#obtain QI attributes' values 
		att_values = []
		for j in range(len(dataset[0])-1):
			att_values.append(list(set([record[j] for record in dataset])))
		#
		#input record -- to be used as input for the machine learning models
		input_record = (len(dataset[0])-1)*[1]
		classes_ref = []
		#3-Iterate for every distinct record inside the dataset
		predicted_distributions, predicted_distributions_w = [], []
		for i in range(len(distinct_records)):
			#encode anonymized (full) dataset 
			encoded_dataset = methods.data_encode_TF(anonymized_dataset, distinct_records[i], paths_to_leaves, is_cat)
			#generate a model M using full dataset
			model_M = methods.model_train(encoded_dataset, MLA)
			#get the order of the classes
			classes = model_M.classes_.tolist()
			#make sure that classes order is coherent
			if len(classes_ref) == 0:
			    classes_ref = copy.deepcopy(classes) 
			elif classes != classes_ref:
				print('Classes order error!')
				break
			#obtain a prediction distribution for distinct record i from the full-data model
			predicted_distributions.append(model_M.predict_proba([input_record]).tolist()[0])
			##### Remove record $i, then generate model M_i #####
			#remove record $i
			index = dataset.index(distinct_records[i])
			temp_dataset_w = copy.deepcopy(dataset)
			temp_dataset_w.pop(index)
			#re-anonymize datset
			anonymized_dataset_w, eval_result = methods.data_anonymize(temp_dataset_w, att_trees, ANON_TECH, p_value)
			#obtain QI attributes' values again (as some values may disappear from the dataset)
			att_values_w = []
			for j in range(len(dataset[0])-1):
				att_values_w.append(list(set([record[j] for record in temp_dataset_w])))
			#encode anonymized dataset $anonymized_dataset_w
			encoded_dataset_w = methods.data_encode_TF(anonymized_dataset_w, distinct_records[i], paths_to_leaves, is_cat)
			#generate a model M_w using $encoded_dataset_w
			model_M_w = methods.model_train(encoded_dataset_w, MLA)
			#make sure that both models M and M_w adopt the same classes order
			if model_M_w.classes_.tolist() != classes_ref:
				print('Classes order error!')
				break
			#obtain a prediction distribution for distinct record i from the model M_w
			predicted_distributions_w.append(model_M_w.predict_proba([input_record]).tolist()[0])
		#print out the computation time in seconds
		print('Computation time in seconds: ' + str(time.time() - time_start))        
		#write obtained prediction into predictions_file, a record per line
		filename_predictions = 'results/predictions/%s/S%s/%s/%s/TF/predictions_%s_%s_p%s'%(NAME, str(SIZE), ANON_TECH, MLA, NAME, ANON_TECH, str(p_value))
		methods.table_write([predicted_distributions[l]+predicted_distributions_w[l] for l in range(len(predicted_distributions))], filename_predictions)
	print('Number of classes:%d'%len(classes_ref))
	print('Done!')