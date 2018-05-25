"""
===================================================================================================
Apply Differential Testing on generalized anonymization techniques like K-anonymity and L-diversity
===================================================================================================

1- Read full dataset from 'data/$data_filename', pick up only the attributes in $ATT_QI + 
   $ATT_SA, and randomly select $SIZE records. Moreover, write the selected dataset into $filename.
   Then, quantize attributes in $ATT_CONTINUOUS. 
   Note that, to read an already selected dataset one has to set $IS_SELECT_NEW to False.

2- Anonymize and encode the full dataset. Then use it to train a machine learning model (M). 

3- For every distinct record (q,s), use the full-data model (M) to obtain a prediction distribution (P) on the domain 
   of possible values of s given q as an input. Then remove this record from the dataset, and repeat the same process: 
   anonymize and encode the data again, train a new machine learning model (M_w) and then use M_i to obtain a prediction 
   distribution (P_w)on the domain of possible values of s given q as an input.

4- For every record, compute a distance between the corresponding distributions P and P_w.
   Repeat steps 2, 3 and 4 for every distance computation method in $DISTANCE_TAGS 
   and for every privacy parameter p_value in $P_VALUES. 

5- Plot the cumulative relative frequency of the computed distances for all epsilon values in one graph. 
   A graph is plotted for each distance method. 

This implementation considers 
Basic Mondrian K-Anonymity: https://github.com/qiyuangong/Basic_Mondrian
and Mondrian L-Diversity: https://github.com/qiyuangong/Mondrian_L_Diversity

To encode data after anonymization, one-hot encoding is used.
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

#adult 
ATT_QI =  ['age', 'education', 'sex', 'native_country'] #quasi-identifier attributes
ATT_SA = 'occupation' #sensitive attributes
IS_CAT = [False, True, True, True]#specifies which attributes are categorical (True) and which are continue (False). Only required when IS_SELECT_NEW=False
#internet
#ATT_QI =  ['age', 'education_attainment', 'major_occupation', 'marital_status', 'race'] #quasi-identifier attributes
#ATT_SA = 'household_income' #sensitive attributes
#IS_CAT = [False, True, True, True, True]#specifies which attributes are categorical (True) and which are continue (False). Only required when IS_SELECT_NEW=False
ATT_QUANTIZE = []#['age'] 
QUANTILES = [
            [[0,25],[26,50],[51,75],[75,100]] #age
            ]
NAME = 'adult' #name of the dataset: 'adult', 'internet'
SIZE = 10000   #size of the dataset to consider
IS_SELECT_NEW = True #True is to select new data
MLA = 'BNB' #Machine Learning algorithm
ANON_TECH = 'l_diversity' #anonymization technique 'k_anonymity', 'l_diversity'
P_VALUES = [2, 5] #[1, 2, 5, 7, 24, 32, 64] [1, 2, 3, 4, 5, 6, 7] # Parameter used for anonymization
COLOR_LIST  = {1:'b', 2:'y', 5:'r', 7:'g', 24:'c', 32:'m', 64:'k'} #{1:'b', 2:'y', 3:'r', 4:'g', 5:'c', 6:'m', 7:'k'}
DISTANCE_TAGS = ['EMD', 'm_ratio'] #distances to consider: EMD for Earth Mover Distance, and m_ratio for Maximal ratio


if __name__ == '__main__':

    #pdb.set_trace()
    time_start = time.time()
    #1- Import data
    print 'Read data ...'
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
    #initiate distance dictionary
    distances_dict = dict()
    for d_tag in DISTANCE_TAGS:
        distances_dict[d_tag] = []
    #Iterate for every $p_value in $P_VALUES
    for p_value in P_VALUES:
        print 'Anonymization parameter: %d'%p_value
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
        #encode anonymized (full) dataset   
        encoded_dataset = methods.data_encode(anonymized_dataset, att_values, paths_to_leaves, is_cat)
        #encode unanonymized (distinct) records -- to be used as input for the machine learning models
        encoded_distinct_records = methods.data_encode(distinct_records, att_values, paths_to_leaves, is_cat)
        #generate a model M using full dataset
        model_M = methods.model_train(encoded_dataset, MLA)
        #get the order of the classes
        classes_ref = model_M.classes_.tolist()
        #
        #Iterate for every distinct record inside the dataset
        predicted_distributions, predicted_distributions_w = [], []
        for i in range(len(encoded_distinct_records)):
            #obtain a prediction distribution for distinct record i from the full-data model
            predicted_distributions.append(model_M.predict_proba([encoded_distinct_records[i][:-1]]).tolist()[0])
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
            encoded_dataset_w = methods.data_encode(anonymized_dataset_w, att_values_w, paths_to_leaves, is_cat)
            #re-encode the input record $distinct_records[i]
            input_record = methods.data_encode([distinct_records[i]], att_values_w, paths_to_leaves, is_cat)
            #generate a model M_w using $encoded_dataset_w
            model_M_w = methods.model_train(encoded_dataset_w, MLA)
            #obtain a prediction distribution for distinct record i from the model M_w
            if model_M_w.classes_.tolist() == classes_ref:#make sure that both models M and M_w adopt the same classes order
                predicted_distributions_w.append(model_M_w.predict_proba([input_record[0][:-1]]).tolist()[0])
            else:
                print 'Classes order error!'
                break
        #print out the computation time in seconds
        print 'Computation time in seconds:',  time.time() - time_start         #write obtained prediction into predictions_file, a record per line
        filename_predictions = 'results/predictions/%s/%s/predictions_%s_S%s_p%s'%(ANON_TECH, NAME, NAME, str(SIZE), str(p_value))
        methods.table_write([predicted_distributions[l]+predicted_distributions_w[l] for l in range(len(predicted_distributions))], filename_predictions)
        #
        print len(list(set([x[-1] for x in distinct_records])))
        #3- Compute distances
        for d_tag in DISTANCE_TAGS:
            distances = []
            #for every distinct record, compute the distance between the corresponding prediction distributions
            for i in range(len(predicted_distributions)):
                d = methods.distance_compute(predicted_distributions[i], predicted_distributions[i], len(list(set([x[-1] for x in distinct_records]))), d_tag)
                distances.append(d)
            distances_dict[d_tag].append(distances)

    #4- Plot the relative CDF of the distances  
    for d_tag in DISTANCE_TAGS:
        filename = 'results/figures/cdf/%s/%s/cdf_DP_%s_S%d_%s.pdf'%(ANON_TECH, NAME, NAME, SIZE, d_tag)
        methods.cdf_plot(distances_dict[d_tag], P_VALUES, COLOR_LIST, d_tag, filename)

    print 'Done!'