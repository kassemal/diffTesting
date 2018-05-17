"""
Methods that read 'Adult' dataset and the related generalization trees. 

'adult.all' file contains the training and testing datasets from http://archive.ics.uci.edu/ml/index.html
#1. age: continuous, range = [17-90]
#2. workclass: 'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay' (7) 
                  (NB: + 'Never-worked' which appears in 7 lines, but all of them contain ? and thus removed ==> nb_catagories = 7) 
#3. final_weight: continuous, range = [13769-1484705]
#4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, 
                  Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. (16)
#5. education_num: continuous, range = [1-16]
#6. marital_status: 'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'. (7)
#7. occupation: 'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving', 
                   'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv', 'Armed-Forces', 
                   'Priv-house-serv'. (14)
#8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. (6)
#9. race: 'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'. (5)
#10. sex: 'Male', 'Female'. (2)
#11. capital_gain: continuous, range = [0-99999]
#12. capital_loss: continuous, range = [0-4356]
#13. hours_per_week: continuous, range = [1-99]
#15. native_country: 'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 
                         'Iran', 'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 
                         'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru', 
                         'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 
                         'Hungary', 'Holand-Netherlands'. (41)
#15. income_level: '<=50K', '>50K' (2)
"""

#!/usr/bin/env python
# coding=utf-8

import utils.utility as ul
import pickle
from pulp import *

#Attribute names as ordered in 'data/adult.data' file 
ATT_NAMES = ['age', 'workclass', 'final_weight', 'education',
             'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
             'native_country', 'income_level']     
#'False' means that the attribute values are continuous or ordinal, and 'True' means that the attribute is categorial 
CATEGORY = [False, True, False, True, False, True, True,  True,  True,  True,  False,  False,  False,  True,  True]


def read():
    """
    Read "Adult" dataset from 'data/adult.data'
    """
    #initialize
    nb_attributes = len(ATT_NAMES)
    data, numeric_dict = [], []
    for i in range(nb_attributes):
        if CATEGORY[i] is False:
       		numeric_dict.append(dict()) #dictionary for continuous attributes
    #read data
    data_file = open('data/adult.all', 'rU')
    for line in data_file:
        line = line.strip()
        #remove empty and incomplete lines, Only 45222 records will remain 
        if len(line) == 0 or '?' in line:
            continue
        #remove spaces
        line = line.replace(' ', '')
        #split
        temp = line.split(',')
        #append to table $data
        data.append(temp)
        #keep a dictionary of continuous attributes
        index = 0
        for i in range(nb_attributes):
            if CATEGORY[i] is False:
                try:
                    numeric_dict[index][temp[i]] += 1
                except:
                    numeric_dict[index][temp[i]] = 1
                index += 1
    #pickle numeric attributes and get NumRange
    index = 0
    for i in range(nb_attributes):
        if CATEGORY[i] is False:
            static_file = open('data/adult_' + ATT_NAMES[i] + '_static.pickle', 'wb')
            sort_value = list(numeric_dict[index].keys())
            sort_value.sort(cmp=ul.cmp_str)
            pickle.dump((numeric_dict[index], sort_value), static_file)
            static_file.close()
            index += 1
    return data


def tree_get(attributes):
    """
    For every attribute in $attributes, read the related tree from data/adult_*.txt, 
    and store it in $att_trees.
    Moreover, store the the paths to leaves in the case of categorical attribute. 
    """
    prefix = 'data/adult_'
    att_trees = []
    paths_to_leaves = []
    for i in range(len(attributes)):
        if CATEGORY[ATT_NAMES.index(attributes[i])]:
            att_tree,  path_to_leaf = ul.read_tree_file(attributes[i], prefix)
            att_trees.append(att_tree)
            paths_to_leaves.append(path_to_leaf)
        else:
            att_tree = ul.read_pickle_file(attributes[i], prefix)
            att_trees.append(att_tree)
    return att_trees,  paths_to_leaves
    
    

