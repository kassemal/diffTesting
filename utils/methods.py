"""
Methods related to data reading, data writing, data manipulation, and computations.   
"""


# !/usr/bin/env python
# coding=utf-8
from pulp import *
import numpy as np
import random
#
import adult_data
import internet_data


def attributes_filter(dataset, attributes, att_names):
    """
    Pick up from $dataset only the attributes whose names are in $att_names. 
    Note that, this method assumes that $dataset contains all the attributes from ($name)_data.ATT_NAMES
    """
    #obtain attribute indices
    att_indices = []
    for att in attributes:
        att_indices.append(att_names.index(att))
    #obtain related columns 
    res_dataset = []
    for record in dataset:
        temp = []
        for j in att_indices:
            temp.append(record[j])
        res_dataset.append(temp)
    return res_dataset


def records_select(dataset, size):
    """
    Take a table $dataset and selects only $size rows from it.
    """
    #Randomly select $size lines from data, if size < len(data). Otherwise, take all the data.
    counter = 0
    selected_dataset = []
    if size < len(dataset):
        counter = 0
        while(counter < size):
            i = random.choice(range(len(dataset)))
            selected_dataset.append(dataset.pop(i))
            counter += 1
    else: # if size > len(dataset), take all data
        selected_dataset = dataset
    return selected_dataset


def data_import(is_select_new, name, size, qi_list, sa, filename):
    """
    Import a dataset from a file: 
    either read already prepared data from $filename if $is_select_new=False, or 
    select $size records from dataset $name while considering only $qi_list and $sa_list if $is_select_new=True. 
    """
    dataset = []
    if is_select_new == True:#select new data
        if name == 'adult':
            dataset = adult_data.read() #get full Adult dataset
            att_names = adult_data.ATT_NAMES
        elif name == 'internet':
            dataset = internet_data.read() #get full Internet Usage dataset
            att_names = internet_data.ATT_NAMES
        else:
            print 'The dataset "%s" doesn\'t exist!'%name
            return []
        #pick up $qi_list+$sa attributes
        dataset = attributes_filter(dataset, qi_list + [sa], att_names) 
        #select $size records
        dataset = records_select(dataset, size)
        #write selected data into a file
        table_write(dataset, filename)
    else:
        #Read already selected data from $filename
        data_file = open(filename, 'r')
        for line in data_file:
            line = line.strip()
            line = line.split(' ') #assume that attributes are separated through 'spaces'
            dataset.append(line)
    return dataset


def data_quantize(dataset, att_indices, quantiles):
    """
    Quantize attributes whose indices are inside $att_indices, according to $quantiles
    """
    for i in range(len(dataset)):
        for j, index in enumerate(att_indices):
            val = None
            for k, interval in enumerate(quantiles[j]): 
                if interval[0] <= int(dataset[i][index]) <= interval[1]:
                    val = k
            if val is None:
                print 'The value %s of the attribute with index %d doesn\'t belong to any interval!'%(dataset[i][index],index)
            else:
                dataset[i][index] = '%s'%val
    return 


# def att_values_get(dataset):
#     """
#     Take a table $dataset and return all the possible values for every attribute (column) 
#     """
#     if len(dataset) != 0:
#         att_values = [[] for j in range(len(dataset[0]))]
#         for row in dataset:
#             for j in range(len(dataset[0])):
#                 if row[j] not in att_values[j]:
#                     att_values[j].append(row[j])
#         return att_values
#     else:
#         return []


def counts_compute(dataset): 
    """
    Input: $dataset, a list of lists [[]]. 
    refer to last column as sensitive attribute (SA), 
    and refer for other columns as quasi-identifier attributes (QI)

    For every pair (val_qi, val_sa) compute the related count: the number of 
    records (rows) that contains this pair, where val_qi is a QI value and val_sa is a SA value. 
    -- do this for every quasi-identifier quasi-identifier attribute --

    Output: a list of dictionaries of dictionaries: [{{}}] if $dataset is not empty, 
            and empty list [] otherwise. 

    e.g: counts_compute([
                         ['a', 'm', 's1'],
                         ['a', 'n', 's2'],
                         ['b', 'm', 's1'],
                         ['a', 'm', 's2']
                        ])
         return [
                 {'a': {'s2': 2, 's1': 1}, 'b': {'s2': 0, 's1': 1}}, 
                 {'m': {'s2': 1, 's1': 2}, 'n': {'s2': 1, 's1': 0}}
                ]
    """
    if len(dataset) != 0: #if dataset is not empty

        #obtain possible attributes' values 
        att_values = []
        for j in range(len(dataset[0])):
            att_values.append(list(set([record[j] for record in dataset])))
        #obtain counts 
        counts = []
        #initialize all counts to zero
        for j in range(len(dataset[0])-1):
            counts.append({val_qi:{val_sa:0 for val_sa in att_values[-1]} for val_qi in att_values[j]})
        #compute counts
        for record in dataset:
            for j, val_qi in enumerate(record[:-1]):
                counts[j][val_qi][record[-1]] += 1
        return counts, att_values[-1]
    else:
        return []


def add_noise(counts, epsilon): 
    """ add noise to counts"""
    counts_noised = [dict() for x in range(len(counts))]
    for j in range(len(counts)):
        for key, value in counts[j].iteritems():
            counts_noised[j][key] = value + np.random.laplace(loc=0.0, scale=1.0/epsilon)
    return counts_noised


def file_create(filename):
    """
    Create all the directories in path to filename, if not exist.
    """
    if os.path.basename(filename) == filename:
        return 0
    else:
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
    return 


def table_write(table,  filename):
    """
    Write $table into $filename. 
    A row per line and separate different attribute values by spaces.
    """
    file_create(filename)
    f = open(filename, 'w')
    for line in table:
        f.write(' '.join(str(x) for x in line) + '\n')
    f.close()
    return 
    