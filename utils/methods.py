"""
This file contains methods related to read, write, manipulate and perform counts on datasets/tables. 
"""


# !/usr/bin/env python
# coding=utf-8
from pulp import *
import random 
import random
#
import adult_data


def attributes_filter(dataset, att_names):
    """
    Pick up from $dataset only the attributes whose names are in $att_names. 
    Note that, this method assumes that $dataset contains all the attributes from $name_data.ATT_NAMES
    """
    #obtain attribute indices
    att_indices = []
    for name in att_names:
        att_indices.append(adult_data.ATT_NAMES.index(name))
    #obtain related columns 
    res_dataset = []
    for record in dataset:
        temp = []
        for i in att_indices:
            temp.append(record[i])
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


def data_import(is_select_new, name='adult', size=1000, qi_list=['occupation'], sa='income_level', filename='results/selected_data'):
    """
    Import a dataset from a file: 
    either read already prepared data from $filename if $is_select_new=False, or 
    select $size records from dataset $name while considering only $qi_list and $sa_list if $is_select_new=True. 
    """
    dataset = []
    if is_select_new == True:#select new data
        if name == 'adult':
            dataset = adult_data.read() #get full Adult dataset
        else:
            print 'The dataset "%s" doesn\'t exist!'%name
            return []
        #pick up $qi_list+$sa attributes
        dataset = attributes_filter(dataset, qi_list + [sa]) 
        #select $size records
        dataset = records_select(dataset, size)
        #write selected data into a file
        table_write(dataset, filename)
    else:
        #Read already selected data from $filename
        data_file = open(filename, 'r')
        for line in data_file:
            line = line.strip()
            line = line.split(' ')
            dataset.append(line)
    return dataset


def data_quantize(dataset, att_indices, quantiles):
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
    return None


def att_values_get(dataset, index=-1):
    """
    Take a table $dataset and return all the possible value of the attribute with index $index. 
    """
    att_values = []
    if len(dataset) == 0:
        return att_values
    else:
        for row in dataset:
            if row[index] not in att_values:
                att_values.append(row[index])
        return att_values


def qi_concatenate(dataset, qi_num=-1):
    """ 
    Take a $dataset and concatenate first $qi_num attribute values.
    Consider only distinct records. 
    """
    qi_concatenation = []
    for row in dataset:
        qi = ','.join(str(x) for x in row[0:qi_num])
        if qi not in qi_concatenation:
            qi_concatenation.append(qi)
    return qi_concatenation


def counts_compute(dataset, qi_num=-1, sa_index=-1): 
    """
    For every distinct pair (q, s) in $dataset, compute the number of records for which the 
    quasi-identifier(s) tuple (first $qi_num attributes) is equal to q and the sensitive value(s) 
    tuple (last $sa_num attributes) is equal to s.
    """
    qi_concatenation = qi_concatenate(dataset)
    sa_values = att_values_get(dataset, sa_index)
    counts = dict()  
    for q in qi_concatenation:
        for s in sa_values:
            counts[q + ';' + s] = 0
    #
    for i in range(len(dataset)):
        key = ','.join(str(x) for x in dataset[i][0:qi_num]) + ';' + ','.join(str(x) for x in dataset[i][qi_num:])
        counts[key] += 1
    return counts


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
    return None


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
    return None
    