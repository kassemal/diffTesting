"""
Methods
"""
#
# !/usr/bin/env python
# coding=utf-8
from import_adult_data import *
from pulp import *
import random 
#
def select_attributes(data, att_names):
    """
    This method selects (in order) from $data only the attributes whose names are in $att_names. 
    Note that, this method assumes that $data contains all the attributes from $ATT_NAMES_adult
    (see utils/import_adult_data.py) 
    """
    global ATT_NAMES_adult #from import_adult_data.py
    #obtain attribute indices
    att_indices = []
    for name in att_names:
        att_indices.append(ATT_NAMES_adult.index(name))
    #obtain related columns 
    res_data = []
    for record in data:
        temp = []
        for i in att_indices:
            temp.append(record[i])
        res_data.append(temp)
    return res_data
#
def select_records(data, size):
    """
    This method takes a table $data and selects only $size rows from it.
    """
    #Randomly select $size lines from data, if size < len(data). Otherwise, take all the data.
    counter = 0
    selected_data = []
    length = len(data)
    if size < length:
        counter = 0
        while(counter < size):
            i = random.choice(range(len(data)))
            selected_data.append(data.pop(i))
            counter += 1
    else: # if size > length, take all data
        selected_data = data
    return selected_data
#
def import_data(select_new, name='adult', size=1000, qi_list=['occupation'], sa_list=['income_level'], filename='results/selected_data'):
    """
    This method imports a dataset from a file. 
    It either reads already selected records from $filename if $select_new=False, or 
    selects $size records from dataset $name while considering only $qi_list and $sa_list if $select_new=True. 
    Note that in the latter case (when $select_new=True) the selected data are written into $filename.
    """
    selected_data = []
    if select_new == True:#select new data
        data = []
        if name == 'adult':
            data = read_adult_data() #get full Adult dataset
        else:
            print 'The dataset "%s" doesn\'t exist!'%name
            return []
        selected_att = [] 
        selected_att = select_attributes(data, qi_list + sa_list) #select specific attributes
        selected_data = select_records(selected_att, size)#select $size records
        #write selected data into a file
        write_data(selected_data, filename)
    else:
        #Read already selected data from $filename
        data_file = open(filename, 'r')
        for line in data_file:
            line = line.strip()
            line = line.split(' ')
            selected_data.append(line)
    return selected_data
#
def get_att_values(data):
    """
    This method takes a table $data and return all values that are taken by each attribute. 
    """
    if len(data) == 0:
        return []
    else:
        att_values = [[] for x in range(len(data[0]))]
        for record in data:
            for j in range(len(data[0])):
                if record[j] not in att_values[j]:
                    att_values[j].append(str(record[j]))
        return att_values
#
# def compute_counts(data,  q_values, s_values): #cij
#    """
#    This method computes, for every pair (q, s) belongs to (q_values, s_values), the number 
#    of records for which the quasi-identifier value is q and the sensitive value is s.
#    This method assumes one quasi-identifier and one sensitive attribute. Those can be tuples for 
#    several values. 
#    """
#    #initiate all counts to zero   
#    counts = dict()     
#    for q in q_values:
#        for s in s_values:
#            counts[str(q) + ';' + s] = 0
#    #compute counts
#    for record in data: 
#        counts[str(record[0]) + ';' + record[-1]] += 1
#    return counts
#
def compute_counts(data, qi_num, sa_num): #cij
    """
    This method computes, for every pair (q, s), the number of records for which the 
    quasi-identifier(s) tuple (first $qi_num attributes) is equal to q and the sensitive value(s) 
    tuple (last $sa_num attributes) is equal to s.
    Note that this method considers only the first record when we have several identical records. 
    """
    counts = dict()     
    indices = []
    for i in range(len(data)): 
        if i in indices: #skip records that are identical to already considered ones
            continue
        counter = 0
        for j in range(len(data)):
            if data[j] == data[i]:
                counter += 1
                if j != i:
                    indices.append(j)
        if len(data[i]) != qi_num + sa_num:
            print "The number of attributes in the record %d is not equal to the summation of QI and SA attributes."%(i+1)
            break
        else:
            key = ','.join(str(x) for x in data[i][0:qi_num]) + ';' + ','.join(str(x) for x in data[i][qi_num:])
            counts[key] = counter
    return counts
#
def create_file(filename):
    """
    This method creates all directories in path to filename, if not exist.
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
    return 0
#
def write_data(data,  filename):
    """
    This method writes table $data into $filename. 
    The different attributes will be separated by spaces. 
    """
    create_file(filename)
    f = open(filename, 'w')
    for line in data:
        f.write(' '.join(str(x) for x in line) + '\n')
    f.close()
    return 0 
    