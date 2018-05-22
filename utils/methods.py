"""
Methods related to data reading, data writing, data manipulation, and computations.   
"""

# !/usr/bin/env python
# coding=utf-8
from pulp import *
import numpy as np
import random
import pyemd
import matplotlib.pyplot as plt
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


def counts_compute(dataset): 
    """
    Input: $dataset, a list of lists [[]]. 
    refer to last column as sensitive attribute (SA), 
    and refer for other columns as quasi-identifier attributes (QI)

    For every pair (val_q, val_s) compute the related count: the number of 
    records (rows) that contains this pair, where val_q is a QI value and val_s is a SA value. 

    Output: a list of dictionaries of dictionaries: [{val_q:{val_s:count}} for j=1,..,QI_num] if $dataset is not empty, 
            and empty list [] otherwise. (QI_num is the number of QI attributes)

    e.g: counts_compute([
                         ['a', 'd', 's1'],
                         ['a', 'c', 's2'],
                         ['b', 'd', 's1'],
                         ['a', 'd', 's2']
                        ])
         return [
                 {'a': {'s2': 2, 's1': 1}, 'b': {'s2': 0, 's1': 1}}, 
                 {'d': {'s2': 1, 's1': 2}, 'c': {'s2': 1, 's1': 0}}
                ]
    *Additionally return a list of sensitive attribute values (last column)
    """
    if len(dataset) != 0: #if dataset is not empty
        #obtain possible attributes' values 
        att_values = []
        for j in range(len(dataset[0])):
            att_values.append(list(set([record[j] for record in dataset])))
        ##### obtain counts #####
        counts = []
        #initialize all counts to zero
        for j in range(len(dataset[0])-1):
            counts.append({val_q:{val_s:0 for val_s in att_values[-1]} for val_q in att_values[j]})
        #compute counts
        for record in dataset:
            for j, val_q in enumerate(record[:-1]):
                counts[j][val_q][record[-1]] += 1
        return counts, att_values[-1]
    else:
        return []


def distance_compute(pd1, pd2, classes_nb, d_tag='EMD', infinity=1000):
    """
    Compute distance between two probability distributions $pd1 and $pd2. 

    e.g. distance_compute([0.45, 0.55], [0.2, 0.8], classes_nb=2)
    rType: float, a distance
    """
    distance = 0.0
    #Switch cases according to the $d_tag
    if d_tag == 'EMD':
        #generate ground matrix: an equal ground distance is taken between any two different attribute values. 
        ground_matrix = []
        for i in range(classes_nb):
            row = []
            for j in range(classes_nb):
                if j == i:
                    row.append(0.0)
                else:
                    row.append(1.0)
            ground_matrix.append(row)
        ground_matrix = np.array(ground_matrix) 
        distance = pyemd.emd(np.array([float(x) for x in pd1]), np.array([float(x) for x in pd2]), ground_matrix)
    #
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

    
def cdf_plot(distances_dict_average, eps_values, color_list, tag, filename):
    '''
    Plot curves and save related figure as $filename
    '''
    curves, legend = [], []
    fig = plt.figure()  
    curves = [0 for x in range(len(eps_values))]
    legend = ['eps = ' + str(x) for x in eps_values]
    index = 0
    for i in range(len(distances_dict_average)):
        size = len(distances_dict_average[i])
        yvals = np.arange(1,  size+1)/float(size)
        curves[index],  = plt.plot(np.sort([x for x in distances_dict_average[i]]), yvals, color_list[eps_values[i]], label=eps_values[i])
        index += 1
    plt.legend([x for x in curves], legend, loc=4, fontsize=12, frameon=False)
    label_x = ''
    if tag == 'EMD':
        label_x = 'Earth Mover\'s Distance'
    elif tag == 'm_ratio':
        label_x = 'Maximal Ratio'
    plt.xlabel('%s'%label_x, fontsize=14)
    plt.ylabel('Cumulative Relative Frequency', fontsize=14)  
    #plt.title('')
    fig = plt.gcf()
    file_create(filename)
    fig.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close(fig)
    return