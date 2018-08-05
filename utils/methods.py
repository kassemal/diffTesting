"""
Methods related to data reading, data writing, data manipulation, and computations.   
"""

# !/usr/bin/env python
# coding=utf-8
from mondrian_l_diversity import mondrian_l_diversity
from mondrian import mondrian
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC,  SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from scipy.stats import entropy
from pulp import *
import numpy as np
import random
import pyemd
#
import utils.utility as ul
import adult_data
import internet_data


def attributes_filter(dataset, attributes, att_names, category):
    """
    Pick up from $dataset only the attributes whose names are in $attributes. 
    Note that, this method assumes that $dataset contains all the attributes from ($name)_data.ATT_NAMES
    """
    #obtain attribute indices
    is_cat = []
    att_indices = []
    for att in attributes:
        index = att_names.index(att)
        att_indices.append(index)
        is_cat.append(category[index])
    #obtain related columns 
    res_dataset = []
    for record in dataset:
        temp = []
        for j in att_indices:
            temp.append(record[j])
        res_dataset.append(temp)
    return res_dataset, is_cat


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


def data_import(name, size, qi_list, sa, filename):
    """ 
    Select $size records from dataset $name while considering only $qi_list and $sa_list. 
    Moreover, write the obtained dataset into $filename
    """
    dataset = []
    is_cat = []
    if name == 'adult':
        dataset = adult_data.read() #get full Adult dataset
        att_names = adult_data.ATT_NAMES
        category = adult_data.CATEGORY
    elif name == 'internet':
        dataset = internet_data.read() #get full Internet Usage dataset
        att_names = internet_data.ATT_NAMES
        category = internet_data.CATEGORY
    else:
        print 'The dataset "%s" doesn\'t exist!'%name
        return []
    #pick up $qi_list+$sa attributes
    dataset, is_cat = attributes_filter(dataset, qi_list + [sa], att_names, category) 
    #select $size records
    dataset = records_select(dataset, size)
    #write selected data into a file
    table_write(dataset, filename)
    return dataset, is_cat


def data_read(filename):
    """ 
    Read (already prepared) data from $filename
    """
    #Read data
    dataset = []
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
        return [], []


def distance_compute(pd1, pd2, classes_nb, d_tag='EMD', infinity=10**30):
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
    #
    elif d_tag == 'TVD':
        pd2_pd1 = [b-a for a, b in zip(pd1, pd2)]
        for x in pd2_pd1:
            distance += abs(x)
        distance = 0.5*distance
    #
    elif d_tag == 'KLD':
        distance = entropy(pd1, pd2)
    #
    elif D == 'JSD':
        M =  [0.5*(b+a) for a, b in zip(pd1, pd2)]
        distance = 0.5*entropy(pd1, M) + 0.5*entropy(M, pd2)
    #
    elif d_tag == 'BCD':
        BC = 0.0
        for i in range(len(pd1)):
            BC += sqrt(pd1[i]*pd2[i])
        distance = 0.0 - log(BC)


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


def table_write(table, filename):
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


def predictions_write(predictions, predictions_w, sa_values, filename, mode='w'):
    """
    Write $predictions and $predictions_w into $filename. 
    A line per record: $predictions[i] + $predictions_w[i]
    """
    if len(predictions) == len(predictions_w):
        file_create(filename)
        f = open(filename, mode)
        for i in range(len(predictions)):
            pred_with = ' '.join(str(predictions[i][val_s]) for val_s in sa_values) + ' '
            pred_without = ' '.join(str(predictions_w[i][val_s]) for val_s in sa_values) + '\n'
            f.write(pred_with+pred_without)
        f.close()
    else:
        print 'The two input lists don\'t have the same length'
    return 


def tree_get(attributes, is_cat, prefix):
    """
    For every attribute in $attributes, read the related tree from data/adult_*.txt, 
    and store it in $att_trees.
    Moreover, store the paths to leaves in the case of categorical attribute. 
    """
    #prefix = 'data/adult_'
    att_trees = []
    paths_to_leaves = []
    for j in range(len(attributes)):
        #if CATEGORY[ATT_NAMES.index(attributes[j])] or attributes[j] in att_quantize: #if the attribute is a categorical one or has been quantized 
        if is_cat[j]: 
            att_tree,  path_to_leaf = ul.read_tree_file(attributes[j], prefix)
            att_trees.append(att_tree)
            paths_to_leaves.append(path_to_leaf)
        else:
            att_tree = ul.read_pickle_file(attributes[j], prefix)
            att_trees.append(att_tree)
    return att_trees,  paths_to_leaves
    

def data_anonymize(dataset, att_trees, technique, p_value):
    """
    Apply anonymization $technique on $dataset
    """
    eval_result = None
    if p_value == 1: #no anonymization
        result = dataset
    else:
        if technique == 'l_diversity':
            result,  eval_result = mondrian_l_diversity(att_trees, dataset, p_value)
        elif  technique == 'k_anonymity':
            result, eval_result = mondrian(att_trees, dataset, p_value)
        else:
            print 'Error!'
            print 'The Anonymization Technique %s is not defined.'%Technique
            print 
    return result, eval_result 


def data_encode_one_hot(anonymized_dataset, att_values, paths_to_leaves, is_cat):
    """
    Apply one-hot encoding on $anonymized_dataset based on $paths_to_leaves hierarchy
    Continuous attributes are scaled into the interval [0,1]
    """
    #encode 
    qi_num = len(anonymized_dataset[0]) - 1
    encoded_data = []
    for i in range(len(anonymized_dataset)):
        record = []
        index = 0
        for j in range(qi_num):
            if is_cat[j]: #categorical
                code_j = []
                for val_q in att_values[j]:
                    if anonymized_dataset[i][j] in paths_to_leaves[index][val_q]:
                        code_j.append(1)
                    else:
                        code_j.append(0)
                record.extend(code_j)
                index += 1
            else:#continuous
                #scale into [0,1] -- y=x-min/max-min
                bounds = str(anonymized_dataset[i][j]).split(',')
                if len(bounds) == 1:
                    x = float(bounds[0])
                elif len(bounds) == 2:
                    x = 0.5*(float(bounds[0]) + float(bounds[1]))
                else:
                    print 'Error'
                    return 
                max_val = max([float(v) for v in att_values[j]])
                min_val = min([float(v) for v in att_values[j]])
                y = (x - min_val)/(max_val - min_val)
                if y > 1.0:
                    y = 1.0
                record.append(y)
        record.append(anonymized_dataset[i][-1])
        encoded_data.append(record)
    return encoded_data
    

def data_encode_TF(anonymized_dataset, target, paths_to_leaves, is_cat):
    """
    Implement TF encoding, that is:
    - For continuous attributes: if the target value is inside an interval, then set the related entry to 1. Otherwise, to 0.  
    - For categorical attributes: if the target value is in the path from a category to an leaf, then set the related entry to 1. Otherwise, to 0.  
    """
    qi_num = len(anonymized_dataset[0]) - 1
    encoded_data = []
    for i in range(len(anonymized_dataset)):
        record = []
        index = 0
        for j in range(qi_num):
            if is_cat[j]:
                if anonymized_dataset[i][j] in paths_to_leaves[index][target[j]]:
                    record.append(1)
                else:
                    record.append(0)
                index += 1
            else:
                temp = anonymized_dataset[i][j].split(',')
                if len(temp) == 1:
                    low = high = temp[0]
                else:
                    low = temp[0]
                    high = temp[1]
                if low <= target[j] <= high:
                    record.append(1)
                else:
                    record.append(0)                
        record.append(anonymized_dataset[i][-1])
        encoded_data.append(record)
    return encoded_data


def model_train(training_data, mla='BNB'):
    """
    Train a ML model based on $mla using $training_data. 
    NB: content of $training_data will be modified
    """
    #Separate label (SA) from features (QI)
    label = []
    for j in range(len(training_data)):
        label.append(training_data[j].pop(-1))
        #del training_data[j][-1] 
    x = np.array(training_data)
    y = np.array(label)
    #Train a model
    if mla == 'GNB':
        model = GaussianNB()
    elif mla == 'BNB': 
        model = BernoulliNB()
    elif mla == 'RFC':
        model = RandomForestClassifier(n_estimators=25)
    elif mla == 'DTC':
        model = tree.DecisionTreeClassifier()
    elif mla == 'lSVC':
        clf = LinearSVC()
        model = CalibratedClassifierCV(clf, cv=2, method='isotonic') 
    elif mla == 'SVC':
        clf = SVC(kernel='linear')
        model = CalibratedClassifierCV(clf, cv=2, method='isotonic') 
    else:
        print 'The algorithm %s is not supported.'%mla
        print 'The default algorithm BNB:BernoulliNB() will be used.'
        print 
        model = BernoulliNB()
    model.fit(x, y)
    return model 