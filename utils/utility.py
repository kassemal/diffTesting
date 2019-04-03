"""
public functions

This code taken from 
https://github.com/qiyuangong/Basic_Mondrian
under the MIT License (MIT)
"""

# !/usr/bin/env python
# coding=utf-8
from models.gentree import GenTree
from models.numrange import NumRange
import pickle

__DEBUG = False
    
def cmp_str(element1, element2):
    """
    compare number in str format correctley
    """
    try:
        return cmp(int(element1), int(element2))
    except ValueError:
        return cmp(element1, element2)

def list_to_str(value_list, cmpfun=cmp, sep=';'):
    """covert sorted str list (sorted by cmpfun) to str
    value (splited by sep). This fuction is value safe, which means
    value_list will not be changed.
    return str list.
    """
    temp = value_list[:]
    temp.sort(cmp=cmpfun)
    return sep.join(temp)

def read_pickle_file(att_name, prefix):
    """
    read pickle file for numeric attributes
    return numrange object
    """
    try:
        static_file = open(prefix + att_name + '_static.pickle', 'rb')
        (numeric_dict, sort_value) = pickle.load(static_file)
        static_file.close()
        result = NumRange(sort_value, numeric_dict)
        return result
    except:
        print("Pickle file not exists!!")

def read_tree_file(att_name, prefix):
    """read tree data from prefix + att_name + postfix
    """
    path_to_leaf = dict()
    att_tree = {}
    postfix = ".txt"
    treefile = open(prefix + att_name + postfix, 'rU')
    att_tree['*'] = GenTree('*')
    if __DEBUG:
        print("Reading Tree" + att_name)
    for line in treefile:
        # delete \n
        if len(line) <= 1:
            break
        line = line.strip()
        temp = line.split(';')
        path_to_leaf[temp[0]] = temp
        # copy temp
        temp.reverse()
        for i, t in enumerate(temp):
            isleaf = False
            if i == len(temp) - 1:
                isleaf = True
            # try and except is more efficient than 'in'
            try:
                att_tree[t]
            except:
                att_tree[t] = GenTree(t, att_tree[temp[i - 1]], isleaf)
    if __DEBUG:
        print('Nodes No. = %d' % att_tree['*'].support)
    treefile.close()
    #print att_tree
    return att_tree,  path_to_leaf
