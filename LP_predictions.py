"""
This program evaluates the effect of adding Laplacian noise on the 
"privacy" of the sensitive attribute inside UCI Adult dataset. 
The idea is to compare between two distribution: 
one which we obtain from a dataset that contains a certain record, 
and another which we obtain from the same dataset after removing this record. 
It considers 'occupation' as a sensitive attribute, and 
'income_level' as a quasi-identifier. 

More precisely, it does the following: 
1- Read some already selected Adult records from a file or 
   select new $size records from 'data/adult.all' based on the value of $Select
2- Repeat for every epsilon in $Epsilons: for every distinct record inside the dataset do the following: 
	- generate $NB_Iterations noised counts for the current record (q,s) by computing 
	  noised_val = min(max(0, count(q,s)+noise), T_q) where T_q = \sum_{s'} count(q, s'), 
	  then obtain from all the noised values a distribution hist_with_r
	- remove the current record and repeat the previous step in order to obtain a distribution hist_without_r
	- obtain the maximal ratio between the frequencies of the two histograms, considered point-wisely. 
3- Plot the ratios cumulative relative frequency for all epsilon values in one graph
"""
#
# !/usr/bin/env python
# coding=utf-8
from pulp import *
from utils.import_adult_data import *
from utils.methods import import_data, compute_counts
from utils.methods import create_file, write_data
import numpy as np
import copy, math
import matplotlib.pyplot as plt
#
def compute_hist(count, total, nb_iterations, epsilon):
	mean = 0.0
	decay = 1.0/epsilon
	noised_values, noised_values_wo = [], []
	for r in range(0, nb_iterations):
		#generate random noise
		noise = np.random.laplace(loc=mean, scale=decay)
		#add the noise to $value
		term = min(max(0, count + noise), total)
		noised_values.append(term)
		#
		noise = np.random.laplace(loc=mean, scale=decay)
		term = min(max(0, count - 1 + noise), total)
		noised_values_wo.append(term)	 
#
	min_val = min(min(noised_values), min(noised_values_wo))
	max_val = max(max(noised_values), max(noised_values_wo))
	h, b_edges = np.histogram(noised_values, bins=range(int(math.floor(min_val)),int(math.ceil(max_val))+1))
	h_wo, b_wo = np.histogram(noised_values_wo, bins=b_edges)
	return h, h_wo, b_edges
#
def plot_hist(h, h_wo, edges, filename):
	width = 0.45 
	bins = np.arange(1, len(edges))
	fig, ax = plt.subplots()
	rects1 = ax.bar(bins-(width/2), h, width, color='b')
	rects2 = ax.bar(bins+(width/2), h_wo, width, color='r')
	ax.set_xticks(bins)
	labels = []
	for i in range(len(edges)-1):
		labels.append('[%s-%s]'%(edges[i],edges[i+1]))
	ax.set_xticklabels(labels, rotation=45)
	ax.legend((rects1[0], rects2[0]), ('with r', 'without r'))
	ax.set_ylabel('Frequency')
	ax.set_xlabel('Ratio')
	ax.set_ylim(0, max(max(h), max(h_wo))+5)
	fig = plt.gcf()
	create_file(filename)
	fig.savefig(filename, bbox_inches='tight')
	plt.close('all')
	return 0
#
#ATT_NAMES_adult = ['age', 'workclass', 'final_weight', 'education',
#             'education_num', 'marital_status', 'occupation', 'relationship',
#             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
#             'native_country', 'income_level']   
QI_ATT = ['occupation'] #quasi-identifier attributes
SA_ATT = ['income_level'] #sensitive attributes
Name = 'adult' #name of the dataset
Size = 1000 #size of the dataset to consider
Select = True #True is to select new data
Epsilons = [0.01, 0.05, 0.1, 0.25] #values of epsilon to consider
color_list = {0.01:'b', 0.1:'y', 0.25: 'r', 0.05:'g', 2.0:'c'}  
NB_Iterations = 100000 #number of noised values to compute per record
#
if __name__ == '__main__':
	
	#1- Import data
	print 'Read data ...'
	selected_DATA = []
	filename = 'results/selected_DATA/%s/selected_%s_S%s'%(Name,Name,str(Size))
	selected_DATA = import_data(Select, name=Name, size=Size, qi_list=QI_ATT, sa_list=SA_ATT, filename=filename)
	#
	#2- Add noise and compute the maximal ratio
	#compute counts (frequency of every pair (q, s) belong to (QI_ATT, SA_ATT) inside the dataset)
	QI_num = len(QI_ATT)
	SA_num = len(SA_ATT)
	COUNTS = compute_counts(selected_DATA, QI_num, SA_num)
	#print COUNTS
	#For every epsilon inside $Epsilons: for every distinct record run $NB_Iterations and compute the related distributions 
	ratio_all_eps = []
	for eps in Epsilons:
		print 'Obtain ratios for epsilon = %s ...'%str(eps)
		considered_records = []
		ratio = []
		for i in range(len(selected_DATA)):
			if selected_DATA[i] not in considered_records:
				#concatenate QI and SA values
				q = ','.join(str(x) for x in selected_DATA[i][0:QI_num]) 
				s = ','.join(str(x) for x in selected_DATA[i][QI_num:])
				#compute the total (summation) for the current record
				total_q = 0.0
				for key, value in COUNTS.iteritems():
					if q == key.split(';')[0]: 
						total_q += value
				#obtain histograms
				hist, hist_wo, bin_edges = compute_hist(COUNTS[q+';'+s], total_q, NB_Iterations, eps)
				considered_records.append(selected_DATA[i])
				#compute the maximal ratio
				r_temp = []
				for p in range(len(hist)):
					if hist[p] != 0 and hist_wo[p] != 0:
						r_temp.append(max(hist[p]/hist_wo[p], hist_wo[p]/hist[p]))
				ratio.append(max(r_temp))
				#plot the histograms and save the related figure into a file. This is just to examine the shape of the histograms 
				#filename = 'results/figures/hist_ratio/%s/S%d/N%d/Ep%s/hist_%s_S%d_N%d_Ep%s_r%d.pdf'%(Name,Size,NB_Iterations,str(eps),Name,Size,NB_Iterations,str(eps),i+1)
				#plot_hist(hist, hist_wo, bin_edges, filename)
		#append ratio list to an overall list for all epsilons
		ratio_all_eps.append(ratio)
	#write the ratios into a file (a line per epsilon value). This is just to keep a copy of the data
	filename = 'results/ratios/%s/ratio_%s_S%d_N%d'%(Name,Name,Size,NB_Iterations)
	create_file(filename)
	f = open(filename, 'w')
	f.write(' '.join(str(x) for x in Epsilons) + '\n')
	for ratio_l in ratio_all_eps:
		f.write(' '.join(str(x) for x in ratio_l) + '\n')
	f.close()
	#
	#3- Plot the relative CDF of the ratios  
	curves, leg = [], []
	fig = plt.figure()  
	curves = [0 for j in range(len(Epsilons))]
	leg = ['eps = ' + str(x) for x in Epsilons]
	index = 0
	for i in range(len(ratio_all_eps)):
	    size = len(ratio_all_eps[i])
	    yvals = np.arange(1,  size+1)/float(size)
	    curves[index],  = plt.plot(np.sort([int(x) for x in ratio_all_eps[i]]), yvals, color_list[Epsilons[i]], label=Epsilons[i])
	    index += 1
	plt.legend([x for x in curves],leg,loc=4,fontsize=12,frameon=False)
	#plt.xlim()
	plt.xlabel('Ratio', fontsize=14)
	plt.ylabel('Cumulative Relative Frequency',fontsize=14)  
	#plt.title('')
	fig = plt.gcf()
	filename = 'results/figures/cdf/%s/conditional/cdf_LB_%s_S%d_N%d.pdf'%(Name,Name,Size,NB_Iterations)
	create_file(filename)
	fig.savefig(filename, bbox_inches='tight')
	print 'Done!'
	plt.show()
	plt.close(fig)


