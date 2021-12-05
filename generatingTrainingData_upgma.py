#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:51:42 2019

@author: au560049
"""

import numpy as np
import os
import io
import random
import json
import gzip
import sys
from scipy.spatial.distance import pdist
from fastcluster import linkage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import leaves_list, dendrogram
from polo import optimal_leaf_ordering




LABELS = {'hard_LINKED_0':3,'hard_LINKED_1.':3,'hard_LINKED_2':3,'hard_LINKED_3':3,'hard_LINKED_4':3,
          'hard_FIXED_5':0,'hard_LINKED_6':3,'hard_LINKED_7':3,'hard_LINKED_8':3,'hard_LINKED_9':3,
          'hard_LINKED_10':3,'soft_LINKED_0':4,'soft_LINKED_1.':4,'soft_LINKED_2':4,'soft_LINKED_3':4,
          'soft_LINKED_4':4,'soft_FIXED_5':2,'soft_LINKED_6':4,'soft_LINKED_7':4,'soft_LINKED_8':4,
          'soft_LINKED_9':4,'soft_LINKED_10':4,'neutral':1}

train_valid_ratio = 0.2
lst_pos_valid,lst_snp_valid,lst_pos_train,lst_snp_train = [],[],[],[]

def pad(l, num, width):
    l.extend([num] * (width - len(l)))
    return l

def upgma(snp_mat):
	D = pdist(pad_mat, 'euclidean')
	Z = linkage(D, 'average')

	optimal_Z = optimal_leaf_ordering(Z, D)

	order = leaves_list(optimal_Z)
	res = pad_mat[order]
	return res.T

def readingmsOutFile(completePath_input,name,outFilePath,flag):
    scenario = [value for key, value in LABELS.items() if key in completePath_input]
    
    
    fz = gzip.open(completePath_input,'rb')
    f = io.BufferedReader(fz)
    
    content = [line.decode('utf8').strip().split() for line in f]

    
    indices = [idx for idx, lst in enumerate(content) if any('segsites' in el for el in lst)]
               
    total_examples = len(indices)
    num_valid_examples = int(total_examples * train_valid_ratio)
    num_train_examples = total_examples - num_valid_examples
    
    train = indices[:num_train_examples]

    validation = indices[num_train_examples:]
   
    counter = 1
    
    for index in train:
        #position,snp_matrix = [],[]
        pos = content[index+1][1:]

        
		pos = np.asarray(pad([float(x) for x in pos],0.0,4986))
		
		l = content[index+2:index+2+144]
		pad_mat = np.array([pad(list(y),0,4986) for x in l for y in x],dtype=int)
		
		res = upgma(pad_mat)

		
		completePath_output_matrix = outFilePath + '/' + name + '_'+str(counter) + '_snp'
		completePath_output_pos = outFilePath + '/' + name + '_'+str(counter) + '_pos'
		
		mat_name = completePath_output_matrix + '.npy'
		pos_name = completePath_output_pos + '.npy'
		
		lst_snp_train.append(mat_name)
		lst_pos_train.append(pos_name)
		
		np.save(completePath_output_matrix, np.asarray(res))
		np.save(completePath_output_pos,np.asarray(pos))    
		
		counter += 1
		labels[mat_name]= scenario[0]
        
        
    for index in validation:
        #position,snp_matrix = [],[]
        pos = content[index+1][1:]
        
		pos = np.asarray(pad([float(x) for x in pos],0.0,4986))
		
		#position.append(pos)
		
		###binary matrix
		l = content[index+2:index+2+144]
		#res = np.asarray(sort_min_diff(np.array([pad(list(y),0,5000) for x in l for y in x],dtype=int)).T)
		pad_mat = np.array([pad(list(y),0,4986) for x in l for y in x],dtype=int)
		
		res = upgma(pad_mat)
		
		
		completePath_output_matrix = outFilePath + '/' + name + '_'+str(counter) + '_snp'
		completePath_output_pos = outFilePath + '/' + name + '_'+str(counter) + '_pos'
		
		mat_name = completePath_output_matrix + '.npy'
		lst_snp_valid.append(mat_name)
		pos_name = completePath_output_pos + '.npy'
		lst_pos_valid.append(pos_name)
		
		np.save(completePath_output_matrix, np.asarray(res))
		np.save(completePath_output_pos,np.asarray(pos))    
		
		counter += 1
		
		labels[mat_name]= scenario[0]
        
      

if __name__ == "__main__":
    
    inFilePath, outFilePath, outFilePath_dict, start= sys.argv[1:]
    
    partition_pos,partition_snp,labels = {},{},{}
    
    filesNamesForTheScenario = [f for f in os.listdir(inFilePath) if f.startswith(start) and f.endswith('msOut.gz')]
    print(filesNamesForTheScenario)

    flag = 0
    
    for file in filesNamesForTheScenario:
        name = file.split('.')
        completePath_input = inFilePath + '/' + file
        if ('hard_FIXED_5' in file) or ('soft_FIXED_5' in file) or ('neutral') in file:
            flag = 1
        
        n = name[0] + '.' +name[1]

        readingmsOutFile(completePath_input,n,outFilePath,flag)
        
        flag = 0
        
        
    zip_train = list(zip(lst_snp_train,lst_pos_train))
    zip_valid = list(zip(lst_snp_valid,lst_pos_valid))
    
    for i in range(3):
        random.shuffle(zip_train)
        random.shuffle(zip_valid)
        
    
        
    lst_snp_train,lst_pos_train = zip(*zip_train)
    lst_snp_valid,lst_pos_valid = zip(*zip_valid)
    
    partition_snp['train'] = lst_snp_train
    partition_snp['valid'] = lst_snp_valid
    
    partition_pos['train'] = lst_pos_train
    partition_pos['valid'] = lst_pos_valid
    
    complete_path_snp = outFilePath_dict + '/' + start + '_dict_snp.json'
    complete_path_pos = outFilePath_dict + '/' + start + '_dict_pos.json'
    complete_path_labels = outFilePath_dict + '/' + start + '_dict_labels.json'
    
    with open(complete_path_snp, 'w') as f:
        json.dump(partition_snp, f)
            
    with open(complete_path_pos, 'w') as f:
        json.dump(partition_pos, f)
            
    with open(complete_path_labels, 'w') as f:
        json.dump(labels, f)
    
    
