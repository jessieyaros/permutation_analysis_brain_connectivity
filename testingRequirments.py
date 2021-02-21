# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:40:48 2021

@author: jyaros
"""
import numpy as np
import random
#import matlab.engine
#eng = matlab.engine.start_matlab()
#instructions on calling matlab func from python- will be needed for running
#BCT toolbox: https://www.mathworks.com/videos/how-to-call-matlab-from-python-1571136879916.html


#initialize variables
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = [0]*16

# initialize shape 0f matrix
shape = (3,3)

#simulate separate condition matrices
[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p] = [np.ones((3,3))*(i+1) for i in range(16)]

#combine matrices and flatten to vector
e = np.concatenate((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)).flatten()

# create list of condition names
labels =   ['sr_enc_lcr', 'sr_enc_lfa', 'or_enc_lcr', 'or_enc_lfa', 
 'sr_enc_th', 'sr_enc_tm', 'or_enc_th', 'or_enc_tm',
 'sr_ret_lcr', 'sr_ret_lfa', 'or_ret_lcr', 'or_ret_lfa', 
 'sr_ret_th', 'sr_ret_tm', 'or_ret_th', 'or_ret_tm']

#each permutaiton of randomlly shuffled data is stored in this dictioanry. 
dict_permutations = {}

for iter in range(10):
    #randomly shuffle vector of all data
    np.random.shuffle(e)
    
    #spliT randomized data back into 16 individual vectors
    [a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1,n1,o1,p1]=np.split(e,16)

    #reshape vectors into matrices matching original size
    [a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2,p2] = \
    [vector.reshape(shape) for vector in [a1,b1,c1,d1,e1,f1,g1, \
                                          h1,i1,j1,k1,l1,m1,n1,o1,p1]]
     
    #store matrices in a list and shuffle one last time
    cond_data = [a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2,p2]
    random.shuffle(cond_data)
    
    #shuffle codnition names by randomly sampling labels without replacement
    cond_labels = random.sample(labels, len(labels))
    
    #store labels and matrices in dictionary
    dict ={cond_labels[i]: cond_data[i] for i in range(len(cond_labels))}
    
    #store dictionary as value for ith iteration in 
    dict_permutations[iter] = dict
    

#Create nested dictionary that will have key for iteration, then inner dict
#with keys for each condition type and the corresponding conditions
#from varname import nameof

#foo = dict()

#fooname = nameof(foo)