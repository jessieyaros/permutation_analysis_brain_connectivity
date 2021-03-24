# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:37:55 2021

@author: jyaros
"""
from datetime import datetime
startTime = datetime.now()
import os
import pandas as pd
import random
import numpy as np
#from statsmodels.stats.anova import AnovaRM
from scipy import stats
from pingouin import rm_anova
from matplotlib import pyplot as plt

#This fucntion structures data to faciliate analysis
def wrangle_data():
    print("Wrangling Data...")
    #import graph metric data to df
    workingDir = os.getcwd()
    filename = 'input_topological_metrics.csv'
    file = os.path.join(workingDir, filename)
    raw_data = pd.read_csv(file)
    
    #use string in Network col to create new cols that better describe data, 
    #and will be used to subset
    raw_data['subj'] = raw_data['Network'].str[11:21]
    raw_data['cond'] = raw_data['Network'].str[22:34]
    raw_data['cost'] = raw_data['Network'].str[36:39]
    
    #create new df with columns of interest
    cols = ['subj', 'cond', 'cost', 'Avg_Clustering_Coef', 
            'Avg_Local_Efficiency', 'Global_Efficiency']
    df = raw_data[cols]
    
    #remove metrics computed  with costs over .25.
    pd.options.mode.chained_assignment = None  
    #pd.options.mode.chained_assignment = 'warn'
    df['cost'] = df['cost'].astype(float)
    df = df.loc[df['cost'] <= .25]
    
    #average across all costs to compute final 
    #graph metric per subject*condition
    df_final = df.groupby(['subj', 'cond']).mean()
    df_final = df_final.drop('cost', axis = 1)
    df_final.reset_index(inplace=True)
    
    #create subsets of df for each ANOVA of interest. In this I am selecting 
    #data from lure encoding and target retrieval trials. 
    enc_lure = df_final[df_final['cond'].isin(['Condition001','Condition002',
                                               'Condition005', 'Condition006'])]
    retr_targ = df_final[df_final['cond'].isin(['Condition011','Condition012',
                                                'Condition015', 'Condition016'])]
 
    #assign race and accuracy columns to use as factors in ANOVA
    enc_lure['race'], retr_targ['race'] = [np.where(df['cond'].\
                        isin(['Condition001','Condition002', 'Condition011',\
                       'Condition012']), 'OR', 'SR')\
                        for df in [enc_lure, retr_targ]]
        
    enc_lure['accuracy'], retr_targ['accuracy'] = [np.where(df['cond'].\
                        isin(['Condition001','Condition005', 'Condition011',\
                       'Condition015']), 'Crr', 'InCrr')\
                        for df in [enc_lure, retr_targ]]
        
    return enc_lure, retr_targ

#This fucntion fits ANOVA to data and computes test statistics
def anova_observed(data):
    print("ANOVA of observed data...")

    #key could be list[1] and results as values
    
    #store full anova results and f_stats for effects of interest for boht
    #the encoding and retrieval conditions
    results = {}

    #for both enc_lurea and retr_targ data
    for df, cond_label in data:
        
        anova_clust = rm_anova(dv = 'Avg_Clustering_Coef', \
                               within = ['race','accuracy'], \
                               subject = 'subj', data =df)
        anova_eff = rm_anova(dv = 'Avg_Local_Efficiency', \
                             within = ['race','accuracy'], \
                               subject = 'subj', data =df)
            

        vals = []
        keys = ['anova_cluster_coef', 'f_clust', 'anova_loc_efficiency', 'f_eff']

        #if encoding select f-stat for interaction of race and accuracy
        if cond_label == 'enc_lure':
            vals = [anova_clust, anova_clust.loc[2,'F'], anova_eff, anova_eff.loc[2,'F']]
        #else if retrival, select f-stat for main effect of accuracy
        elif cond_label == 'retr_targ':
            vals = [anova_clust, anova_clust.loc[1,'F'], anova_eff, anova_eff.loc[1,'F']]
        results[cond_label] = dict(zip(keys,vals))
        
    return results

#This function generates random permuations of shuffled data ans stores in 
#dfs for each graph metric and each condtion
def define_permuations(data, numPermutations):
    print("Defining Permutations...")

    #dictionary to store dataframes with final permutations
    permuted_dfs = {}
    
    #for enc_lure and retr_traget data
    for df, cond_label in data:
        
        #define unique subjects
        subjects = df['subj'].unique().tolist()
        
        #Intialize dictionaries that will store a randomized shuffling of data
        #for each subject, for as many permutations as specified
        clust_permutations = {}
        eff_permutations = {}    
        
        #for current condition's dataset cycle through each subject's record
        for subj in subjects:
            #for each subject get list of observed cluster and local 
            #efficiency metrics
            clust_metrics = df['Avg_Clustering_Coef'].loc[df['subj'] == subj].tolist()        
            local_eff_metrics = df['Avg_Local_Efficiency'].loc[df['subj'] == subj].tolist()
            
            
            #Create dictionaries for temp storage of each permutation's data, 
            #where keys are permuation numbers and values are the shuffled list
            subj_clust_permutations = {}
            subj_eff_permutations = {}
            
            #incrase iteration when not testing
            #Create randomly shuffled samples of the cluster and locel eff lists
            for permutation in range(numPermutations):
                #shuffle order of cluster and local efficiency metrics by randomly
                #sampling both lists
                clust, eff = [random.sample(x,len(x)) for x in 
                             [clust_metrics,local_eff_metrics]]
                
                #store current iteration's randomized data
                subj_clust_permutations[permutation] = clust
                subj_eff_permutations[permutation] = eff
        
            #For each subject, store cluster and efficiency dictionaries in larger
            #nested dictionary, where keys are subjects
            clust_permutations[subj] = subj_clust_permutations
            eff_permutations[subj] = subj_eff_permutations
        
        #Now that shuffled data has been generated for the current condition 
        #for each subject for n permutations, 
        #craate dataframes from the subj_clust_permutations and subj_eff_permutations
        #dictionaries. Transpose so that subjects are across rows rather than columns
        #Stack the  permuation columns. If there are n permutation 
        #columsn, their will now be n rows for each permutation per subject  
        
        df_clust_permutations, df_eff_permutations = \
            [pd.DataFrame(df).T.stack().to_frame().reset_index() for df in \
             [clust_permutations, eff_permutations]]
                
        #assign meaningful column names
        df_clust_permutations, df_eff_permutations = \
            [df.rename(columns = dict(zip(df.columns.values, 
            ['subj', 'permutation', 'shuffled_data']))) 
            for df in [df_clust_permutations, df_eff_permutations]]
            
        #store these dfs in a dictionary so they are not overwritten by
        #subsequent condition dfs. 
        df_dict = {}
        df_dict['clust_permutations'], df_dict['eff_permutations'] = \
            [df for df in [df_clust_permutations, df_eff_permutations]]
        
        #store cluster  and local efficiency permuatations relative to 
        #their condition. Ie. Key is condition (enbc or retr). Values is 
        #nested dictionary of both cluster and efficiency permutation dfs
        permuted_dfs[cond_label] = df_dict

    return permuted_dfs, numPermutations

def structure_permutations(permuted_dfs):
    print("Restructuring permutation data...")

    #Goal: Restructure each df. On input, each subject column contains lists of 
    #length 4 (one element for each condition). Goal is to 'explode' lists
    #such that each element has a dedicated row. The permutation number
    #will still be maintained by the index, which will no logner be unique
    
    #store restructured dfs in new dictionary
    structured_permutation_dfs = {}
   
    #cycle through nested dictionary containing noncompliant df structures
    for cond in permuted_dfs.keys():
        structured_permutation_dfs[cond] = {}
        for metric, df in permuted_dfs[cond].items():
           
            #intialize new df
            df_explode =  pd.DataFrame()
            
            #cycle through each permutation (i.e. row)
            for row in df.index:
                permutation = df.iloc[[row]]
                #Convert index to column that serves as reference point for 
                #other columsn to be exploded across
                #ermutation.reset_index(inplace=True)
                #permutation = permutation.rename(columns={'index':'permutation'})
                #Exploded columns relative to permuation column
                #Then reset permutation column to be index again
                permutation_explode = permutation.set_index('permutation')\
                                            .apply(pd.Series.explode).reset_index()
                
                #Append expanded permutation to dataframe, where each original
                #row is now 4 rows            
                df_explode = df_explode.append(permutation_explode)            
                
            #Assing race and accuracy factors to each observation. Data 
            #are already randomized so the order of condition assignment is 
            #arbitrary. #The only contraitnt is that each permutation (ie each 
            #set of four rows must contain exactly one of each of four 
            #conditions : SR Corr, SR InCorr, OR Corr, OR InCorr . We do this 
            #by initializing list that satisfy the constraint for a single 
            #permutation and then repeating this pattern for all permutations, 
            #for all subjects
            
            #initialize the four required conditions per each permutation
            race = ['SR', 'SR', 'OR', 'OR']
            acc = ['Corr', 'InCorr', 'Corr', 'InCorr']
            
            #calculate number of total permutations across the full subject sample
                #Add one since permutations are 0-indexed
            numPermutations = df_explode['permutation'].max() + 1
            numSubjects = 22
            num_distinct_permutations = numPermutations * numSubjects
            
            #create lists for race and accuracy label assignment, with length
            #equal to the total number of permutations. Assign as columns
            race_col, acc_col = \
            [factors * num_distinct_permutations for factors in [race, acc]]

            df_explode['race'], df_explode['acc']  = race_col, acc_col
            df_explode = df_explode.reset_index()
            
            
            #store restructured df
            structured_permutation_dfs[cond][metric] = df_explode
            
    return structured_permutation_dfs

def anova_permutations(shuffled_data_dict):
    print("ANOVAs on permuted data...")

    #to run on actual data do. Would have to change column names
    #run_anova2(enc_lure[['subj','shuffled_data','race', 'acc']])

    def run_anova(df_subset):
        anova = rm_anova(dv = 'shuffled_data', within = ['race','acc'], \
                        subject = 'subj', data = df_subset)
        #Save the correct F-stat. 
        #Ie. ME of acc for target retrieval and interaction for lure encoding
        if  df['cond'][0] == 'enc_lure':
            f_stat = anova.loc[anova['Source'] == 'race * acc', 'F']
        elif df['cond'][0] == 'retr_targ':
            f_stat = anova.loc[anova['Source'] == 'acc', 'F']

        return f_stat
    # Call run_anova() on each dataframe and store f_stats for each 
    # condition* metric in nested dictionary, store_f_statistics
    store_f_statistics ={}
    for cond in shuffled_data_dict.keys():
        store_f_statistics[cond] = {}
        for metric, df in shuffled_data_dict[cond].items():         
            #convert shuffled_data to numeric datatype
            df['shuffled_data'] = df['shuffled_data'] = df.shuffled_data.astype(float)
            #add column specifying condition
            cond_col = [cond]*len(df)
            df['cond'] = cond_col
            #run anova
            df_fstat = df[['subj','shuffled_data','race', 'acc','cond']].groupby(df['permutation']).apply(run_anova)
            #reformat df_fstat to aid in plotting histogram in next function
            df_fstat = df_fstat.T.reset_index(drop = True).T
            df_fstat = df_fstat.rename(columns = {0:'f_stat'})
            store_f_statistics[cond][metric] = df_fstat
            
    return  store_f_statistics

def evaluate_significance(observed_effects, null_f_distributions, numPermutations):
    print("Evaluating Significance...\n")
    
    def plot_hist(observed_fstat, null_dist, cond, metric):
        
        #Construct string variables for histogram titles
        if 'enc' in cond:
            effect = 'Interaction of Race and Accuracy'
            condition = 'Lure Encoding'
        elif 'retr' in cond:
            effect = 'Main Effect of Accuracy'
            condition = 'Target Retrieval'
        if 'clust' in metric:
            measure = 'Network Clustering'
        elif 'eff' in metric:
            measure = 'Local Network Efficiency'
            
        #title = f'F-statistic Distribution: {effect} on {measure} during {condition}'


        #Constuct the plot - histogram using kernel density estimation of distibution
        #fig, ax = plt.subplots(figsize = (10,8))
        fig, ax = plt.subplots()
        plt.style.use('bmh')
        
        ax.axvline(x = observed_fstat, ymax = .5, linestyle = ":", color = 'red')
        ax.axvline(x = np.percentile(dist, 95), ymax = .25, linestyle = ":", color = 'red') 
        #ax.text(np.percentile(dist, 95)-.5,13, s = "95th\nPercentile", size = 10)
        #ax.text(observed_fstat-.5,25, s = "Observed\nF-Stat", size = 10)

        #null_dist.plot(kind = "hist", bins = 15)
        null_dist.plot(kind = "hist", bins = 15)
        #Color scheme / style
        #set axis titla and label
        ax.set_title(f'F-statistic Distribution: {effect}\non {measure} during ' 
                     f'{condition}', fontsize=12, pad = 10)
        ax.set_xlabel('Value of F Statistic', fontsize=10)
        ax.set_ylabel(f'Frequency Across {numPermutations} Permutations', fontsize=10)
        
        # Formate axes, #remove ticks, lines and grids
        ax.set_xlim(0, round(observed_fstat)+ .5)  #F-dist begins at 0   
        ax.grid(False)
        ax.tick_params(left = False, bottom = False)
        for ax, spine in ax.spines.items():
            spine.set_visible(False)
        
        plt.savefig('hist_'+ cond + '_' + metric)
        plt.show()
        return
    #plot_hist(observed_fstat, dist.f_stat)
    

    print("Permuatation Analysis Results:\n")   
    print('Number of Permutations:', numPermutations, '\n')
    results_file = open("Results_Permutation_Analsysis.txt", "w")  
    results_file.writelines(['Results_Permutation_Analsysis\n\n',
                             f'Number of Permutations: {numPermutations}\n\n'])
    for cond in null_f_distributions.keys():
        for metric, df in null_f_distributions[cond].items():
            #print(cond, metric)
            if metric == 'clust_permutations':
                observed_fstat = observed_effects[cond]['f_clust']
            elif metric ==  'eff_permutations':
                 observed_fstat = observed_effects[cond]['f_eff']
            dist = null_f_distributions[cond][metric]
            
            #return observations under which the null distribution produced 
            #greater effects than the data that was observed
            count_greater_effects = dist.f_stat[dist.f_stat >= observed_fstat].count()
            total_observations =  dist.f_stat.count()
            p_val = count_greater_effects / total_observations
            #Calculate the percentile ranking of the observed F-stat
            percentile_rank = stats.percentileofscore(dist, observed_fstat)            

            print ("ANOVA:", cond, metric)
            print ("\tObserved F_stat:", observed_fstat)
            print("\tObseved F-stat is in the", str(percentile_rank) + "th percentile")
 
            results_file.writelines([f'ANOVA: {cond} {metric}\n',
                            f'\tObserved F-stat: {observed_fstat}\n',
                            f'\tObserved F-stat falls in the {percentile_rank}th '
                               'percentile of the null distribution\n'])   
            
            if p_val < .05:
                print('\tp-value:', p_val, '\n \tObserved results SIGNIFICANT***\n')
                results_file.write(f'\tp-value: {p_val}\n \tObserved results SIGNIFICANT***\n')
            else:
                print("\tp-value", p_val, '\n \tObserved Results NS' )
                results_file.write(f'\tp-value: {p_val}\n \tObserved results NS\n')

            #plot histogrqam of the data
            plot_hist(observed_fstat, dist.f_stat, cond, metric, )
    results_file.close()

    return
#########################################################################

#Structure data for subsequent analsis
enc_lure, retr_targ = wrangle_data()
#store data in object to pass into functions
observed_data = [[enc_lure, 'enc_lure'],[retr_targ, 'retr_targ']]


#Run RM Analysis of variance on observed data. Store F-stats for the effects
#of interest for permutation analysis. 
#I.e. F-stat for interaction of race and accuracy in lure encoding 
#     F-stat for main effect of accuracy during target retrieval
#       F_Stats can be retrieved later by 
#       observed_effects['enc_lure']['f_clust']
#       observed_effects['retr_targ']['f_eff']
observed_effects = anova_observed(observed_data) 

#Pass in observed data and generate randomize possible permuatations across 
#both lure and target conditions for both graph metrics. Store in dfs
#       Access dfs like permuted_dfs['enc_lure']['clust_permutations']
#       or              permuted_dfs['retr_targ']['eff_permutations']
permuted_dfs, numPermutations = define_permuations(observed_data, numPermutations = 10)

#The dataframes with all permuations still need to be structured in accordance
#with the function pingouin.anovaRM's  expectations. This requires the dfs to 
#be converted to long format (one observation per row), and to have categorical
#columns for the factors we are modelling (ie. race and accuracy).
    #Access dfs like structured_permuted_dfs['enc_lure']['clust_permutations']
    #or              structured_permuted_dfs['retr_targ']['eff_permutations']
structured_permuted_dfs = structure_permutations(permuted_dfs)

#Run ANOVAs on each dataframe of shuffled conditions and store F-stats for the
#same comparisons run through the anova_observed fucntion. 
#null_distributions = anova_permutations(structured_permuted_dfs)
    #Access dfs like null_f_distributions['enc_lure']['clust_permutations']
    #or              null_f_distributions['retr_targ']['eff_permutations']
null_f_distributions = anova_permutations(structured_permuted_dfs)


#Evaluate significance of observed results given null distributions
# and plot histograms
evaluate_significance(observed_effects, null_f_distributions,numPermutations)

print("Runtime:", datetime.now() - startTime)


   #permutations = full_df['permutation'].unique()    #df_eff_permutations = pd.DataFrame(eff_permutations)
#for perm_num in permutations:
#    print(full_df['permutation'][full_df['permutation'] == perm_num])'''
