# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:06:09 2021

@author: jyaros
"""
from datetime import datetime
import os
import numpy as np
import pandas as pd
import random

# Load and clean data. 
def fetch_data():
    
    #import graph metric data to df
    workingDir = os.getcwd()
    filename = 'input_efficiency_metrics.csv'
    file = os.path.join(workingDir, filename)
    df = pd.read_csv(file)
    # Clean column names
    df.columns = df.columns.str.lower()
    df.rename(columns = {'type':'trial_type'}, inplace = True)    
    return df

'''# Engineer columns for experimental conditions. 
The following functions take string labels and a priori knowledge of each 
condition to engineer new descriptive columns labeling the exact experimental 
conditions'''

# This function takes the string in the Network column and decomposes this into three 
#     descriptive condition columns
def engineer_categories_from_string(input):
    # Use string in Network col to create new columns that better describe the data
    # New columns will be used for later subsetting
    input['subj'] = input['network'].str[11:21]
    input['cond'] = input['network'].str[22:34]
    input['cost'] = input['network'].str[36:39]
    #Network column no longer neded now that information separated into feature columns
    #input.drop(input['network'], axis = 1, inplace = True)
    input = input.drop('network', axis =1)
    return input

# This function removes rows with costs corresponding to netwroks that do not have low conenction 
#     densities, as small world and non-random properties are most observable in low cost networks.
def reduce_costs(df):
    #convert costs from string to float
    df['cost'] = df['cost'].astype(float)
    #raw_data = 
    df = df.loc[df['cost'] <= .25]
    return df

# This function groups records by unique subject and condition combinations, and averages all data within  
#     the graph and local efficiency columns. This creates average efficiency metrics across the five 
#     topological cost conditions. 
def average_across_costs(df):
    # Group and average across unique subject/condition combinations
    df = df.groupby(['subj', 'cond']).mean()
    df.reset_index(inplace=True)
    #drop cost column, which is now, just an average of all five costs, and no longer meaningful
    df = df.drop('cost', axis =1)
    return df

# This fucntion categorize recrods based on nature of encoding/retrieval, lure/target, accurate/innacurate 
#     and other-race/same-race conditions. Function  uses a priori knowledge of CONN labels to hard-code 
#     the conditions. 
def engineer_condition_features(df):

    enc_cols = ['Condition001','Condition002','Condition003','Condition004','Condition005',\
           'Condition006','Condition007','Condition008']

    lure_cols = ['Condition001', 'Condition002', 'Condition005','Condition006','Condition009',\
            'Condition010', 'Condition013', 'Condition014']

    acc_cols = ['Condition001', 'Condition003', 'Condition005', 'Condition007', 'Condition009',\
           'Condition011', 'Condition013', 'Condition015']

    or_cols = ['Condition001', 'Condition002', 'Condition003', 'Condition004', 'Condition009', \
          'Condition010', 'Condition011', 'Condition012']

    df['phase'] = np.where(df['cond'].isin(enc_cols), 'enc', 'retr')
    df['trial_type'] = np.where(df['cond'].isin(lure_cols), 'lure', 'target')
    df['accuracy'] = np.where(df['cond'].isin(acc_cols), 'corr', 'incorr')
    df['race'] = np.where(df['cond'].isin(or_cols), 'or', 'sr')
    return df

'''# Quality Control
At this point run quality control checks to comfirm that records are labeled 
correctly. This includes random spot checks, and confirming that the number of
 categories match our expectations.

For instance, the dataframe should have one row for each unique subject/condition 
combo, given that there are 22 subjects and 16 conditions, there should be 
exactly 352 rows. Furthermore, every unique combination of phase*trial_type 
condition should have equal numbers of SR and OR records. The sanity_check 
function, checks all these expectations, and returns errors if they are not met.'''

# if dataframe does not have exactly 352 rows, return error message
def sanity_check(df):
    # Establish df is correct overall size
    expected_count = 16 * 22
    if df.shape[0] == expected_count:
        print(f"QC Pass: Datframe has correct number of rows: {expected_count}")
    else:
        print("Error: Dataframe is not the right size!")
    
    # Establish that each subject has 16 unique conditions, adn each condition has 22 unique subjects. 
    #     If there are any deviations from these constraints, throw error
    
    # Create bools to check that condition and subject counts are as expected
    assess_cond_num = df_final.groupby(['subj']).size() == 16
    assess_subj_num = df_final.groupby(['cond']).size() == 22
    # If ALL records are not true (if even one is False, throw error
    if assess_cond_num.all() == False:
        print ("Error: At least one subject does not have 16 conditions")
    else:
        print("QC Pass: All subjects have 16 conditions")
        
    if assess_subj_num.all() == False:
        print ("Error: At least one conditiomn does not have 22 subjects")
    else: 
        print("QC Pass: All conditions have 22 subjects")
        
    # Confirm that each condition has the expected and same number of SR and OR records. If not throw error.
    #     This confirms the repeated measured nature of our experiment and MUST be correct. If counts do not equal
    #     If counts do not equql 22 as expected, throw error and return counts. If counts are as expected, do nothing
    for dv in ['global_efficiency', 'avg_local_efficiency']:
        for phase in ['enc', 'retr']:
            for trial_type in ['lure','target']:
                for i, accuracy in enumerate(['corr', 'incorr']): 
                    #err_count = 0
                    #print(i,dv, phase, trial_type, accuracy)
                    df_subset =df_final[['subj','phase','trial_type', 'race', 'accuracy',dv]].loc[(df_final.phase == phase) & (df_final.trial_type == trial_type) & (df_final.accuracy == accuracy)]
                    assess_rm_race = df_subset['race'].value_counts() == 22
                    #if at least SR or OR does not have 22 records return error.
                    if assess_rm_race.all() == False:
                        print(f'Error in expected race counts of {phase} {trial_type} {accuracy} : {dv}')
                        print(df_subset['race'].value_counts())

# Lastly for good measure lets to a random spotcheck and manually confrimt the returned conditions are coded correctly. 
#     Here I am just cross-referencing the table with the coding scheme for each condition.

#df_final.sample(30).sort_values(by=['cond'])                       

'''
# Permutation Testing to Evaluate Significance of Observed Results
Because each cluster of data within our models is very small (22 observations), 
it is more than possible that any effects deemed significnat by our glm, could
be due to chance. This is compounded by the fact that one of our factors is 
accuracy, and we know from our behavioral analysis that participants performed 
fairly poorly. They seemed to perform above chance, but only slightly. So we 
have to be wary of how strong a signal we can get in the neroimaging results 
of conditions where responses could be due to chance. Despite this weaker 
behavioral signal, some of our z scores were above 2 SD from the distribution 
expected if the null hypothesis that the race/accuracy interaction has no 
effect on the depenedent variable were true. Becuase these results indicate 
an MR signal might be sensitive to accuracy (And race) despite low performance, 
we should do our best to establish that these test statistics could not be 
commonly reproduced purely by shuffling (randomizing) the condition labels 
on our data.

Therefore to establish significance of any interaction of race/accuracy, for 
each previously modeled conditions we will randomize the datasets to create 
10,000 permutations of shuffled data. Then, the glm will be fit to each of 
these datasets, and the Z statistic for each fitted model will be saved. 
Then we will generate the (10,000 item) Z distbrutions of all 8 conditions, 
and use the saved Z-statistics for the observed (non-shuffled) data to 
calculate p-values.

Finally we will use a FDR correction to ajdust p-valeus to reduce the chance 
of falsely claiming an interaction is significant (false positive). 
(Either use B-H or B-Y based on assessment of independence).'''

#This function generates random permutations of shuffled data and stores in 
#dfs for each graph metric and each condtion
def generate_permutations(cond, df, num_permutations): 
    startTime = datetime.now()
    # Use condition string to create phase and trial_type strings
    phase = cond.split("_")[0]
    trial_type = cond.split("_")[1]
    
    print(f"Generating and storing {phase} {trial_type} permutations...")
    
    #Create sub-dictionaries for each efficiency metric
    dict_global_efficiency_permutations = {}
    dict_local_efficiency_permutations = {}
    
    #define unique subjects
    subjects = df['subj'].unique().tolist()
    
    # We must shuffle conditions WITHIN rather than across subjects due to the 
    #      paired/RM nature of the experimental design. If all observations were 
    #      independent of one another we could instead shuffle the whole dataset.
    #      In effect, we are taking the four datapoints for each subject, and 
    #      randomnly shuffling and relabeling them 10,000 times. 
    for subj in subjects:
        global_efficiency = df['global_efficiency'].loc[(df['subj'] == subj) & \
                    (df.phase == phase) & (df.trial_type == trial_type)].tolist()        
        #local_efficiency = df['avg_local_efficiency'].loc[df['subj'] == subj].tolist()
        local_efficiency = df['avg_local_efficiency'].loc[(df['subj'] == subj) & \
                    (df.phase == phase) & (df.trial_type == trial_type)].tolist()   

        #Create dictionaries for temporary storage of each subejcts shuffled data,
        #where keys are the permutation number and values are the shuffled list of date
        subj_global_permutations = {}
        subj_local_permutations = {}

        #Randomly shuffle the subject's four datapoints 10,000 times, storing each
        for iteration in range(num_permutations):
            #shuffle order of cluster and local efficiency metrics by randomly
            #sampling both lists
            global_permutation, local_permutation = [random.sample(metric,len(metric)) \
                        for metric in [global_efficiency,local_efficiency]]

            #store current iteration's randomized data
            subj_global_permutations[iteration] = global_permutation
            subj_local_permutations[iteration] = local_permutation  

        #For each subject, store these permutations in dictionaries where all subject's data
        #     is aggregated. Key is subject number and value is the efficiency metric.
        dict_global_efficiency_permutations[subj] = subj_global_permutations
        dict_local_efficiency_permutations[subj] = subj_local_permutations
        
    # Convert these all encompassing dictionaries into dataframes, with columns defining the
    #     subject and iteration number corresponding to the shuffled data in each row. 
    #     some manipulation is needed to restucture data this way. After loading
    #     dictionaries into dataframes, transpose so that subjects are organized into
    #     separate rows, rather than columns. Then combine and stack the permutation columns 
    #     into  one column. Given 10,000 permutations, there will be 10,000 rows per subject.
    #     The shuffled data are still stored within lists, within the permutation column
              
    df_global_efficiency_permutations, df_local_efficiency_permutations = \
              [pd.DataFrame(df).T.stack().to_frame().reset_index() for df in \
              [dict_global_efficiency_permutations, dict_local_efficiency_permutations]]
    
    # Assign meaningful column names
    df_global_efficiency_permutations, df_local_efficiency_permutations = \
    [df.rename(columns = dict(zip(df.columns.values, \
    ['subj', 'permutation', 'shuffled_data']))) \
    for df in [df_global_efficiency_permutations, df_local_efficiency_permutations]]
    
    #store both dataframes in one dictionary. These will then be stored under
    #the corresponing condition, upon return from this function
    df_dictionary = {}
    df_dictionary['global_efficiency'] = df_global_efficiency_permutations
    df_dictionary['local_efficiency'] = df_local_efficiency_permutations
    print("Runtime:", datetime.now() - startTime)
    return df_dictionary, num_permutations

# This function calls the generater_permutation function for each condition. It then stores
#     the returned dictionary of dataframes into a parent dictionary called 'permuations' 
#     with keys set to condition labels. As such, any dataframe can be accessed via a 
#     series of two keys. For example:
#          permutations['enc_lure'][local_efficiency]
#          permuted_dfs['retr_targ']['global_efficiency']
def run_generate_permutations():
    permutations = {}
    condition_list =  ['enc_lure', 'enc_target', 'retr_lure', 'retr_target']
    #condition_list  = ['enc_lure']
    for cond in condition_list:
        permutations[cond], num_permutations =  generate_permutations(cond, df_final, num_permutations = 10000)
        #store in csvs for later access
        permutations[cond]['global_efficiency'].to_csv(os.path.join('output', f'{cond}_global_eff_permutations.csv'))
        permutations[cond]['local_efficiency'].to_csv(os.path.join('output', f'{cond}_local_eff_permutations.csv'))
        print ("Saved to csv ...\n")
    return permutations, num_permutations

# Given a priori knowledge of the number of permutations we ran and the number of 
#     subjects we should be able to accurately calculate the number of rows in the
#     dataframes storing permutations. Also do spotchecks. This is just a crude test
#     to ensure the dfs meet our expectations
def sanity_check_confirm_permutation_count(permutations, num_permutations):
    for cond_key in permutations.keys():
        for metric_key in permutations[cond_key].keys():
            print(cond_key, metric_key)
            
            #Run QC calculations
            df = permutations[cond_key][metric_key]
            num_subjects = len(df['subj'].unique().tolist())
            num_expected_subjects = 22
            num_rows = len(df)
            num_expected_rows = num_expected_subjects * num_permutations
            
            #Return Pass or Fail statements
            if num_subjects == num_expected_subjects:
                print ('QC Pass: Number of subjects meets expectations')
            else: 
                print(f'ERROR in subject count. There are {num_subjects} instead of {num_expected_subjects} subjects')

            if num_rows == num_expected_rows:
                print ('QC Pass: Number of rows meets expectations')
            else:
                print (f'ERROR in number of rows. There are {num_rows} instead of {num_expected_rows} rows.')
            print('\n')
    return

'''So far we've generated and stored 10,000 permutations per subject in 
dataframes and csvs. However the structure of this data still needs to be 
manipulated to be compatable for running the regression analysis. 

We need to group subejects together by permutation number such that we have 
10,000 full  datasets per condition to fit with glm models. We further need 
to get data formatted like df_final, where each row corresponds to a trial, 
and labels the conditions of that trial. As of now, each row of our 
permutations df inlcludes a full list of 4 shuffled trials. These must be 
decomposed into four separate rows and labeled. Then, we will be able to 
fit the data to 10000 separate linear models per condition.'''

# Goal: Retructure each permutation df. Each row contains lists of 
#     length 4 (one element for each condition). Goal is to 'explode' lists
#     such that each condition has a dedicated row. The permutation number
#     will still be maintained by the index, which will no longer be unique/
#     Function returns a df that is stored in a master dictionary in the 
#     run_transform_permutations function, nested under condition and metric
#     keys. 
# Note this is a slow process at ~ 45 minutes per dataframe. There is surely 
#     a more efficient solution that cycling through each row and exploding 
#     them iteratively. However in the interest of not spending too much time
#     recoding something I'll only use once, I'm going to stick with it. 
#     I tested out using apply, and running the iterations within groped 
#     permutations, but I'm getting a duplicate axis error, so will stick
#     with this slow way for not
def transform_permutations(df,num_permutations):
    startTime = datetime.now()
    
    #Intialize new df for restructured data 
    df_restructure =  pd.DataFrame()
    
    #Cycle through each permutation (i.e. row)
    for row in df.index:
        permutation = df.iloc[[row]]
        # This takes the list of values in the shuffled_data column for this specific row
        #     and redistributes the values arcross 4 separate rows. Permuation number and
        #     subject number will remain constant across these rows.
        permutation_explode = permutation.set_index('permutation')\
                                    .apply(pd.Series.explode).reset_index()
        
        #Append expanded permutation to new dataframe, where each original row in the 
        #     permutation df is now 4 rows in df_explode        
        df_restructure = df_restructure.append(permutation_explode)  
                      
    # Next assign race and accuracy labels to each observation. Data are already 
    #     randomized so the order of condition assignment is arbitrary. The only 
    #     contraitnt is that each permutation (ie each set of four rows must 
    #     contain exactly one of each of the four conditions : SR Corr, SR InCorr, 
    #     OR Corr, OR InCorr. We do this by initializing two lists that will be 
    #     converted into column labels, satisfying the constraint. Before conversion
    #     to columns, we extend the list by the length of total permuations. This way,
    #     when we convert to columns, each row will have a unique label (per permutation) 
    #     satisfying the constraint.

    # Initialize the four required conditions per each permutation
    race = ['SR', 'SR', 'OR', 'OR']
    accuracy = ['Corr', 'InCorr', 'Corr', 'InCorr']

    # Calculate number of total permutations across the full subject sample
    num_subjects = len(df['subj'].unique().tolist())
    num_distinct_permutations =  num_subjects * num_permutations

    # Create lists for race and accuracy label assignment, with length
    #     equal to the total number of permutations. Assign as columns       
    race_col = race * num_distinct_permutations
    accuracy_col = accuracy * num_distinct_permutations

    #Assign these lists to new columns in df_restructure
    df_restructure['race'], df_restructure['accuracy']  = race_col, accuracy_col
    df_restructure = df_restructure.reset_index()

    print("Runtime:", datetime.now() - startTime)

    return df_restructure

def run_transform_permutations():
    restructured_permutations = {}
    for cond_key in permutations.keys():
        restructured_permutations[cond_key] = {}
        for metric_key in permutations[cond_key].keys():
            print(f"Restructuring permutation data for {metric_key} {cond_key}s...")
            restructured_permutations[cond_key][metric_key] = transform_permutations(permutations[cond_key][metric_key], num_permutations)    
            restructured_permutations[cond_key][metric_key].to_csv(os.path.join('output', f'restructured_{cond_key}_{metric_key}.csv'))
            print('Saved to csv...')
    return restructured_permutations

# At this point the restured permutations were exported to csv. Load data back 
#     into a dictionary of dataframes for additional analysis. (Should have 
#     saved dfs to dictionary in the run_transform_permutations function but
#     forgot to implement. Function took ~ 8 hours so faster to read data back 
#     in this way.)
def fetch_restructured_permutations():
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir, "output")
    #grab restructure permutation files based on csv file neme
    files = [file for file in os.listdir(outputDir) if \
             file.startswith("restructured")]
    #create dictionary to store dataframes
    restructured_permutations = {}    
    
    #Cycle through files, label conditions and assign to dictionary
    for file in files:
        # Use file name to create phase, trial_type, and metric labels
        print (f'Importing {file}')
        phase = file.split("_")[1]    
        trial_type = file.split("_")[2]
        metric = file.split("_")[3] + '_efficiency'
        cond = phase + '_' + trial_type+ '_' + metric
        
        # Read in file to dataframe and adjust column names
        df = pd.read_csv(os.path.join(outputDir,file), index_col = 0)
        df.rename(columns = {'index':'index_explode'}, inplace = True)
        
        # Store df in dicitonary under condition label
        restructured_permutations[cond] = df
        
    return restructured_permutations

def sanity_check_restructured_dfs(restructured_permutations, num_permutations):
    expected_conditions = 8
    expected_subjects = 22
    expected_permutations = num_permutations
    exploded_rows = 4
    expected_rows = expected_subjects * expected_permutations * exploded_rows
    
    num_conditions = len(restructured_permutations)
    if num_conditions == expected_conditions:
        print (f'QC Pass: {num_conditions} conditions, as expected\n')
    else:
        print(f'ERROR: {num_conditions} instead of {expected_conditions} conditions\n')

    
    for key in restructured_permutations.keys():
        print(f'Checking {key}...')
        num_subjects =  len(restructured_permutations[key]['subj'].unique())
        num_permutations = len(restructured_permutations[key]['permutation'].unique())
        num_rows = len(restructured_permutations[key])

        if num_subjects == expected_subjects:
            print (f'QC Pass: {num_subjects} subjects, as expected')
        else:
            print(f'ERROR: {num_subjects} instead of {expected_subjects} subjects')

        if num_permutations == expected_permutations:
            print (f'QC Pass: {num_permutations} permutations,  as expected')
        else:
            print(f'ERROR: {num_permutations} instead of {expected_permutations} permutations')
            
        if num_rows == expected_rows:
            print (f'QC Pass: {num_rows} rows,  as expected')
        else:
            print(f'ERROR: {num_rows} instead of {expected_rows} rows') 
        print('\n')
    return

############################### FUNCTION CALLS ################################
''' Uncomment to call individual functions'''

''' Read in raw data for global and local efficiency'''
#input = fetch_data()

''' Engineer subject, condition, and cost columns'''
#df = engineer_categories_from_string(input)

''' Filter topoogiocal costs of interest '''
#df = reduce_costs(df)

''' Collate metrics across topological costs'''
#df =  average_across_costs(df)

''' Engineer columns to label phase, trial_type, race, and accuracy features.
This structures data to be compatable for regression analysis in statsmodels.
Export to csv'''
#df_final = engineer_condition_features(df)
#df_final.to_csv(os.path.join('output', "output_efficiency_metrics_for_analysis.csv"),\
#                             index = False)
''' Run QC checks to confirm df meets structural expectations'''
#sanity_check(df_final)

''' Generate dataframes with 10,000 possible shufflings of each participant's 
  phase * trial_type * accuracy conditions.  '''
#permutations, num_permutations = run_generate_permutations()

''' Run QC checks to confirm the dfs storing permutations meet structural 
expectations. ''' 
#sanity_check_confirm_permutation_count(permutations, num_permutations)

''' Restructure permutation in dfs to be compatable for glm regression 
analysis. Export to csv'''
#run_transform_permutations()

'''Read back in restructured permutation dfs, store in dictionary and run QC 
checks to confirm they meet structural expectations. '''
#restructured_permutations = fetch_restructured_permutations()
#sanity_check_restructured_dfs(restructured_permutations, num_permutations)