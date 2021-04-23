# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:26:08 2021

@author: jyaros
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('seaborn') # nicer plots
from scipy import stats
import numpy as np
import statsmodels.api as sm
import statsmodels.stats as sm_stats



''' Load in the osberved data as well as the randomly shuffled data '''
def fetch_data():  
    
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir, "output_cleaned_observed_and_permuted_data")
    
    #Read in actual observed data
    print ("Importing observed data...")
    filename = 'output_efficiency_metrics_for_analysis.csv'
    observed_data = pd.read_csv(os.path.join(outputDir, filename)) 
    
    #Read in the 8 dataframes with 10,000 permuted observations per condition
    files = [file for file in os.listdir(outputDir) if \
             file.startswith("restructured")]
    #create dictionary to store dataframes
    permuted_data_dictionary = {}    
    
    #Cycle through files, label conditions and assign to dictionary
    for file in files:
        # Use file name to create phase, trial_type, and metric labels
        print (f'Importing {file}...')
        phase = file.split("_")[1]    
        trial_type = file.split("_")[2]
        metric = file.split("_")[3] + '_efficiency'
        cond = phase + '_' + trial_type+ '_' + metric
        
        # Read in file to dataframe and adjust column names
        df = pd.read_csv(os.path.join(outputDir,file), index_col = 0)
        df.rename(columns = {'index':'index_explode'}, inplace = True)
        
        # Store df in dicitonary under condition label
        permuted_data_dictionary[cond] = df
        
    print("\n")
    return observed_data, permuted_data_dictionary

''' Run QC checks on the observed data'''
def sanity_check_observed_data(df):
    print ('Quality Control Check for Observed Data:')
    # Establish df is correct overall size
    expected_count = 16 * 22
    if df.shape[0] == expected_count:
        print(f"QC Pass: Datframe has correct number of rows: {expected_count}")
    else:
        print("Error: Dataframe is not the right size!")
    
    # Establish that each subject has 16 unique conditions, adn each condition 
    #     has 22 unique subjects. If there are any deviations from these 
    #     constraints, throw error
    
    # Create bools to check that condition and subject counts are as expected
    assess_cond_num = df.groupby(['subj']).size() == 16
    assess_subj_num = df.groupby(['cond']).size() == 22
    # If ALL records are not true (if even one is False, throw error
    if assess_cond_num.all() == False:
        print ("Error: At least one subject does not have 16 conditions")
    else:
        print("QC Pass: All subjects have 16 conditions")
        
    if assess_subj_num.all() == False:
        print ("Error: At least one conditiomn does not have 22 subjects")
    else: 
        print("QC Pass: All conditions have 22 subjects")
        
    # Confirm that each condition has the expected and same number of SR and OR 
    #     records. If not throw error. This confirms the repeated measured 
    #     nature of our experiment and MUST be correct. If counts do not equal
    #     If counts do not equql 22 as expected, throw error and return counts. 
    #     If counts are as expected, do nothing.
    for dv in ['global_efficiency', 'avg_local_efficiency']:
        for phase in ['enc', 'retr']:
            for trial_type in ['lure','target']:
                for i, accuracy in enumerate(['corr', 'incorr']): 
                    #err_count = 0
                    #print(i,dv, phase, trial_type, accuracy)
                    df_subset =df[['subj','phase','trial_type', 'race', 'accuracy',dv]].loc[(df.phase == phase) & (df.trial_type == trial_type) & (df.accuracy == accuracy)]
                    assess_rm_race = df_subset['race'].value_counts() == 22
                    #if at least SR or OR does not have 22 records return error.
                    if assess_rm_race.all() == False:
                        print(f'Error in expected race counts of {phase} {trial_type} {accuracy} : {dv}')
                        print(df_subset['race'].value_counts())
    print("\n")
    return

''' Run QC checks on the permuted data'''
def sanity_check_permuted_data(permuted_data_dictionary, num_permutations):
    print ('Quality Control Check for Permuted Dataframes:')
    #Initialize expected values from dataframe
    expected_conditions = 8
    expected_subjects = 22
    expected_permutations = num_permutations
    exploded_rows = 4
    expected_rows = expected_subjects * expected_permutations * exploded_rows
    
    # Calculate actual attributes of dataframe and return QC evaluations 
    
    # Establish whether there are dataframes for all 8 phase * trial_type conditions
    num_conditions = len(permuted_data_dictionary)
    if num_conditions == expected_conditions:
        print (f'QC Pass: {num_conditions} conditions, as expected\n')
    else:
        print(f'ERROR: {num_conditions} instead of {expected_conditions} conditions\n')

    for key in permuted_data_dictionary.keys():
        print(f'Checking {key}...')
        num_subjects =  len(permuted_data_dictionary[key]['subj'].unique())
        num_permutations = len(permuted_data_dictionary[key]['permutation'].unique())
        num_rows = len(permuted_data_dictionary[key])

        # Establish whether all subjects are included
        if num_subjects == expected_subjects:
            print (f'QC Pass: {num_subjects} subjects, as expected')
        else:
            print(f'ERROR: {num_subjects} instead of {expected_subjects} subjects')
        
        # Establish whether number of permutations match expectations
        if num_permutations == expected_permutations:
            print (f'QC Pass: {num_permutations} permutations,  as expected')
        else:
            print(f'ERROR: {num_permutations} instead of {expected_permutations} permutations')
            
        # Establish that number of rows are correct, since each permutations was
        #     decomposed into 4 separate rows
        if num_rows == expected_rows:
            print (f'QC Pass: {num_rows} rows,  as expected')
        else:
            print(f'ERROR: {num_rows} instead of {expected_rows} rows') 
        print('\n')
    return


''' EDA: Understand Data Distributions and Detect Outliers
    Before modeling the observed data, I'd like to familiarize myself with 
    the dataset and their quirks a bit more. For instance, we should 
    establish whether efficiency metrics appear to be drawn from a normal 
    distribution, as this could influence the statistical test we decide to 
    run. Further lets see if there are any  egregious outliers, and if so, 
    if any one subject appears as an outlier repeatedly across conditions.''' 

# The test_distributional_assumptions function plots histograms and boxplots 
#     for each subset of data that I am modeling. In particular it separates 
#     data into the eight phase*trial_type conditions , but data is still 
#     collapsed across accuracy and race. Since I will model accuracy and race,
#     I'm not 100% positive whether I should be looking at the distribution of 
#     data within each factor i intend to model. Doing so would reduce each 
#     distribution to 22 rather than 88 datapoints, which is too small to get
#     an accurate idea of distribution shape. Considering this, we must keep in 
#     mind that these distributions are constructed from repeated measures data, 
#     and include 4 datapoints per subject.
#
# The function also runs a Shapiro-wilk normality test for each distribution, 
#     and labels the x-axis of the histogram with the result of the normality 
#     test.
#
# Lastly, the function creates a boxplot for each distribution to aid in 
#     identification of 
#     outliers.
#
# The run_eda function calls test_distributional_assumptions for each condition
#     condition*graph metric combination.

def test_distributional_assumptions(df, dv, phase, trial_type, outputDir):
    df_subset =df[['subj', 'race', 'accuracy', dv]].loc[(df.phase == phase) & (df.trial_type == trial_type)]
    # Initalize histogram
    fig, ax = plt.subplots()
    df_subset[dv].plot(kind = "hist")
    # Set axis title and label
    ax.set_title(f"{dv} for {phase} {trial_type} trials")
    ax.set_ylabel('Frequency')

    # Run Shapiro-Wilk normality test, which tests the null hypothesis that the 
    #     data was drawn from a normal distribution.
    stat, p = stats.shapiro(df_subset[dv])
    alpha = 0.05
    if p > alpha:
        ax.set_xlabel(f'Distribution looks Gaussian, p-val: {p:.3f}', fontsize=14)
    else:
        ax.set_xlabel(f'Distribution Not Gaussian, p-val: {p:.3f}', fontsize=14)
        
    #display and save histograms   
    plt.savefig(os.path.join(outputDir, f'dist_{dv}_{phase}_{trial_type}.png'), \
                format = 'png', transparent= True)            
    plt.savefig(os.path.join(outputDir,f'dist_{dv}_{phase}_{trial_type}.pdf'), \
                format = 'pdf', transparent = True)    
    plt.show()
    #display and save boxplot
    boxplot = df_subset[dv].to_frame().boxplot()
    fig = boxplot.get_figure()
    fig.savefig(os.path.join(outputDir,f'boxplot_{dv}_{phase}_{trial_type}.png'), \
                format = 'png', transparent = True)    
    fig.savefig(os.path.join(outputDir,f'boxplot_{dv}_{phase}_{trial_type}.pdf'), \
                format = 'pdf', transparent = True)    
                                
    return

# Call test_distributional_assumptions on each condition.
def run_eda(df):
    # Create directory to store plots
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir,'output_distributions')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    # For each set of conditions we will be modeling, create distributions and
    # boxplots by calling 'test_distributional_assumptions' function
    for dv in ['global_efficiency', 'avg_local_efficiency']:
        for phase in ['enc', 'retr']:
            for trial_type in ['lure','target']:
                test_distributional_assumptions(df, dv, phase, trial_type, outputDir)
    return

# This outlier_detection function  identifes outliers according to the IQR 
#     outlier detection rule, where an oultier is defined as 1.5 times the IQR
#     above or below the 75th and 25th percentiles, respectively. It then 
#     returns all records that are identified as outliers
def outlier_detection(df, dv, phase, trial_type):
    df_subset =df[['subj', 'race', 'accuracy', dv]].loc[(df.phase == phase) & (df.trial_type == trial_type)]
    # calculate interquartile range
    q25, q75 = np.percentile(df_subset[dv], 25), np.percentile(df_subset[dv], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outlier_records = df_subset.loc[(df_subset[dv] <lower) | (df_subset[dv] > upper)]
    print(f"For {phase} {trial_type}, identified {outlier_records.shape[0]} outliers")
    return outlier_records

# The following function creates a table of all records that are detected as  
#     outliers WITHIN the data subsetted by phase * trial condition. (Ie. the 
#     same subsets of data plotted as histograms and boxplots.) Prints out
#     statments of how many records per conditions were identified as outliers. 
#     Alse assembles table 'df_outliers' with all outlier records, and exports
#     to, csv as 'observed_data_outliers.csv', into the output directory.
def run_outlier_detection(df):
    # Initialize empty df to store records indentified as outliers
    df_outliers = pd.DataFrame(columns = ['subj','race','accuracy', 'dv', 'phase','trial_type'])
    # Cylce through conditions
    for dv in ['global_efficiency', 'avg_local_efficiency']:
        print(dv)
        for phase in ['enc', 'retr']:
            for trial_type in ['lure','target']:
                # Identify outliers using outlier_detection function
                outlier_records = outlier_detection(df, dv, phase, trial_type)
                # Reset index for the returned records 
                outlier_records.reset_index(inplace = True, drop = True)
                # If there are outliers...
                if len(outlier_records) > 0:
                    # Generate dataframe of the conditions that these records 
                    #     correspond to
                    [col_dv, col_phase, col_trial_type] = [[col] * len(outlier_records) for col in [dv, phase, trial_type]]
                    col_dict = {'dv':col_dv, 'phase':col_phase, 'trial_type':col_trial_type}
                    outlier_conditions = pd.DataFrame(col_dict)
                    # Append condition infomation to outlier_records. Remove 
                    #     actual metric value columns from outlier records
                    outlier_records = outlier_records.iloc[:,:3].join(outlier_conditions)
                    # Append fully labeled recoeds to the final df_outliers table
                    df_outliers = df_outliers.append(outlier_records, ignore_index = True)

    #Save outlier records to csv.
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir, "output_cleaned_observed_and_permuted_data")
    df_outliers.to_csv(os.path.join(outputDir,"observed_data_outliers.csv"))
    print ('\nSaved outlier records to output directory...')
    return df_outliers

# This function groups outliers by condition and prints out count statements 
#     to aid in assessment of patterns in outliers. 
def outlier_pattern_detection(df_outliers):
    for col in df_outliers.columns:
        print (f'Outlier counts by {col}:')
        print(df_outliers[col].value_counts())
        print ('\n')
    return

# This function takes in a condition and fits a glm wih gee to the global and 
#     local efficiency metric data for that condition. The model is defined as 
#     dv = factor_1 + factor 2 + factor_1 * factor_2. Returns a nested 
#     dictionary with results stored under the key for the metric. Results for 
#     each metric include the full regression results, as well as the Z-stat 
#     for the race * accuracy interaction.
def gee_analysis(cond, df):
    # Use condition string to create phase and trial_type variables
    phase = cond.split("_")[0]
    trial_type = cond.split("_")[1]
    
    # Initialize dictionary to store results
    glm_results = {}  
    
    # Subset data using the current condition, for both efficiency metrics
    for dv in ['global_efficiency', 'avg_local_efficiency']:
        df_subset =df[['subj', 'race', 'accuracy',dv]].loc[(df.phase == phase) & (df.trial_type == trial_type)]
        
        # Initialize the generalized linear model with generalized estimating
        #     equations.
        lin_equation = f'{dv} ~ accuracy + race + accuracy*race'
        # Below equation takes log transform of dv, but does not improve residuals
        #lin_equation = f'np.log({dv}) ~ accuracy + race + accuracy*race'
        glm = sm.GEE.from_formula(lin_equation, groups = 'subj',data = df_subset, cov_struct = sm.cov_struct.Exchangeable())
                    
        # Fit the data to the model and save full regression and z statistic 
        #     for interaction of race and accuracy to dictionary
        dv_result = {}
        fit_data = glm.fit()
        zstat_interaction = fit_data.tvalues['accuracy[T.incorr]:race[T.sr]']
        dv_result['regression_results'] = fit_data
        dv_result['zstat_interaction'] = zstat_interaction
        glm_results[dv] = dv_result
    return glm_results

# This function calls the gee_anlaysis function on each condition and stores 
#     all data in one all encompassing dictionary, 'glm_results_all_conditions'.
def run_gee_analysis(df):
    results_all_conditions = {}
    condition_list =  ['enc_lure', 'enc_target', 'retr_lure', 'retr_target']
    # Perform glm/gee regression on each condition
    for cond in condition_list:
        results_all_conditions[cond] =  gee_analysis(cond, df)
    return results_all_conditions

#This function assesses normlaity of each models residuals. Generates report
#     of Shapiro-Wilk results, and QQ-Plot saved to 'output_model_fit' 
#     directory.
def assess_model_fit(glm_results):
    #Create directory to store results. 
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir,'output_model_fit')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    # Write Shapiro-Wilk results to this logile, saved in output_model_fit directory
    file_path = os.path.join(outputDir, 'Results_Normality_Model_Residuals.txt')
    log_file = open(file_path, "w")  
    log_file.write('Results of Shapiro-Wilk Normality Test on Model Residuals:\n\n')
    
    # For each condition and dependent variable
    for cond_key in glm_results.keys():
        for metric_key in glm_results[cond_key].keys():
            print(f'{cond_key} {metric_key} residuals')
            log_file.write(f'{cond_key} {metric_key} residuals\n')
            # Assess normality of residual distirbution 
            stat, p = stats.shapiro(glm_results[cond_key][metric_key]['regression_results'].resid)
            alpha = 0.05
            if p > alpha:
                print (f"Normal, p = {p}\n")
                log_file.write(f"Normal, p = {p}\n\n")
            elif p < alpha:
                print (f"Not normal, p = {p}\n")
                log_file.write(f"Not normal, p = {p}\n\n")
            # Generate QQ-Plot of residuals
            stats.probplot(glm_results[cond_key][metric_key]['regression_results'].resid, plot=plt)
            
            #save and display plots

            plt.savefig(os.path.join(outputDir, f'QQ_resid_{metric_key}_{cond_key}.png'), \
                format = 'png', transparent= True)            
            plt.savefig(os.path.join(outputDir,f'QQ_resid_{metric_key}_{cond_key}.pdf'), \
                format = 'pdf', transparent = True)   
            plt.show()
   
    log_file.close()

    return     

# The permutation_analysis function takes in a single dataframe with 10,000 
#     randomized susbets of the observed data subsest for a specific 
#     phase * trial_type condition. Then it applies the glm/gee model separately 
#     to each of the susbets, and produces a test statistic for the 
#     race * accuracy  interaction. It returns a dataframe with 10,000 test 
#     statistics. 
# The run_permutation_analysis function cycles through each condition, and calls
#     the permuation_analysis_function. It saves the results of each analysis
#     to a direcitory, 'output_permutation_analysis/test_statistics
def permutation_analysis(df):
    # For each dataframe subsetted/grouped by unique permutation number, 
    #     run regression analysis and return resulting test statistic.
    def run_gee_on_permuted_data(df_subset_by_permutation):
        # Initalize model and then fit to data
        lin_equation = 'shuffled_data ~ accuracy + race + accuracy*race'
        glm = sm.GEE.from_formula(lin_equation, groups = 'subj', \
              data = df_subset_by_permutation, cov_struct = sm.cov_struct.Exchangeable())
        fit_model = glm.fit()
        # Extract test_statistic
        zscore_interaction = fit_model.tvalues['accuracy[T.InCorr]:race[T.SR]']
        return zscore_interaction
    
    # Before analysis, dependent variable must be coverted to numerical datatype.
    #     Because it was in list format before, data is still string values.
    df['shuffled_data'] = df.shuffled_data.astype(float)
    # Model each unique permuation of data, and return test statistics in a series
    df_test_statistic = df.groupby(df['permutation']).apply(run_gee_on_permuted_data)
    # Resturcutre series to df and rename columns
    df_test_statistic = df_test_statistic.to_frame().rename(columns = {0: 'z_score'})
    return df_test_statistic

# Call permutation_analysis on each condition and export results to csv
def run_permutation_analysis(permuted_data_dictionary):
    #Create directory to store results. 
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir,'output_permutation_analysis', 'test_statistics')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    permutation_test_results = {}
    for cond in permuted_data_dictionary.keys():
        print (f'Modeling Permuted Data for {cond}...')
        permutation_test_results[cond] =  permutation_analysis(permuted_data_dictionary[cond])      
        permutation_test_results[cond].to_csv(os.path.join(outputDir, \
                                f'permutation_test_statistics_{cond}.csv'))
    print ("Permutation Analysis complete")
    return permutation_test_results




# Confirm correct number of z-scores per condition, given 10,000 permutations.
def sanity_check_permutation_results(permutation_test_results):
    expected_permutations = 10000
    for cond in permutation_test_results.keys(): 
        print (f'Checking {cond}...')
        num_permuatations = permutation_test_results[cond].index.nunique()
        num_rows = len(permutation_test_results[cond])
        if (num_permuatations == expected_permutations) & (num_rows == expected_permutations):
            print ('QC Pass: There is a z-score for every permutation\n')
        else:
            print ('Error: Something is off. Investigate further\n')
    return
  
# For each condition, calculate the poprotion of times the observered 
#     z-score was equal to or higher than z-scores in the distribution of 
#     z-scores from the permutations. This produces a p-value from which to 
#     assess significance (though it still needs to be ajdusted for multiple
#     tests.
# Additionally plot historgram of the z-distributions with a merker for the
#     location of the observed z-score as well as the 95th percetile above 
#     which the z-score must fall to be deemed signifincant (pre-adjustment).
# Save results in a log_file and histogram named with a suffix of the 
#     current condition to:
#     '/output_permutation_analysis/significance_evaluation'
# The run_evaluate_significance function acutally calls evaluate significance
#     on each condition. Like any function that has a corresponding run 
#     function, evaluate significance can also be run on individual conditions, 
#     by bypassing the run_evaluate_significnace function and calling it on 
#     the condition of interest. 
def evaluate_significance(z_score, z_distribution, cond, metric):
    #Create directory to store results. 
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir,'output_permutation_analysis', 'significance_evaluation')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    log_file_path = os.path.join(outputDir, f"results_perm_test_{cond}_{metric}.txt")

        
    # Function to plot historgrams of distirbutions and observed z-score location
    def plot_hist(z_score, z_distribution, cond, metric, num_permutations):
        #Constuct the plot
        fig, ax = plt.subplots()
        plt.style.use('bmh')
        
        # Place lines at location of observed z-score and 95th percentile
        #ax.axvline(x = z_score, ymax = .5, linestyle = ":", color = 'red')
        #ax.axvline(x = np.percentile(z_distribution, 95), ymax = .25, linestyle = ":", color = 'red') 
        #ax.text(np.percentile(dist, 95)-.5,13, s = "95th\nPercentile", size = 10)
        #ax.text(observed_fstat-.5,25, s = "Observed\nF-Stat", size = 10)
        
        # Plot distribution
        z_distribution.plot(kind = "hist", bins = 15)
        #ax.hist(z_distribution)
        ax_histx.hist(z_distribution)
        ax.axvline(x = z_score, ymax = .5, linestyle = ":", color = 'red')
        #ax.axvline(x = np.percentile(z_distribution, 95), ymax = .25, linestyle = ":", color = 'red') 


        # Set axes titles and labels
        effect =  'Interaction of Race and Accuracy'
        ax.set_title(f'Z-score Distribution: {effect}\non {metric} during ' 
                     f'{cond}', fontsize=12, pad = 10)
        ax.set_xlabel('Value of Z-Score', fontsize=10)
        ax.set_ylabel(f'Frequency Across {num_permutations} Permutations', fontsize=10)
        
        # Formate axes, #remove ticks, lines and grids for prettyniess
        #ax.set_xlim(0, round(observed_fstat)+ .5)  #F-dist begins at 0   
        ax.grid(False)
        ax.tick_params(left = False, bottom = False)
        for ax, spine in ax.spines.items():
            spine.set_visible(False)
        
        plt.savefig('hist_'+ cond + '_' + metric)
        plt.show()
        return
    
    print("Permutation Analysis Results:\n")   
    num_permutations = z_distribution.index.nunique()
    log_file = open(log_file_path, "w")  
    log_file.writelines(['Results_Permutation_Analsysis\n\n',
                             f'Number of Permutations: {num_permutations}\n\n'])
    
    # Establish whether z-score is positive or negative to determine which 
    #      direction to make calculation in. The count the number of times 
    #     z-scores in the z-distrbution weere equal to or 
    #     greater/less (depending on direction) than the observed z-score.
    if z_score > 0 : # if positive
         count_greater_effects = z_distribution[z_distribution >= z_score].count()[0]
    elif z_score < 0 : # if negative
         count_greater_effects = z_distribution[z_distribution <= z_score].count()[0]
    # Calculate p-value which is the PROPORTION of times the distribution has 
    #     a z-score more extreme than the observed one
    p_val = count_greater_effects / num_permutations
    #Calculate the percentile ranking of the observed z-score
    percentile_rank = stats.percentileofscore(z_distribution, z_score)            

    print ("Analysis of:", cond, metric)
    print ("\tObserved z_score:", z_score)
    print("\tObseved z_score is in the", str(percentile_rank) + "th percentile")
 
    log_file.writelines([f'GLM/GEE Permutation Test: {cond} {metric}\n',
                    f'\tObserved z-score: {z_score}\n',
                    f'\tObserved z-score falls in the {percentile_rank}th '
                      'percentile of the z-distribution\n'])   
    
    if p_val < .05:
        print('\tp-value:', p_val, '\n \tObserved results SIGNIFICANT***')
        log_file.write(f'\tp-value: {p_val}\n \tObserved results SIGNIFICANT*** ')
    else:
        print("\tp-value", p_val, '\n \tObserved Results NS' )
        log_file.write(f'\tp-value: {p_val}\n \tObserved results NS ')
    print(f"\twith an uncorrected p_value of {p_val}")
    log_file.write(f"with an uncorrected p_value of {p_val}.")
    #plot histogrqam of the data
    #NOT WORKING AT THE MO - FIGURE OUT LATER
    #plot_hist(z_score, z_distribution, cond, metric, num_permutations)
    log_file.close()
    return p_val

# Call evaluate significance funciton on each condition
def run_evaluate_significance(glm_results_all_conditions, permutation_test_results):
    p_values = {}
    for key in permutation_test_results.keys(): 
        # Create condition and metric strings that matche the labeling used in 
        # glm_results_all_conditions
        phase = key.split("_")[0]
        trial_type = key.split("_")[1]
        cond = phase + "_" + trial_type
        metric = key.split("_")[2] + "_" +  key.split("_")[3]
        
        #correct for difference in glm_results labeling of local eff metrics
        if "local" in metric:
            metric2 = "avg_" + metric
        else:
            metric2 = metric
        #Call evaluate_significance function
        print (f'\nEvaluating significance of {cond} {metric} interaction...')
        p_val = evaluate_significance(glm_results_all_conditions[cond][metric2]['zstat_interaction'],\
                              permutation_test_results[key], cond, metric)
        p_values[key] = p_val
    return p_values

# This function reads in the uncorrected p_values from ther permuatation 
#     permuatation analysis and adjusts them to reduce the chance of false
#     discovery given that we ran tests for 8 conditions. Returns the corrected
#     a table with the uncorrect and corrected p_values for each conditon, and 
#     exports to csv in '/output_permutation_analysis/significance_evaluation'
def fwer_correction(p_values_uncorrected):
    # Convert p_value dictionary to list to numpy array
    p_vals_in = np.array(list(p_values_uncorrected.values()))
    # Run FWER correction, which returns boolean series of whether to reject
    #     the null for each test, as well as an array of adjusted p_values and
    #     lastly the bonferroni adjusted_alpha which is just .05/number of test.
    multiple_test_correction = sm_stats.multitest.multipletests(p_vals_in,\
                                alpha=0.05, method='bonferroni')
    # Assign results to different variables which will used to construct a df
    reject_null_bool =  list(multiple_test_correction[0])
    p_values_adjusted = list(multiple_test_correction[1]) 
    adjusted_alpha = multiple_test_correction[3]
    p_values_uncorr = list(p_vals_in)
    test_condition = list(p_values_uncorrected.keys())    
    #assemble dictionary in order we want columns of df to appear
    fwer_dict = {'test_condition' : test_condition,
                 'p_value_uncorr' : p_values_uncorr,
                 'adjusted-alpha' : adjusted_alpha,
                 'p_value_adjust' : p_values_adjusted,
                 'sig_reject_null' : reject_null_bool}
    fwer_df = pd.DataFrame(fwer_dict)
    
    #Path  to store results. 
    workingDir = os.getcwd()
    outputDir = os.path.join(workingDir,'output_permutation_analysis', 'significance_evaluation')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    fwer_df.to_csv(os.path.join(outputDir, "fwer_correction_significance_evaluation.csv"), index = False)
    print('Exported significance evaluations to csv...')
    return fwer_df
############################### FUNCTION CALLS ################################

''' Load in observed as well as random/permuted data. '''
observed_data, permuted_data_dictionary = fetch_data()

''' Run crude QC checks on observed and permuted data confirm they meet 
    structural expectations. '''
sanity_check_observed_data(observed_data)
sanity_check_permuted_data(permuted_data_dictionary, num_permutations= 10000)

''' Exploratory Data Analysis - view and save distributions and boxplots with 
    normality assessments. '''
run_eda(observed_data)

''' Create dataframe storing all records that are deemed outliers within their
    corresponding phase*trial_type data subset. Export to csv. Also print out 
    statements describing number of outliers per condition. '''
df_outliers = run_outlier_detection(observed_data)

''' The detected outlier seems to be in agreement with the boxplots from run_eda
    except for identifying an additional outlier each for global efficiecy 
    encoding targets, and retrieval lures. Group data by subjects, as well
    as other categorical columns to ensure that no one person or condition
    can account for the majority of the outliers. '''
outlier_pattern_detection(df_outliers)

''' Results: No subjects are overrepresented as outliers. Slightly more SR, 
    Accurate, global efficiency, and encoding conditions are outliers. No
    outlier removal will be done. Since no subject is 
    overrepresented, it is unlikely that outliers are not true values. '''
    
''' Next model the interaction of Race and Accuracy in the observed data, for
    each of the 4 (phase*trial_type) condtions of interest, and each dependent
    variable. (16 models in total.) Because the efficiency distributions often
    deviate from normality, we will use generalized linear models paired with
    generalized estimator equations. The GLM approach does not make  
    distributional assumptions of nromality, and the GEE technique generalizes 
    GLMs for repeated measures designs.
    For each model, save the results, and the test statistic for the race*accuracy
    interaction to a nested dictionary, 'glm_results_all_conditions'. The test 
    statistic, z, for each factor is calculated by dividing the beta estimate  
    for that population parameter by the robust standard error. The test  
    statisic is a score that indicates how far the data falls from the null 
    hypothesis/ what would be expected if the interaction had no effect on the 
    dependent variables. '''
glm_results_all_conditions = run_gee_analysis(observed_data)
    # Note full GLM results can be accessed like:
    # glm_results_all_conditions['enc_lure']['global_efficiency']['regression_results'].summary()
    # Z-stats can be accessed like:
    # glm_results_all_conditions['retr_target']['avg_local_efficiency']['zstat_interaction']
    
''''Even though GEE can handle correlated and non‐normally distributed 
    data, we still should assess how well the linear model we used can explain 
    the data, since we've made the assumption that brain network topology 
    be modeled as a linear combination of race and accuracy and race * accuracy 
    factors. If this linear model does well to account for the variation in  
    data, the residuals of the regressions should be normally distributed, 
    and the QQ plots of residuals should visually not deviate too much from 
    the 45 degree diagogonal. 
    Print out residual normality assessments, and display/save QQ plots and log
    report to 'output/model_fit directory''.'''
assess_model_fit(glm_results_all_conditions)    
    
''' Permutation Testing to Evaluate Significance of Observed Results:
    Because each cluster of data within our models is small (22 observations), 
    it is possible that effects deemed significant by our glm, could be due to 
    chance. This is compounded by the fact that one of our model's factors is 
    accuracy, and we know from our behavioral analysis that performance was low.
    Despite weak performance, the observed neuro-topology results for interaction 
    produced several z scores above 2 SD from the distribution expected if the 
    null hypothesis (that the race/accuracy interaction has no effect on the 
    depenedent variable) were true. To be more confident that any interaction
    effect observed is significant, instead of using p;-values calculated in 
    the regression models, we will calculate p-values using permutation tests
    on all 8 conditions.
    Specifically, 10,000 iterations of randomized data were generated for each 
    condition. Here we model each condition 10,000 times, and store each
    resulting Z-score. '''
permutation_test_results = run_permutation_analysis(permuted_data_dictionary)
# Alternatively load in permutations if you've alread run this fucntion and 
#     data was saved to csv
permutation_test_results = fetch_z_distributions()   


''' Confirm each condition has 10,000 intearction z-scores corresponging to the
    10,000 regressions. '''
#sanity_check_permutation_results(permutation_test_results)  

''' Calculate singnificance assesments for each condition, and outpu to logfiles
    located in '/output_permutation_analysis/significance_evaluation'''
#p_values_uncorrected = run_evaluate_significance(glm_results_all_conditions, permutation_test_results)

''' Get adjusted p_values, corrected for multiple tests. Use Bonferroni 
    Correction which is quite conservative and makes no assumptions about
    the independence of p_values'''
#significance_df = fwer_correction(p_values_uncorrected)

