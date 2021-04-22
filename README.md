# permutation_analysis_brain_connectivity
Permutation analysis of connectivity matrices to assess statistical significance of observed effects

<b>regression_prep_generate_permutations.py</b> transforms data into a structure compatible for regression analysis, and generate random permutations to run permutation tests with. Specifically, the script reads in whole-brain raw topological metric data associated with experimental fMRI conditions. These metrics were previously calculated using the Brain Connectivity toolbox. It outputs the observed data, structured for modeling in the next script. It also outputs tables with 10,000 permutations per subject of randomized data, for use in subsequent permutation tests to evaluate statistical significance. 

<b>permutation_analysis.py</b> runs regression analysis, permutation tests, and FWER correction on the datasets structured in regression_prep_generate_permutations. Specifically, the script models each condition with a generalized linear model using generalized estimator equations. It then reruns this model on the permuted data to create distributions of test statistics of interest, from which p values are calculated. Lastly it runs a FWER Bonferroni correction on the resulting p values to reduce the risk of 
making false discoveries. Evaluation of the significance of each test is then exported to csv. The script additionally includes functions for exploring distributions and assessing model fit. 

Data used for these analyses are protected under IRB and cannot be shared.
