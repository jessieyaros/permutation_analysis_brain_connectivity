# permutation_analysis_brain_connectivity
Permutation analysis of connectivity matrices to assess statistical significance of observed effects

This script takes in graph topological metrics calculated in the Brain Connectivity Toolbox for multiple experimental conditions in a neuroimaging task. 
Because our sample was small, I decided to run permutation analyses instead of just ANOVAs to assess the significance of a few metrics. 
The script randomizes data to create F-distributions to test whether our observed results might be expected by chance. 
Script outputs histograms marking the 95% percentile and location of the observed F-stat, as well as a log file with significance assessments based on the results of the permutation test. 
