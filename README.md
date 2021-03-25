# permutation_analysis_brain_connectivity
Permutation analysis of connectivity matrices to assess statistical significance of observed effects

permutation_analysis_graph_metrics_v16.py takes in graph topological metrics calculated in the Brain Connectivity Toolbox for multiple experimental conditions in a neuroimaging task. 
Because our sample was small, I decided to run permutation analyses instead of just ANOVAs to assess the significance of a few metrics. 
The script randomizes data to create F-distributions to test whether our observed results might be expected by chance. 
Script outputs histograms marking the 95% percentile and location of the observed F-stat, as well as a log file with significance assessments based on the results of the permutation test. 

testingRequirments.py includes some tests i ran when I was formulating how to best randmonize the graph metric data. 
