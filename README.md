# ToxicityPredictionChallenge
##### With new chemicals being synthesized every day, toxicity prediction of newly synthesized chemicals is mandatory before they could be released in the market. For a long time, *in-vivo* methods have been used for toxicity prediction which involves studying bacteria, human cells, or animals. With thousands of new chemicals being synthesized every day, it is not feasible to detect toxicity with traditional laboratory animal testing. One great alternative for *in-vivo *methods is the *in-silico* techniques that have great potential to reduce time, cost, and animal testing involved in detecting toxicity. ToxCast dataset is one of the greatest data available in the field of toxicogenomics. ToxCast has data for approximately 9,000 chemicals with more than 1500 high-throughput assay endpoints that cover a range of high-level cell responses. In this challenge, you will have access to a prepared subset of data from ToxCast.


# System setup and Requirement
Programming-Language: **Python**
	
###### Python package requirements:
> ###### (Following packages can be installed using "pip", refer python official documentation for installing packages)
	
pandas   
numpy   
sklearn   
mlxtend   
statistics   

# Data Preparation

##### Directory input contains Test and train set with unmapped features data, and separate feamat.csv with features.
Mapping of features from feamat.csv to test and train dataset carried out by following script.
		
RUN "data_preparation.py" File in order to generate Test and Train dataset.
	
```
python data_preparation.py 
```

### Data prep for multiple assay_id  model
	
RUN "multi_assay_data_prep.py" for generating individual test and train set for each assay id.
```
python multi_assay_data_prep.py
```
> Above script will generate "./assay_split/test/" and "./assay_split/test/" Folder with test and training sets.    
> NOTE:- Before executing split_by_assay.py make sure data_preparation.py executed successfully. 

# Feature Selection
> ###### prerequisite: Data Preparation

RUN "feature_selection_rfe.py" for Selecting features using Recursive feature elimination.    
RUN "feature_selection_sfs.py" for Selecting features using Sequential feature selector.    
RUN "feature_selection_anova.py" for Selecting features using Analysis of Variance.    
RUN "feature_selection_variance.py" for Selecting features using Variance Threshold.    

e.g.,
```
python feature_selection_rfe.py
```


# Building Classifier (Single model for entire dataset)    
> ###### prerequisite: Data Preparation


# Building Classifier (Multiple model for each assay_id)    
> ###### prerequisite: Data Preparation

Process of Training, Testing and predicting individual assay_id model works in following manner:    
1. Split data set into each assay_id csv file. (RUN "multi_assay_data_prep.py", refer "Data Preparation" Part of README)    
2. Train model and predict result for each assay_id. (RUN "multi_assay_internal_eval.py")
> GUID TO RUN: There are two option present for running "multi_assay_classifier.py" A. all_features B. predict_output
> SET 'all_features' True when training against all the features, if set False, then provide list of features in "FEATURES".
> SET 'predict_output' Set is as False while performing internal evaluations, as it will only perform cross-validation instead of predicting and storing result in csv's.    
3. Merge output as per the format of test.csv, using prediction stored in csv's in step 2. (RUN "multi_assay_merge_output.py")


  
 