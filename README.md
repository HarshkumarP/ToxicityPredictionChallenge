# ToxicityPredictionChallenge
##### With new chemicals being synthesized every day, toxicity prediction of newly synthesized chemicals is mandatory before they could be released in the market. For a long time, *in-vivo* methods have been used for toxicity prediction which involves studying bacteria, human cells, or animals. With thousands of new chemicals being synthesized every day, it is not feasible to detect toxicity with traditional laboratory animal testing. One great alternative for *in-vivo *methods is the *in-silico* techniques that have great potential to reduce time, cost, and animal testing involved in detecting toxicity. ToxCast dataset is one of the greatest data available in the field of toxicogenomics. ToxCast has data for approximately 9,000 chemicals with more than 1500 high-throughput assay endpoints that cover a range of high-level cell responses. In this challenge, you will have access to a prepared subset of data from ToxCast.


# System setup and Requirement
Programming-Language: **Python**
	
###### Python package requirements:
> ###### (Following packages can be installed using "pip", for e.g., "pip install pandas")
	
pandas   
numpy   
scikit-learn   
mlxtend   
statistics   
xgboost   
lightgbm   
   

# Data Preparation

##### Directory input contains Test and train set with unmapped features data, and separate feamat.csv with features.
Mapping of features from feamat.csv to test and train dataset carried out by following script.   
> Move to directory containing "data_preparation.py"
		
RUN "data_preparation.py" File in order to generate Test and Train dataset.
	
```
python data_preparation.py 
```
> Above mentioned script will create csv's 1. test_mapped.csv, 2. train_mapped.csv used for testing and training model.   

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

RUN "classifier.py" for Building BaggingClassifier as a single model for whole dataset.
```
python classifier.py
```
> Modify "features" List in classifier.py for selecting features for training a model.   
> For other types of classifier, update line no 56. with the required classifier, for example;    
> model = tree.DecisionTreeClassifier(random_state=1,criterion='entropy',max_depth=30)   
> model = RandomForestClassifier(random_state=1)   
> model = svm.SVC()   
> model = LogisticRegression(solver='liblinear')   
> model = GradientBoostingClassifier(n_estimators=150, learning_rate=1.0,max_depth=1, random_state=1)   
> model = MLPClassifier()   
> model = GaussianNB()   
> model = AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=1,criterion='entropy',max_depth=30),algorithm="SAMME",n_estimators=200,random_state=1)   
> model = GradientBoostingClassifier(random_state=1, learning_rate=1.0, loss='exponential', max_depth=10, n_estimators=140)   
> Also, import respecting classifier from the sklearn package.    

## Ensemble model(RUN Following code to reproduce our Best performing model)


##### NOTE:- We failed to reproduce our best score.   
##### Reason:- Different machine(kernel, processor) produce different result using LGBM, as LGBM optimized for optimized with multi-threading. which we realized after training model on different environment after submission on kaggle.    
#### However, we have fixed issue which prodcue fixed result, please check expected output in below table which should be the expected result.
Submission | Internal_score | Private_score | Public_score
---------- | -------------- | ------------ | -------------
Submission on Kaggle | 0.80424 | 0.80775 | 0.81694 
Expected output of following code | 0.80354 | 0.80675 | 0.81649


RUN "ensemble_votingclassifier.py" to ensemble xgb and lgbm. 
```
python ensemble_votingclassifier.py
```  

> NOTE:- Currently weights for voting classifier are hardcoded as [1,1] which was identified by iterating with range(1,3)   
> Modify code as per following to determine best weight distribution before building classifier.    
``` 
  # loop iteration to tune weight
   for i in range(1,3):
        for j in range(1,3):
  		vc = VotingClassifier(estimators=estimators,voting='hard',weights=[i,j])
  		f1 = cross_val_score(vc,X,y,cv=5,scoring='f1_macro')
  		acc = cross_val_score(vc,X,y,cv=5)
```


# Building Classifier (Multiple model for each assay_id)    
> ###### prerequisite: Data Preparation

Process of Training, Testing and predicting individual assay_id model works in following manner:    
1. Split data set into each assay_id csv file. (RUN "multi_assay_data_prep.py", refer "Data Preparation" Part of README)     
2. Train model and predict result for each assay_id. (RUN "multi_assay_internal_eval.py")
> GUID TO RUN: There are two option present for running "multi_assay_classifier.py" A. all_features B. predict_output    
> SET 'all_features' True when training against all the features, if set False, then provide list of features in "FEATURES".    
> SET 'predict_output' Set is as False while performing internal evaluations, as it will only perform cross-validation instead of predicting and storing result in csv's.    
```
python multi_assay_classifier.py
```  
3. Merge output as per the format of test.csv, using prediction stored in csv's in step 2. (RUN "multi_assay_merge_output.py")    
```
python multi_assay_merge_output.py
``` 


# Tunning using GridSearchCV    
> ###### prerequisite: Data Preparation

For performing GridSearchCV, pass parameters into file "model_parameters.json"    
Sample example:    
```
{
	"decision_tree" : {
		"model" : "DecisionTreeClassifier()",
		"parameters" : {
			"criterion" : ["gini","entropy"],
			"max_depth" : [4,5,6,7]
		}
	},
	"random_forest" : {
		"model" : "RandomForestClassifier()",
	    "parameters" : {
			"n_estimators": [10,20,25,],
			"max_depth" : [4,5,6]
		}
	},
	"logistic_regression" : {
		"model" : "LogisticRegression(solver='liblinear')",
	    "parameters" : {
			"C": [1,5,10],
			"random_state": [0,1,3,8],
			"warm_start": ["True","False"]
		}
	},
	"svm" : {
		"model" : "svm.SVC()",
	    "parameters" : {
			"C": [25,30,35]
		}
	},
	"Kn_neighbors_classifier" : {
		"model" : "KNeighborsClassifier()",
	    "parameters" : {
			"n_neighbors" : [4,5],
			"algorithm": ["ball_tree"],
			"leaf_size": [25,30,35]
		}
	}
}
```

> Using above sample params GridSearchCV will return best scores with params for all models.    
> In above example "model" contains classifier method, and "parameters" contains range of parameters for tuning classifier.    
> NOTE:- Make sure to import classifier into "grid_search_cv.py". 

After setting parameters RUN "grid_search_cv.py".   
```
python grid_search_cv.py
```
