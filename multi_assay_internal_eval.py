'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca

This script train individual model against each assay_id and returns cv score for each model.

Output:
Mean cross validation, and individual assay_id cross validation score. 
'''

import pandas as pd
import os

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from statistics import mean 

DROP_COLUMNS = ['Expected','Assay']
FEATURES=['Id', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V34', 'V40', 'V58', 'V95', 'V206', 'V208', 'V224', 'V252', 'V290', 'V327', 'V350', 'V419', 'V452', 'V663', 'V673', 'V674', 'V676', 'V918', 'V1073']
TARGET="Expected"
DIRECTORY = './assay_split/train'
MODEL='RandomForestClassifier(random_state=1)'

all_features = False

le = preprocessing.LabelEncoder()

assay_id_list = []
accuracy_score_list = []
f1_score_list = []

for filename in os.listdir(DIRECTORY):
    if filename.endswith(".csv"):
    
        assay_id = pd.read_csv(DIRECTORY+"/"+filename)
        le.fit(assay_id.Id.unique())
        
        # Encode chemical Id
        assay_id['Id']=le.transform(assay_id['Id'])
        
        # Get TARGET values from dataset before dropping column
        data_traget = [assay_id[TARGET]]
        headers_traget = [TARGET]
        split_traget = pd.concat(data_traget, axis=1, keys=headers_traget)
        
        # Drop columns 
        assay_id.drop(DROP_COLUMNS, axis = 1, inplace=True) 
        
		# Check whether to consider all features or the selected ones.
        if all_features:
            X = pd.get_dummies(assay_id)
        else:
            X = pd.get_dummies(assay_id[FEATURES])
            
        y = split_traget[TARGET]
        
        # Classification model
        model = eval(MODEL)

        # Internal evaluation using cross validation
        accuracy_scores = cross_val_score(model, X, y, cv=5)
        f1_macro_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        
        print("SCORES FOR ASSAY ID",filename)
        print("%0.4f accuracy with a standard deviation of %0.4f" % (accuracy_scores.mean(), accuracy_scores.std()))
        print("%0.4f f1_score(macro) with a standard deviation of %0.4f" % (f1_macro_scores.mean(), f1_macro_scores.std()))
        
		# Generate table for internal scores for each assay
        assay_id_list.append(filename)
        accuracy_score_list.append(accuracy_scores.mean())
        f1_score_list.append(f1_macro_scores.mean())

    else:
        continue


print(mean(accuracy_score_list))
print(mean(f1_score_list))
pd.set_option('display.max_rows', None)
dict = {'assay_id': assay_id_list, "accuracy_score" : accuracy_score_list, "f1_score":f1_score_list}
df = pd.DataFrame(dict)

print(df)
