"""grid_search.py
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca

Following Script perform the GridSearchCV on the basis of F1_score to provide best params.

model_parameter.json imported for reading model configuration.
"""

import pandas as pd
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb


TRAIN_DATA_FILE_PATH = './train_mapped.csv'
MODEL_PARAMETERS = 'model_parameters.json'

# Reading Train and Test Data files
all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)

le = preprocessing.LabelEncoder()
le.fit(all_train_data['Id'])
all_train_data['Id']=le.transform(all_train_data['Id'])

data = [all_train_data["Expected"]]
headers = ["Expected"]
split_expected = pd.concat(data, axis=1, keys=headers)
all_train_data.drop(['Expected'], axis = 1, inplace=True)


features = ["Id","Assay","V3","V4","V5","V8","V9","V10","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V56","V62","V78","V94","V95","V102","V104","V106","V107","V110","V114","V117","V118","V123","V128","V131","V132","V136","V137","V142","V143","V144","V154","V156","V157","V159","V165","V169","V177","V183","V214","V229","V376","V377","V379","V381","V530","V535","V536","V538","V563","V569","V572","V601","V615","V626","V635","V636","V647","V666","V697","V734","V740","V763","V767","V774","V785","V841","V851","V854","V862","V867","V880","V886","V887","V891","V892","V900","V905","V907","V914","V916","V971","V972","V1013","V1028"]


# Get test and train data on the basis of the feature. Also, and get target data.
X = pd.get_dummies(all_train_data[features])
y = split_expected["Expected"]

# Read .Json file containing models configuration.
with open(MODEL_PARAMETERS) as json_file:
        all_model_data = json.load(json_file)


result = []


# Iterating over specified models with set of parameters to get best scores and best parameters.
for model, model_data in all_model_data.items():
    search = GridSearchCV(eval(model_data['model']), model_data['parameters'], cv=5, scoring=make_scorer(f1_score, average='macro'))
    search.fit(X,y)
    result.append({
        'model_name': model,
        'best_score': search.best_score_,
        'best_params': search.best_params_
       })
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    params = search.cv_results_['params']
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(result)
print(df)