'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca
This script runs Ensemble voting classifier

Output:
CSV file with predictions

For tuning votingclassifier refer: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

TRAIN_DATA_FILE_PATH = './train_mapped.csv'
TEST_DATA_FILE_PATH = './test_mapped.csv'
RAW_TEST_DATA = './test.csv'

# Reading Train and Test Data files
all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)
all_test_data = pd.read_csv(TEST_DATA_FILE_PATH)
raw_test_data = pd.read_csv(RAW_TEST_DATA)

# Encode chemical IDS
train_id =  list(all_train_data.Id.unique())
test_id = list(all_test_data.Id.unique())
le = preprocessing.LabelEncoder()
le.fit(train_id + test_id)
all_train_data['Id']=le.transform(all_train_data['Id'])
all_test_data['Id']=le.transform(all_test_data['Id'])

# Splitting input and output
data_train = [all_train_data["Expected"]]
headers_train = ["Expected"]
split_train = pd.concat(data_train, axis=1, keys=headers_train)
all_train_data.drop(['Expected'], axis = 1, inplace=True)

features = ["Id","Assay","V3","V4","V5","V8","V9","V10","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V34", "V35","V36","V40","V43","V56","V58","V62","V78","V94","V95","V96","V102","V104","V106","V107","V110","V114","V117","V118","V123","V128","V129","V131","V132","V136","V137","V142","V143","V144","V154","V156","V157","V159","V165","V169","V173","V177","V183","V186","V214","V229","V232","V290","V349","V374","V376","V377","V379","V381","V409","V413","V530","V535","V536","V538","V563","V569","V572","V573","V575","V576","V591","V601","V615","V626","V627","V635","V636","V647","V666","V667","V697","V711","V734","V740","V763","V767","V774","V785","V841","V851","V854","V862","V867","V880","V886","V887","V891","V892","V900","V905","V907","V914","V916","V971","V972","V1013","V1028","V746","V848","V888","V1038","V1062"]

# Get test and train data on the basis of the feature. Also, and get target data.
X = pd.get_dummies(all_train_data[features])
X_test = pd.get_dummies(all_test_data[features])
y = split_train["Expected"]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#defining base estimators
clf1 = xgb.XGBClassifier(random_state=1,booster="gbtree",learning_rate=0.25,n_estimators=250,max_depth=12, min_child_weight=4)
clf2 = lgb.LGBMClassifier(boosting_type= 'goss',learning_rate=0.1,n_estimators=1000,max_depth=10,num_leaves=100,max_bin = 5000)
estimators = [('XGB',clf1),('LGBM',clf2)]

#loop iteration to tune weight
# for i in range(1,3):
#      for j in range(1,3):

vc = VotingClassifier(estimators=estimators,voting='hard',weights=[1,1])
f1 = cross_val_score(vc,X,y,cv=5,scoring='f1_macro')
acc = cross_val_score(vc,X,y,cv=5)
print("F1 Macro score for weights (1,1): ",np.round(np.mean(f1),5))
print("Accuracy score for weights (1,1): ",np.round(np.mean(acc),5))

vc = vc.fit(X,y)
val_predictions = vc.predict(X_test)
y_pred = vc.predict(val_X)

output = pd.DataFrame({'Id': raw_test_data.Id, 'Predicted': val_predictions})
output.to_csv('ensemble.csv', index=False)