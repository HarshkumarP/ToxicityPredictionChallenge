'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca

This script train and predict results considering whole database,
and write prediction into csv file.

NOTE:- While having complex model like bagging, code might run for several minutes.

Output:
.csv: prediction of each assay_id model combined in single csv file.
'''

import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier


TRAIN_DATA_FILE_PATH = './train_mapped.csv'
TEST_DATA_FILE_PATH = './test_mapped.csv'
RAW_TEST_DATA = './input/test.csv'

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


features=["Id","Assay","V3","V4","V5","V8","V9","V10","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V34", "V35","V36","V40","V43","V56","V58","V62","V78","V94","V95","V96","V102","V104","V106","V107","V110","V114","V117","V118","V123","V128","V129","V131","V132","V136","V137","V142","V143","V144","V154","V156","V157","V159","V165","V169","V173","V177","V183","V186","V214","V229","V232","V290","V349","V374","V376","V377","V379","V381","V409","V413","V530","V535","V536","V538","V563","V569","V572","V573","V575","V576","V591","V601","V615","V626","V627","V635","V636","V647","V666","V667","V697","V711","V734","V740","V763","V767","V774","V785","V841","V851","V854","V862","V867","V880","V886","V887","V891","V892","V900","V905","V907","V914","V916","V971","V972","V1013","V1028","V746","V848","V888","V1038","V1062"]

# Get test and train data on the basis of the feature. Also, and get target data.
X = pd.get_dummies(all_train_data[features])
X_test = pd.get_dummies(all_test_data[features])
y = split_train["Expected"]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Building model -> Predicting results -> Storing prediction output into .CSV
model = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(random_state=1,
                                                                     criterion='entropy',
                                                                     max_depth=35,
                                                                     class_weight='balanced'),
                                                                     random_state=1,
                                                                     n_estimators=20)

model = model.fit(X, y)
val_predictions = model.predict(X_test)

accuracy_scores = cross_val_score(model, X, y, cv=5)
f1_macro_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')

print("%0.4f accuracy with a standard deviation of %0.4f" % (accuracy_scores.mean(), accuracy_scores.std()))
print("%0.4f f1_score(macro) with a standard deviation of %0.4f" % (f1_macro_scores.mean(), f1_macro_scores.std()))

output = pd.DataFrame({'Id': raw_test_data.Id, 'Predicted': val_predictions})
output.to_csv('submission_single_model_bc.csv', index=False)

