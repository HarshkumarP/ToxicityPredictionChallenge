'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca

This script use Feature selector that removes all low-variance features.
 
Output:
List of features after elimination of low-variance features.


For tuning VarianceThreshold refer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
'''

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

# Specify threshold for eliminating features
THRESHOLD_N=0.50
TRAIN_DATA_FILE_PATH = './train_mapped.csv'

all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)

# Convert string values of ID using lable encoder
le = preprocessing.LabelEncoder()
train_id =  list(all_train_data.Id.unique())
le.fit(train_id)
all_train_data['Id']=le.transform(all_train_data['Id'])

# Splitting input and output
data_train = [all_train_data["Expected"]]
headers_train = ["Expected"]
split_train = pd.concat(data_train, axis=1, keys=headers_train)

all_train_data.drop(['Expected'], axis = 1, inplace=True) 
X = pd.get_dummies(all_train_data)
y = split_train["Expected"]

# Setting up variance threshold
sel = VarianceThreshold(threshold=(THRESHOLD_N* (1 - THRESHOLD_N) ))
X_new = sel.fit_transform(X)

# Retrieve list of features from the index of features selected by variance threshold.
list_x = []
col_names = []
selected_features = []

list_x = sel.get_support(indices=True)
col_names = X.columns.tolist()

for i in list_x: 
    selected_features.append(col_names[i])	
	
print("Selected features:")
print(selected_features)
print("Done...")
