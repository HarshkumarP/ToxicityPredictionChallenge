'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca

This script gets the features from feamat.csv and generate the train and test dataset on the Basis of Chemical ID.

'ID' Column of train and test used to Join column 'V1' from the feamat data.
Removes Zerovariance features (All 0's) 

Output:
train_mapped.csv: Train dataset
test_mapped.csv: Test dataset
train_feature_analysis.csv: General statistical information of train dataset
test_feature_analysis.csv: General statistical information of test dataset
'''

import pandas as pd
import numpy as np

# Provide appropriate path for location of csv's in your own system.
TRAIN_DATA_FILE_PATH = './input/train.csv'
TEST_DATA_FILE_PATH = './input/test.csv'
FEATURE_MATRIX = './input/feamat.csv'

# Reading Train, Test and Feature Data files
all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)
all_test_data = pd.read_csv(TEST_DATA_FILE_PATH)
all_features = pd.read_csv(FEATURE_MATRIX)

# Remove all features having all value as 0
all_features = all_features.loc[:, (all_features != 0).any(axis=0)]

# Split Id and Assay
all_train_data[['Id','Assay']] = all_train_data.Id.str.split(";",expand=True,)
all_test_data[['Id','Assay']] = all_test_data.Id.str.split(";",expand=True,)

# Get features from feamat.csv into test and train data
train_feature_map = all_train_data.merge(all_features, left_on='Id', right_on='V1', how='left')
test_feature_map = all_test_data.merge(all_features, left_on='Id', right_on='V1', how='left')

# Transform previously divided Assay column into int(only necessary if you would like to use assay as a feature)
train_feature_map['Assay'] = train_feature_map.Assay.astype(int)
test_feature_map['Assay'] = test_feature_map.Assay.astype(int)

# Drop Extra ID column generated while merge, and V2 (PubChem ID)
train_feature_map.drop(columns=['V1', 'V2'],inplace=True)
test_feature_map.drop(columns=['V1', 'V2'],inplace=True)

# Replace Infinity value with NaN -> <Mean>
train_feature_map['V15'].replace([np.inf, -np.inf], np.nan, inplace=True)
test_feature_map['V15'].replace([np.inf, -np.inf], np.nan, inplace=True)
train_feature_map['V15'].fillna((train_feature_map['V15'].mean()), inplace=True)
test_feature_map['V15'].fillna((test_feature_map['V15'].mean()), inplace=True)

# Get the Column name having all value as '0'
columns_list = list(train_feature_map.columns) 

unwanted_feature_list = []
for col in columns_list:
	if train_feature_map[col].sum() == 0:
		unwanted_feature_list.append(col)

# Drop of column from testset and trainset having '0' in whole column
train_feature_map.drop(columns=unwanted_feature_list, inplace=True)
test_feature_map.drop(columns=unwanted_feature_list, inplace=True)

# Store mappings in CSV
train_feature_map.to_csv('train_mapped.csv', index=False)
test_feature_map.to_csv('test_mapped.csv', index=False)

# Storing General describe data for analysis of dataset
train_feature_map.describe().to_csv('train_feature_analysis.csv', index=False)
test_feature_map.describe().to_csv('test_feature_analysis.csv', index=False)

print("Done...")