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
import os

# Reading Train and Test Data files, provide train dataset, assuming train data set has "Assay" column in Integer.
TRAIN_DATA_FILE_PATH = './train_mapped.csv'
TEST_DATA_FILE_PATH = './test_mapped.csv'

all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)
all_test_data = pd.read_csv(TEST_DATA_FILE_PATH)


if not os.path.exists('./assay_split'):
    os.mkdir('assay_split')
if not os.path.exists('./assay_split/train'):
    os.mkdir('./assay_split/train')
if not os.path.exists('./assay_split/test'):
	os.mkdir('./assay_split/test')

# Divide all assay from train set to different csv's
train_assay_group = all_train_data.groupby('Assay')
train_assay_ids = train_assay_group.groups.keys()

for assay_id in train_assay_ids:
    assaydf = train_assay_group.get_group(assay_id) 
    assaydf.to_csv('./assay_split/train/'+str(assay_id)+".csv", index=False)


# Divide all assay from test set to different csv's
test_assay_group = all_test_data.groupby('Assay')
test_assay_ids = test_assay_group.groups.keys()

for assay_id in test_assay_ids:
    assaydf = test_assay_group.get_group(assay_id) 
    assaydf.to_csv('./assay_split/test/'+str(assay_id)+".csv", index=False)
	
	
print("Done...")