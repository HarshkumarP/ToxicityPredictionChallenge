'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca

This script combines the individual prediction of each assay_id model -
as per the final test.csv format for submission,

Output:
.csv: prediction of each assay_id model combined in single csv file.
'''

import pandas as pd
import os

RAW_TEST_DATA = './input/test.csv'
OUTPUT_DIRECTORY = './assay_split/output/'

raw_test_data = pd.read_csv(RAW_TEST_DATA)

df = pd.DataFrame()


for filename in os.listdir(OUTPUT_DIRECTORY):
    if filename.endswith(".csv"):
    
        df = df.append(pd.read_csv(OUTPUT_DIRECTORY+"/"+filename), ignore_index=True)
       
    else:
        continue
        
        
print(df)


output_map = raw_test_data.merge(df, left_on='Id', right_on='Id', how='left')
print(output_map)

output_map.to_csv('my_submission_multi_assay_rf.csv', index=False)