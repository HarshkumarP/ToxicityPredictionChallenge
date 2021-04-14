# ToxicityPredictionChallenge
##### With new chemicals being synthesized every day, toxicity prediction of newly synthesized chemicals is mandatory before they could be released in the market. For a long time, *in-vivo* methods have been used for toxicity prediction which involves studying bacteria, human cells, or animals. With thousands of new chemicals being synthesized every day, it is not feasible to detect toxicity with traditional laboratory animal testing. One great alternative for *in-vivo *methods is the *in-silico* techniques that have great potential to reduce time, cost, and animal testing involved in detecting toxicity. ToxCast dataset is one of the greatest data available in the field of toxicogenomics. ToxCast has data for approximately 9,000 chemicals with more than 1500 high-throughput assay endpoints that cover a range of high-level cell responses. In this challenge, you will have access to a prepared subset of data from ToxCast.


# System setup and Requirement
Programming-Language: **Python**
	
Python package requirements:
###### (Following packages can be installed using "pip", refer python official documentation for installing packages)
	
- pandas
- numpy
- os

# 1. DATA PREPARATION

##### Directory input contains Test and train set with unmapped features data, and separate feamat.csv with features.
Mapping of features from feamat.csv to test and train dataset carried out by following script.
		
RUN "data_preparation.py" File in order to generate Test and Train dataset.
	
```
python data_preparation.py 
```
