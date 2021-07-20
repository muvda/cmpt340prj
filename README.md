# CMPT 340 Code Sample
This is the part of a CMPT 340 Project that I created
## Python Installation
* The code makes use of Python 3.9.4 as such it is required to run the code. Additional libraries can be installed with requirements.txt
* ```
  pip install -r requirements.txt
  ```
## Run Instruction
* Enter the following commands to run the code:
* ```
  python3 1-data-cleaning.py
  python3 2-classification.py
  ```
## Expected result
* Result for running the second file can be viewed below:
* ```
  Examining different Feature Selection Methods (this may take approx 3 minutes)
  Begin Recursive feature elimination
  Begin Recursive feature elimination with cross-validation
  Begin L1-based feature selection
  Begin tree-based feature selection
  Result:
  Recursive_feature_elimination train data accuracy : 1.0
  Recursive_feature_elimination test data accuracy : 0.5586592178770949
  Recursive_feature_elimination_cross-validation train data accuracy : 1.0
  Recursive_feature_elimination_cross-validation test data accuracy : 0.6368715083798883
  Number of features for RFECV: 59 
  L1-based_feature_selection train data accuracy : 1.0
  L1-based_feature_selection test data accuracy : 0.5921787709497207
  Tree-based_feature_selection train data accuracy : 1.0
  Tree-based_feature_selection test data accuracy : 0.5921787709497207
  ```
* Figures produced by the code can be view in the folder figures