## Ans3 - folder that contains results of Exersise 3: ipynb file, readme file, requirement.txt, csv file, sav file and 3 python files.
* Analysis.ipunb - jupyter notebook with analysis
* Preprocessing_data.py - class for uploading data, splitting and scaling
* Model_learning.py - class that has four functions:
    * function for validation certain model,
    * function for choosing best model by RMSE,
    * function that learns chosen model on all train data and save model
    * function that make prediction for new data
* main.py - main file where I used functionality of those two class
* prediction_results.csv - contains results of predictions
* finalized_model - contains saved model