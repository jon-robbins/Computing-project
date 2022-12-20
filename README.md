# Computing project


 This project aims to create a Python package containing modular functions which can be 
 imported and used for solving a ML problem.
 
 ## Modules
 
~ data_util.py
 
    This module retrieves an sepcified dataset from Mysql database.
 
 
     * Class DataUtil:
 
         - Class Attributes: host, user, password, port, databasename, charset
 
     * Methods: 
         
         - connect: creates the connection engine for connecting to the databasae
         
         - close: closes the connection with the data base
         
         - datafrom: reads from sql database
         
         - datafrompath: reads csv file from specified path
 
 
~ data_preprocessing.py
 
    This module contains data pre-processing functions.

       * Methods:
           
         - impute_na: imputes null values of a column with specified value
         
         - delete_unnec_cols: deletes columns which have more that a specifies percentage  of null values
               
         - to_num: converts the datatype of objects to numeric
         
         
~ feature_creation.py

    This module contains functions for feature creation.
        
        * Methods:
        
          - create_dummies: creates dummies for specified columns
          
          - upper_outlier_dummy: creates dummy value 1 against values which were identified as upper outliers, else 0
          
          - lower_outlier_dumyy: creates dummy value 1 against values which were identified as lower outliers, else 0
          
          
~ test_train_split.py
        
    This module splits the data set into test dataset and train dataset.
    
        * Methods:
        
            - test_train_split: creates the fetaures and target dataframe and further perform split to create test and train dataset in specefied ratio
            

~ model_output.py

    This module contains different regression models, one of which could be selected to apply on the dataset to make predictions. 
    
        * Methods:
        
            - print_r2_score: prints the r-square statistic for the model
            
            - regression_output: return the prediction from the model


~  model_evaluation.py

    This model provides model evaluation metrics.
    
        * Methods:
        
            - mean_absolute_error: retruns the value of MAE
            
            - mean_squared_error: returns the values of MSE
            
            - rsqures: returns the value of rsquared
          
          
         
         
         
               
 
 
 
 