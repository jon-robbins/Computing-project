# Computing project


 This project aims to create a Python package conating modular functions which can be 
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
 
    This module conatins data pre-processing functions.

       * Methods:
           
         - impute_na: imputes null values of a column with specified value
         
         - delete_unnec_cols: deletes columns which have more that a specifies percentage  of null values
               
         - to_num: converts the datatype of objects to numeric
         
         
~ 
         
         
         
               
 
 
 
 