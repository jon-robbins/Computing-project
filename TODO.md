# oldfartsfinalproject library TODO
This library is a work in progress. We welcome all contributors! If you want to contribute, make a pull request into the sandbox branch, and admins will approve merges to main. 
## data_preprocessing.py
* impute_na
	* Verbose mode: output how many na's exist in each column
	* error handling: give user option to skip column if error found in only one column
* delete_unnec_cols
	* cols: take dictionary as argument
		* key: col name
		* value: error threshold 
## feature_creation.py
* create_dummies
* upper_outlier_dummy
	* Give more options for IQR threshold (e.g. using 2 instead of 1.5 as a user input
* lower_outlier_dummy
	* Give more options for IQR threshold (e.g. using 2 instead of 1.5 as a user input
* New feature: lat/long cleaning. Allow proper formatting of latitude and longitude columns
* New feature: Get city info
	* Currently have draft code for connecting to a third party API to get city/state/county information about each house. This can be used to correlate each house with crime rates. Anyone who can make this work will get a beer on us. 
