#!/usr/bin/env python
# coding: utf-8

# Machine Learning Assignment 3: Benjamin Bialuchukwu Bakwenye, Jonathan Robbins, Tobias Pfeiffer
# 

# First we import and clean the test and training data

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Import the data
training = pd.read_csv('Regression_Supervised_Train.csv', sep = ";")
test = pd.read_csv('Regression_Supervised_Test.csv', sep = ";")

# In[6]:


training['parcelvalue'] = pd.to_numeric(training['parcelvalue'], errors='coerce')

training['poolnum'] = training['poolnum'].fillna(0)
test['poolnum'] = test['poolnum'].fillna(0)

training['unitnum'] = training['unitnum'].fillna(1)
test['unitnum'] = test['unitnum'].fillna(0)

# Turn Aircond into a dummy, the 4th column in the data
training['aircond'] = training['aircond'].fillna(0)
test['aircond'] = test['aircond'].fillna(0)
#%%
# Create Variables for the heating type
heating_labels = {'heatingtype': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25}, 'heatingdesc': {0: 'Other', 1: 'Baseboard', 2: 'Central', 3: 'Coal', 4: 'Convection', 5: 'Electric', 6: 'Forced air', 7: 'Floor/Wall', 8: 'Gas', 9: 'Geo Thermal', 10: 'Gravity', 11: 'Heat Pump', 12: 'Hot Water', 13: 'None', 14: 'Other', 15: 'Oil', 16: 'Partial', 17: 'Propane', 18: 'Radiant', 19: 'Steam', 20: 'Solar', 21: 'Space/Suspended', 22: 'Vent', 23: 'Wood Burning', 24: 'Yes', 25: 'Zone'}}
heating_labels = pd.DataFrame.from_dict(heating_labels)
training['heatingtype'] = training['heatingtype'].where(~training['heatingtype'].isin([1,2,3]),0)

training = pd.merge(training, heating_labels, on='heatingtype', how='left')

heating_dummies = pd.get_dummies(training['heatingtype'])
                                 
training['hd1'] = heating_dummies.iloc[:, 0]

training['hd2'] = heating_dummies.iloc[:, 1]
training['hd3'] = heating_dummies.iloc[:, 2]
training['hd4'] = heating_dummies.iloc[:, 3]

training.drop(columns = 'heatingdesc', axis = 1)
training


# In[7]:


test['heatingtype'] = test['heatingtype'].where(~test['heatingtype'].isin([1,2,3]),0)
test = pd.merge(test, heating_labels, on='heatingtype', how='left')

heating_dummies2 = pd.get_dummies(test['heatingtype'])

test['hd1'] = heating_dummies2.iloc[:, 0]
test['hd2'] = heating_dummies2.iloc[:, 1]
test['hd3'] = heating_dummies2.iloc[:, 2]
#test['hd4'] = heating_dummies2.iloc[:, 3]

test.drop(columns = 'heatingdesc', axis = 1)
test


# Delete Columns with a lot f missing values and 'totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer'

def delete_unnec_cols(training_df, test_df):
    test_column_check = []
    for col in training.columns:
        if training_df[col].isna().sum() > 0.4*22000 or col in ('totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer'):
            test_column_check.append(col)
            
    # Here we actually delete the columns        
    for col in training_df.columns:
        if training_df[col].isna().sum() > 0.4*22000 or col in ('totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer'):
            del training_df[col]


    # Performing the same transformations on the test data
    for col in test_df.columns:
        if col in test_column_check:
            del test_df[col]
    
    return training_df, test_df


def create_county_dummies(training_df, test_df):
    dummy_train = pd.get_dummies(training_df['countycode'])
    
    training_df['la_county'] = dummy_train.iloc[:, 0]
    training_df['organge_county'] = dummy_train.iloc[:, 1]
    training_df['ventura_county'] = dummy_train.iloc[:, 2]
    
    dummy_test = pd.get_dummies(test_df['countycode'])
    
    test_df['la_county'] = dummy_test.iloc[:, 0]
    test_df['organge_county'] = dummy_test.iloc[:, 1]
    test_df['ventura_county'] = dummy_test.iloc[:, 2]
    
    return training_df, test_df


# Here we plotted data to figure out what would best fit in our model

# In[12]:


import seaborn as sns
sns.pairplot(data = training, y_vars = ["parcelvalue"], x_vars = ['numbath', 'numbedroom', 'finishedarea', 'countycode', 'numfireplace'],  hue = "countycode", palette = "Set2")


# In[13]:


sns.pairplot(data = training, y_vars = ["parcelvalue"], x_vars = ['numfullbath', 'heatingtype', 'lotarea', 'poolnum', 'roomnum'],  hue = "countycode", palette = "Set2")


# In[14]:


sns.pairplot(data = training, y_vars = ["parcelvalue"], x_vars = ['year', 'la_county', 'organge_county', 'ventura_county','aircond'], hue = "countycode", palette = "Set2")


# In[15]:


sns.pairplot(data = training, y_vars = ["parcelvalue"], x_vars = ['unitnum', 'regioncode', 'neighborhoodcode'], hue = "countycode", palette = "Set2")


# Here we start the model

# In[16]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

tr = training.copy()
tr = tr.dropna()
fyv = np.percentile(tr['parcelvalue'], 5)
nty = np.percentile(tr['parcelvalue'], 90)

tr = tr[(tr['parcelvalue'] < 1000000) & (tr['parcelvalue'] > fyv)]

# We check the "candidate" data one last time
# profile2 = ProfileReport(tr, title="Report")
# profile2

# potential is just an array with meaning full regressor and makes trying out models easier (Ctrl+C & Ctrl+V) 
potential = ['latitude', 'longitude', 'numbath', 'ventura_county']
X = tr.drop(columns = ['roomnum', 'ventura_county', 'lotid','heatingtype', 'numbath', 'heatingdesc', 'countycode', 'citycode', 'countycode2', 'taxyear', 'parcelvalue', 'neighborhoodcode', 'regioncode', 'numfireplace', 'unitnum', 'hd1', 'hd2', 'hd3', 'hd4', 'organge_county'], axis = 1)
X
y = tr['parcelvalue']
X


# In[20]:


nty


# Here we split the data set, and set a seed so we can have a more accurate comparison of the different models. 

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 43)


# In[22]:


import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, Y_pred))
print(list(zip(regr.coef_, X.columns)))


# Plot outputs
plt.scatter(y_test, Y_pred, color="black")
#plt.plot(y_test, Y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[23]:


sns.distplot(Y_pred)


# Now we use a Lasso to better choose our alphas. 

# In[24]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
#F
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train, y_train)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# The Lasso regression helps us with the feature selection. Ideally none of the weights/coefficents should be zero.

# In[25]:


# reg2 = Lasso(alpha = 100)
# reg2 = Lasso(alpha = 10)
# reg2 = Lasso(alpha = 1)
# reg2 = Lasso(alpha = 0.1)
reg2 = Lasso(alpha = 55)
reg2.fit(X_train, y_train)
Y_pred_lasso = reg2.predict(X_test)

# The coefficients
print("Coefficients: \n", reg2.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred_lasso))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, Y_pred_lasso))
print(list(zip(reg2.coef_, X.columns)))


# In[26]:


X


# In[27]:


sns.distplot(Y_pred_lasso)


# The Ridge Regression
# 
# We again start by looking for the right alpha

# In[28]:


from sklearn.linear_model import Ridge
ridge=Ridge()
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train, y_train)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# We double check with ridge 

# In[29]:


reg3 = Ridge(alpha = 5)
reg3.fit(X_train, y_train)
Y_pred_ridge = reg3.predict(X_test)

# The coefficients
print("Coefficients: \n", reg3.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred_ridge))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, Y_pred_ridge))


# In[37]:


te = test.copy()
te.dropna()
te.isna()


# ### FINAL TEST ###

# In[32]:



# Impute missing data from test file
X_final_1 = te.drop(columns = ['roomnum', 'organge_county', 'ventura_county', 'lotid','heatingtype', 'numbath', 'heatingdesc', 'countycode', 'citycode', 'countycode2', 'taxyear', 'neighborhoodcode', 'regioncode', 'numfireplace', 'unitnum', 'hd1', 'hd2', 'hd3'], axis = 1)
X_final_1.isnull().sum()
X_final_1[X_final_1['lotarea'].isna()]
X_final_1[X_final_1['year'].isna()]
X_final_1 = X_final_1.drop([3, 19, 33, 130], axis = 0)
a = X_final_1['lotarea'].median()
b = X_final_1['year'].median()
#re = ['lotid']
X_final = te.drop(columns = ['lotid', 'roomnum', 'organge_county', 'ventura_county','heatingtype', 'numbath', 'heatingdesc', 'countycode', 'citycode', 'countycode2', 'taxyear', 'neighborhoodcode', 'regioncode', 'numfireplace', 'unitnum', 'hd1', 'hd2', 'hd3'], axis = 1)
X_final['lotarea'][3] = a
X_final['lotarea'][33] = a
X_final['lotarea'][130] = a
X_final['year'][19] = b
X_final.isnull().sum()

#Now that the data is workable we can apply the predictions
Y_pred_final = regr.predict(X_final)
print(Y_pred_final)
his = sns.histplot(Y_pred_final, bins='auto')
his                   


# In[31]:


# Create CSV
test_predictions_submit = pd.DataFrame({"lotid": te['lotid'], "parcelvalue": Y_pred_final})
test_predictions_submit.to_csv("test_predictions_submit(Ben, Jon, Tobi).csv", index = False)


# In[ ]:


te2 = pd.DataFrame
te.to_csv("city.csv", index = False)


# # Appendix (Learning Experiences) #

# ## Getting crime data
# 
# We had the idea of adding a new feature: crime statistics for a particular set of geographical coordinates. By getting the city name for each row, we could correlate that with FBI crime statistics. 
# 
# However, we found that the region codes and zip codes were all wrong in the training data; the only accurate location data we could use was longitude and latitude. 
# 
# We used a third party API to get the city name for each lotid, but the API wasn't giving us accurate information. Furthermore it had a limit of one request per second, so it wasn't time efficient. 
# 
# We will think more carefully before using exciting API's in the future. See our code below. 

# In[ ]:


import pandas as pd
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="GoogleV3")
import socket
from time import sleep

#define crime data
us_data = pd.read_csv('us_country_data.csv')            .rename(columns={'CITY': 'city', 'POSTAL_CODE': 'regioncode'})
crime = pd.read_csv('ca_crime.csv')
#convert lat/long into API usable integer formats
training['longitude'] = training['longitude'] / 1000000
training['latitude'] = training['latitude'] / 1000000

lats_and_longs = training[['latitude', 'longitude']]
lats_and_longs = lats_and_longs.to_dict()


#initialize and use API
def get_city_info(startnum, stopnum):
    city_values = ['town', 'city', 'suburb', 'village']
    
    count = 0
    output = {"city":[],"count":[]}
    for i in range(startnum, stopnum):
        lotid = str(training['lotid'][i])
        lat = str(lats_and_longs['latitude'][i])
        long = str(lats_and_longs['longitude'][i])
        location = geolocator.reverse(f'{lat},{long}').raw
        for c in city_values:
            if c in location['address'].keys():
                town = str(location['address'][c])
                naming_convention = c
                count += 1
                print(f'{count} | {naming_convention} | {lotid} | {town}')
                output['city'].append(f'{lotid}, {town}, {naming_convention}')
                output['count'].append(count)
                
            else:
                output['city'].append(location)
                continue
    return output
# handle request limitations
try:
    output = get_city_info(0,22000)
except socket.timeout:
    sleep(30)
    output = get_city_info(output['count'][-1], 22000)


# ## Visualizing data with maps
# 
# We wanted to visualize crime data, or other forms of geographical data. 

# In[ ]:


# import the library and its Marker clusterization service
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap, HeatMapWithTime
from datetime import datetime, timedelta
crime = pd.read_csv('ca_crime.csv', sep = ",")

crime_heatmap_df = crime.merge(training, left_on='state', right_on='usa_state_code', how='left')[['usa_state_latitude','usa_state_longitude','new_case']]
covid_heatmap_df.dropna(inplace=True)
covid_heatmap_df.head()
#tiles='Stamen Terrain'
# Create a map object and center it to the avarage coordinates to m
m = folium.Map(location=training[["latitude", "longitude"]].mean().to_list(), zoom_start=10)
# if the points are too close to each other, cluster them, create a cluster overlay with MarkerCluster, add to m
marker_cluster = MarkerCluster().add_to(m)
# draw the markers and assign popup and hover texts
# add the markers the the cluster layers so that they are automatically clustered
for i,r in training.iterrows():
    location = (r["latitude"], r["longitude"])
    folium.Marker(location=location,
                      )\
    .add_to(marker_cluster)
# display the map
m

lat = str(training['latitude'][0])
long = str(training['longitude'][0])
training_locations = training[["latitude", "longitude"]]
t = folium.Map(location=[training_locations.latitude.mean(), training_locations.longitude.mean()],zoom_start=8, control_scale=True)

crime_fbi = pd.read_csv('ca_crime.csv', sep = ",")
crime_fbi.head()

sample_city = pd.read_excel('sample_city_data.xlsx')
sample_city.rename(columns = {'city':'City'}, inplace = True)
sample_city['City'] = sample_city.City.apply(lambda x: x.strip())
crime_fbi.rename(columns = {'Violent\ncrime':'Violent_crime', 'Property\ncrime':'Property_crime','Murder and\nnonnegligent\nmanslaughter':'Murder', 'Aggravated\nassault':'Assault', 'Larceny-\ntheft':'Larceny', 'Motor\nvehicle\ntheft':'Motor_theft'}, inplace = True)
crime_fbi.head()
sample_city_crime=sample_city.merge(crime_fbi, on='City', how='left')

sample_city_crime.isnull().sum()

missing_sample_city_crime= np.where(sample_city_crime['Population'].isnull() == True)
missing_sample_city_crime

#No data on Newbury Park, Lemon Heights,Cowan Heights, Las Posas Estates.
sample_city_crime.iloc[19,:]

sample_city_crime.info()

found_crime=np.where(crime_fbi['City'] == "Newbury Park")
found_crime

sample_city_crime.head(4)

sample_city_crime["soft_crime"] = sample_city_crime['Assault']+sample_city_crime['Property_crime'] + sample_city_crime['Burglary']+sample_city_crime['Larceny']+sample_city_crime['Motor_theft']+sample_city_crime['Arson']
sample_city_crime.head()

#heatmap_df = df.merge(location_df, left_on='state', right_on='usa_state_code', 
                                    # how='left')[['usa_state_latitude','usa_state_longitude','new_case']]
    
    heatmap_df= sample_city_crime.merge(training, on ='lotid', how='left')[['lotid','latitude','longitude','City','soft_crime']]
heatmap_df.dropna(inplace=True)
heatmap_df = heatmap_df.reset_index(drop=True)
hm = folium.Map(location=[34.191029, -118.914689], #Center of USA
               tiles='stamentoner',
               zoom_start=10)
HeatMap(heatmap_df, 
        min_opacity=0.4,
        blur = 18
               ).add_to(folium.FeatureGroup(name='Heat Map').add_to(hm))
folium.LayerControl().add_to(hm)
hm


# Here we have a zoomable heatmap of crime statistics. This is partial due to API problems, but paints an interesting portrait

# ![crimeheatmap.png](attachment:crimeheatmap.png)

# We made a heatmap of the housing locations, which we were hoping to correlate with housing prices. But we didn't have enough time at the end. 

# ![property_locations.png](attachment:property_locations.png)

# In[ ]:




