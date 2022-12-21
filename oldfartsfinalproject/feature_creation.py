# feature creation
import pandas as pd


def create_dummies(df, cols):
    try:
        df = pd.get_dummies(df, columns=cols)
        return df
    except:
        print("there's an error")
        return None


def upper_outlier_dummy(df, col: str):
    from scipy.stats import iqr
    import numpy as np
    iqr = iqr(df.col, nan_policy='omit') * 1.5

    q3 = np.nanquantile(df.col, 0.75)
    upper_bound = iqr + q3

    df.loc[df.col > upper_bound, f'{col}_upper_outlier'] = 1
    df.loc[df.col <= upper_bound, f'{col}upper_outlier'] = 0
    return df


def lower_outlier_dummy(df, col: str):
    from scipy.stats import iqr
    import numpy as np
    iqr = iqr(df.col, nan_policy='omit') * 1.5

    q1 = np.nanquantile(df.col, 0.25)
    lower_bound = q1 - iqr

    df.loc[df.col < lower_bound, f'{col}_upper_outlier'] = 1
    df.loc[df.col >= lower_bound, f'{col}upper_outlier'] = 0
    return df

# import pandas as pd
# from geopy.geocoders import Nominatim
# geolocator = Nominatim(user_agent="GoogleV3")
# import socket
# from time import sleep

# #define crime data
# us_data = pd.read_csv('us_country_data.csv')\
#             .rename(columns={'CITY': 'city', 'POSTAL_CODE': 'regioncode'})
# crime = pd.read_csv('ca_crime.csv')
# #convert lat/long into API usable integer formats
# training['longitude'] = training['longitude'] / 1000000
# training['latitude'] = training['latitude'] / 1000000

# lats_and_longs = training[['latitude', 'longitude']]
# lats_and_longs = lats_and_longs.to_dict()


# #initialize and use API
# def get_city_info(startnum, stopnum):
#     city_values = ['town', 'city', 'suburb', 'village']
    
#     count = 0
#     output = {"city":[],"count":[]}
#     for i in range(startnum, stopnum):
#         lotid = str(training['lotid'][i])
#         lat = str(lats_and_longs['latitude'][i])
#         long = str(lats_and_longs['longitude'][i])
#         location = geolocator.reverse(f'{lat},{long}').raw
#         for c in city_values:
#             if c in location['address'].keys():
#                 town = str(location['address'][c])
#                 naming_convention = c
#                 count += 1
#                 print(f'{count} | {naming_convention} | {lotid} | {town}')
#                 output['city'].append(f'{lotid}, {town}, {naming_convention}')
#                 output['count'].append(count)
                
#             else:
#                 output['city'].append(location)
#                 continue
#     return output
# # handle request limitations
# try:
#     output = get_city_info(0,22000)
# except socket.timeout:
#     sleep(30)
#     output = get_city_info(output['count'][-1], 22000)
