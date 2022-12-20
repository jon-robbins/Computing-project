# feature creation

import pandas as pd


def create_heating_dummies(training_df, test_df):
    # Create Variables for the heating type
    # This is a roundabout method. If anyone has a way to optimize this code, feel free to contribute. For our purposes it works.
    heating_labels = {
        'heatingtype': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
                        14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25},
        'heatingdesc': {0: 'Other', 1: 'Baseboard', 2: 'Central', 3: 'Coal', 4: 'Convection', 5: 'Electric',
                        6: 'Forced air', 7: 'Floor/Wall', 8: 'Gas', 9: 'Geo Thermal', 10: 'Gravity', 11: 'Heat Pump',
                        12: 'Hot Water', 13: 'None', 14: 'Other', 15: 'Oil', 16: 'Partial', 17: 'Propane',
                        18: 'Radiant', 19: 'Steam', 20: 'Solar', 21: 'Space/Suspended', 22: 'Vent', 23: 'Wood Burning',
                        24: 'Yes', 25: 'Zone'}}
    heating_labels = pd.DataFrame.from_dict(heating_labels)

    training_df['heatingtype'] = training['heatingtype'].where(~training['heatingtype'].isin([1, 2, 3]), 0)
    test_df['heatingtype'] = test_df['heatingtype'].where(~test_df['heatingtype'].isin([1, 2, 3]), 0)

    training_df = pd.merge(training, heating_labels, on='heatingtype', how='left')
    test_df = pd.merge(test_df, heating_labels, on='heatingtype', how='left')
    heating_dummies = pd.get_dummies(training['heatingtype'])
    heating_dummies2 = pd.get_dummies(test_df['heatingtype'])

    training_df['hd1'] = heating_dummies.iloc[:, 0]
    training_df['hd2'] = heating_dummies.iloc[:, 1]
    training_df['hd3'] = heating_dummies.iloc[:, 2]
    # training_df['hd4'] = heating_dummies.iloc[:, 3]

    test_df['hd1'] = heating_dummies2.iloc[:, 0]
    test_df['hd2'] = heating_dummies2.iloc[:, 1]
    test_df['hd3'] = heating_dummies2.iloc[:, 2]
    # test_df['hd4'] = heating_dummies2.iloc[:, 3]

    training_df.drop(columns='heatingdesc', axis=1)
    test_df.drop(columns='heatingdesc', axis=1)
    return training_df, test_df


## create dummies for each county
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


def create_dummies(df, cols):
    try:
        df = pd.get_dummies(df, columns=cols)
        return df
    except:
        print("there's an error")
        return None
