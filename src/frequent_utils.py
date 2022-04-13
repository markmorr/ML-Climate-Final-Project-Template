# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:11:38 2022

@author: 16028
"""
def genDatetimeFeatures(df):
    df['datetime'] = pd.to_datetime(df['feature_0'], format='%m-%d %X')
    min_day = dt(1900, 1, 1, 00, 00)
    df['days_since'] = (df['datetime'] - min_day).dt.days
    df["m"] = df.datetime.dt.month
    df["d"] = df.datetime.dt.day
    df["h"] = df.datetime.dt.hour
    df['day_of_week'] = df['days_since'] & 7
    return df


def runOneHot(df, feature_name):
    one_hot = pd.get_dummies(df[feature_name], prefix=feature_name)
    df = df.drop(feature_name,axis = 1)
    df = df.join(one_hot)
    return df