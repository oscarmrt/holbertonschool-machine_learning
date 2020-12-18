#!/usr/bin/env python3
"""Program that preprocess the data"""
import numpy as np


def preproc(datafile, days):
    """Function that preprocess the data"""
    df = pd.read_csv(datafile)
    df = df.iloc[-days * 24 * 60:]
    df = df.drop(['Volume_(Currency)'], axis=1)  # volume dollar deleted
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.drop(['Timestamp'], axis=1)
    df = df.set_index('Datetime')
    df = df.interpolate()
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df_mul = pd.DataFrame()
    df_mul['High'] = df.High.resample('H').max()
    df_mul['Low'] = df.Low.resample('H').min()
    df_mul['Weighted_Price'] = df.Weighted_Price.resample('H').mean()
    df_mul['Volume_BTC'] = df.Volume_BTC.resample('H').sum()
    df_W = df.Weighted_Price.resample('H').mean()
    df_W.plot(subplots=True)
    return df_W, df_mul
