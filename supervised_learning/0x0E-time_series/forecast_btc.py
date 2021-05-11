#!/usr/bin/env python3
""" XGBoost Forecasting for bitcoin"""

from xgboost import plot_importance, plot_tree
import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')
preprocess_data = __import__('preprocess_data').preprocess_data


def visualize_data(data):
    """Visualize in a histogram the initial data of weight vs time

    Args:
            data (df): data to be visualized
    """
    color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38",
                 "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
    _ = data.plot(style='', figsize=(
        15, 5), color=color_pal[0],
        title='BTC Weighted_Price Price (USD) by Hours')


def create_features(df, label=None):
    """Creating features for data

    Args:
        df: data-frame variable containing extracted data
        label: determine whether label should be present or not

    Returns:
        data
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


def forecast_btc():
    """Predicts the bitcoin value in the upcoming days

    Returns:
        tuple: returns the MSE and MAE
    """

    # preprocess data and prepare it for training
    file = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    data = preprocess_data(file)

    # Initial vizualiztion of data
    visualize_data(data)

    # splitting data into trrain data and text data
    split_date = '25-Jun-2018'
    data_train = data.loc[data.index <= split_date].copy()
    data_test = data.loc[data.index > split_date].copy()

    # Visualize the splitted data
    _ = data_test \
        .rename(columns={'Weighted_Price': 'Test Set'}) \
        .join(data_train.rename(
            columns={'Weighted_Price': 'Training Set'}), how='outer') \
        .plot(figsize=(15, 5),
              title='BTC Weighted_Price Price (USD) by Hours', style='')

    # create features for training and testing data
    X_train, y_train = create_features(data_train, label='Weighted_Price')
    X_test, y_test = create_features(data_test, label='Weighted_Price')

    model = xgb.XGBRegressor(objective='reg:linear', min_child_weight=10,
                             booster='gbtree', colsample_bytree=0.19,
                             learning_rate=0.135, max_depth=5, alpha=10,
                             n_estimators=100)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=50,
              verbose=False)

    data_test['Weighted_Price_Prediction'] = model.predict(X_test)
    data_all = pd.concat([data_test, data_train], sort=False)

    _ = data_all[['Weighted_Price', 'Weighted_Price_Prediction']].plot(
        figsize=(15, 5))

    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = data_all[['Weighted_Price_Prediction',
                  'Weighted_Price'
                  ]].plot(ax=ax,
                          style=['-', '.'])
    ax.set_xbound(lower='08-01-2018', upper='09-01-2018')
    ax.set_ylim(0, 10000)
    plot = plt.suptitle('August 2018 Forecast vs Actuals')

    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = data_all[['Weighted_Price_Prediction',
                  'Weighted_Price']].plot(ax=ax,
                                          style=['-', '.'])
    ax.set_xbound(lower='08-01-2018', upper='08-08-2018')
    ax.set_ylim(0, 10000)
    plot = plt.suptitle('First Week of August 2018 Forecast vs Actuals')

    MSE = mean_squared_error(y_true=data_test['Weighted_Price'],
                             y_pred=data_test['Weighted_Price_Prediction'])

    MAE = mean_absolute_error(y_true=data_test['Weighted_Price'],
                              y_pred=data_test['Weighted_Price_Prediction'])

    return MSE, MAE


# data length is 2099760
if __name__ == '__main__':
    result = forecast_btc()
    print(result)
