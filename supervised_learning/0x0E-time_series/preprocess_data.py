#!/usr/bin/env python3
"""Preprocess data"""
import pandas as pd
from datetime import datetime
import pytz


# data processing, CSV file I/O (e.g. pd.read_csv)
def preprocess_data(file):
    """ Preprocess the csv file data and select desirable features

    Args:
            file (csv): the coinbase dataset
    """

    def dateparse(time_in_secs):
        """Parse the data string into a more favorable format

        Args:
                time_in_secs: time to be arsed

        Returns:
                datetime: in corresponding timezone
        """
        return pytz.utc.localize(datetime.fromtimestamp(float(time_in_secs)))

    df = pd.read_csv(file, parse_dates=[0], date_parser=dateparse)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    df = df.groupby([pd.Grouper(key='Timestamp', freq='H')]
                    ).first().reset_index()
    df = df.set_index('Timestamp')
    df = df[['Weighted_Price']]
    df['Weighted_Price'].fillna(method='ffill', inplace=True)

    return df
