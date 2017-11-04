import csv
import glob
import json
import os

import numpy as np
import pandas as pd

from features import (
    LABEL, NUMERICAL_AGG_AVG, NUMERICAL_AGG_MIN, NUMERICAL_AGG_MAX,
    NUMERICAL_AGG_SUM, NUMERICAL_NON_ULOG, USER_CATEGORICAL,
)

MEMBERS_CSV_PATH = './data/members.csv'
MEMBERS_DF_PICKLE = './data/members_df.pkl'
TRAIN_CSV_PATH = './data/train.csv'
TRAIN_DF_PICKLE = './data/train_df.pkl'
TRANSACTIONS_CSV_PATH = './data/transactions.csv'
STATISTICS_DF_PICKLE = './data/statistics_df.pkl'


def compile_csv_parts_to_larger_csv(csv_parts_path, to_write_path):
    glob_path = os.path.join(csv_parts_path, '*.csv')
    # TODO: clean this up somehow
    fieldnames = [
        'avg(num_100)', 'sum(total_secs)', 'sum(num_50)', 'min(num_75)', 'avg(num_50)',
        'avg(num_985)', 'min(total_secs)', 'min(num_985)', 'sum(num_25)', 'min(num_unq)',
        'sum(num_100)', 'max(num_985)', 'min(num_50)', 'min(num_100)', 'min(num_25)',
        'is_churn', 'sum(num_75)', 'avg(total_secs)', 'avg(num_25)', 'max(num_100)',
        'max(num_75)', 'avg(num_unq)', 'max(total_secs)', 'msno', 'sum(num_985)',
        'max(num_25)', 'avg(num_75)', 'sum(num_unq)', 'max(num_unq)', 'max(num_50)',
    ]
    with open(to_write_path, 'w') as csv_to_write:
        writer = csv.DictWriter(csv_to_write, fieldnames=fieldnames)
        writer.writeheader()

        for fname in glob.glob(glob_path):
            print("Now processing file name {}...".format(fname))
            with open(fname, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                for line in reader:
                    writer.writerow(line)
    print("Finished writing to {}!".format(to_write_path))


def return_column_as_category(uncategorized_column, null_category_val=np.nan):
    """Return a pandas DataFrame column as a categorical Series.

    Let's say you have a pandas Series of 'city' values ranging from 1 to 21. This method converts
    that Series to a categorical Series, which is better to work with.

    Args:
        uncategorized_column - The DataFrame column (of type Series) you want to categorize.
        null_category_val - The category value you want to use for null category values. By default,
                            the category val for null values is np.nan. This arg can be used to
                            specify a more sensible, human-readable null value
                            (e.g., 'not_specified' for a column with gender information).

    Returns:
        A Categorical pandas Series.
    """
    categories = set(uncategorized_column.get_values())
    if null_category_val != np.nan:
        if np.nan in categories:
            categories.remove(np.nan)
        categories.add(null_category_val)

    categorized_column = uncategorized_column.astype(
        'category',
        categories=categories,
    ).fillna(
        null_category_val,
    )
    return categorized_column


def get_or_build_members_df(force_build=False):
    if os.path.isfile(MEMBERS_DF_PICKLE) and not force_build:
        return pd.read_pickle(MEMBERS_DF_PICKLE)

    print("Getting members_df...")
    members_df = pd.read_csv(MEMBERS_CSV_PATH)
    members_df.city = return_column_as_category(members_df.city, 0)
    members_df.registered_via = return_column_as_category(members_df.registered_via, 0)
    members_df.gender = return_column_as_category(members_df.gender, 'not_specified')
    # There are some weird outliers in the age of the users. Clip it to the range [0, 100].
    members_df.bd = members_df.bd.clip(0, 100)
    print("Done getting members_df")

    members_df.to_pickle(MEMBERS_DF_PICKLE)
    return members_df


def get_or_build_statistics_df(left_df, force_build=False):
    """Builds valuable statistics from 'transactions.csv' to use as features
    """
    if os.path.isfile(STATISTICS_DF_PICKLE) and not force_build:
        return pd.read_pickle(STATISTICS_DF_PICKLE)

    print("Reading and grouping transactions_df...")
    transactions_df = pd.read_csv(TRANSACTIONS_CSV_PATH)
    # Join here to get rid of those rows in transactions_df that do not appear in 'left_df'.
    transactions_df = pd.merge(left_df, transactions_df, how='left', on='msno')
    grouped_transactions = transactions_df.groupby('msno')
    print("Finished reading and grouping transactions_df")

    print("Preparing statistics DataFrame from transactions_df...")
    stats_df = grouped_transactions.agg({
        # How many times an individual 'msno' showed up in 'transactions_df'
        'msno': {'num_transactions': 'count'},
        'plan_list_price': {'plan_net_worth': 'sum'},
        'actual_amount_paid': {
            'mean_payment': 'mean',
            'total_payments': 'sum',
        },
        'is_cancel': {
            'times_canceled': lambda x: sum(x == 1)
        },
    })
    stats_df.columns = stats_df.columns.droplevel(0)
    stats_df.reset_index(inplace=True)
    print("Finished preparing statistics DataFrame")

    stats_df.to_pickle(STATISTICS_DF_PICKLE)
    return stats_df


def get_or_build_training_df(force_build=False):
    if os.path.isfile(TRAIN_DF_PICKLE) and not force_build:
        return pd.read_pickle(TRAIN_DF_PICKLE)

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    members_df = get_or_build_members_df(force_build=force_build)

    print("Merging with members_df...")
    train_df = pd.merge(train_df, members_df, how='left', on='msno')
    del members_df

    train_df.city.fillna(0, inplace=True)
    train_df.registered_via.fillna(0, inplace=True)
    train_df.gender.fillna('not_specified', inplace=True)
    train_df.bd.fillna(0, inplace=True)
    train_df.registration_init_time.fillna(0, inplace=True)
    train_df.expiration_date.fillna(0, inplace=True)
    print("Finished merging with members_df")

    stats_df = get_or_build_statistics_df(train_df, force_build=force_build)

    print("Merging with stats_df...")
    train_df = pd.merge(train_df, stats_df, how='left', on='msno')
    del stats_df
    print("Finished merging with stats_df")

    train_df.to_pickle(TRAIN_DF_PICKLE)
    return train_df

###
### UTILITIES FOR WORKING WITH VOWPAL WABBIT
###
def build_vw_json_obj_from_csv_dict(csv_dict):
    json_obj = {
        'user_categorical': {},
        'numerical_non_ulog': {},
        'avg_ulog': {},
        'min_ulog': {},
        # 1 prepended for namespace purposes
        '1max_ulog': {},
        'sum_ulog': {},
    }
    for key, value in csv_dict.items():
        if key == LABEL:
            json_obj['_label'] = -1 if value == 0 else 1
        elif key in USER_CATEGORICAL:
            json_obj['user_categorical'][key] = str(value)
        elif key in NUMERICAL_NON_ULOG:
            json_obj['numerical_non_ulog'][key] = float(value)
        elif key in NUMERICAL_AGG_AVG:
            json_obj['avg_ulog'][key] = float(value)
        elif key in NUMERICAL_AGG_MIN:
            json_obj['min_ulog'][key] = float(value)
        elif key in NUMERICAL_AGG_MAX:
            json_obj['1max_ulog'][key] = float(value)
        elif key in NUMERICAL_AGG_SUM:
            json_obj['sum_ulog'][key] = float(value)

    return json_obj


def write_vw_json_lines(csv_file_to_read, json_file_to_write):
    with open(json_file_to_write, 'w') as json_file:
        with open(csv_file_to_read, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for i, row in enumerate(reader):
                if i % 4000 == 0:
                    print("Now processing line {} of {}...".format(i, csv_file_to_read))
                json_obj = build_vw_json_obj_from_csv_dict(row)
                json_file.write('{}\n'.format(json.dumps(json_obj)))
    print("All done writing to {}!".format(json_file_to_write))
    return
