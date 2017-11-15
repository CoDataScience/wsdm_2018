import csv
import glob
import json
import os

import numpy as np
import pandas as pd

from features import (
    LABEL, NUMERICAL_AGG_AVG, NUMERICAL_AGG_MIN, NUMERICAL_AGG_MAX,
    NUMERICAL_AGG_STDDEV, NUMERICAL_AGG_SUM, NUMERICAL_NON_ULOG, USER_CATEGORICAL,
)

MEMBERS_CSV_PATH = './data/members_v3.csv'
MEMBERS_DF_PICKLE = './data/members_df.pkl'
TRAIN_CSV_PATH = './data/train_v2.csv'
TRAIN_DF_PICKLE = './data/train_df.pkl'
TRAIN_DF_PICKLE_VW = './data/train_df_vw.pkl'
TRAIN_ULOG_PATH = './data/train_ulog_features.csv'
VALIDATION_CSV_PATH = './data/sample_submission_v2.csv'
VALIDATION_DF_PICKLE = './data/validation_df.pkl'
VALIDATION_DF_PICKLE_VW = './data/validation_df_vw.pkl'
VALIDATION_ULOG_PATH = './data/validation_ulog_features.csv'
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
        'stddev(num_25)', 'stddev(num_unq)', 'stddev(num_100)', 'stddev(num_50)',
        'stddev(total_secs)', 'stddev(num_75)', 'stddev(num_985)',
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
    members_df.gender = return_column_as_category(members_df.gender, 'not_specified')
    members_df.city = return_column_as_category(members_df.city, 0)
    members_df.registered_via = return_column_as_category(members_df.registered_via, 0)
    print("Done getting members_df")

    members_df.to_pickle(MEMBERS_DF_PICKLE)
    return members_df


def get_or_build_statistics_df(left_df, force_build=False):
    """Builds valuable statistics from 'transactions.csv' to use as features
    """
    if os.path.isfile(STATISTICS_DF_PICKLE) and not force_build:
        return pd.read_pickle(STATISTICS_DF_PICKLE)

    print("Reading and grouping transactions_df...")
    df_transactions = pd.read_csv(TRANSACTIONS_CSV_PATH)
    # Join here to get rid of those rows in df_transactions that do not appear in 'left_df'.
    df_transactions = pd.merge(left_df, df_transactions, how='left', on='msno')
    # Add some new features to df_transactions
    df_transactions['discount'] = (
        df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']
    )
    df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
    df_transactions['amt_per_day'] = (
        df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']
    )
    date_cols = ['transaction_date', 'membership_expire_date']
    for col in date_cols:
        df_transactions[col] = pd.to_datetime(df_transactions[col], format='%Y%m%d')
    df_transactions['membership_duration'] = (
        df_transactions.membership_expire_date - df_transactions.transaction_date
    )
    df_transactions['membership_duration'] = (
        df_transactions['membership_duration'] / np.timedelta64(1, 'D')
    )
    df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)

    grouped_transactions = df_transactions.groupby('msno')
    print("Finished reading and grouping df_transactions")

    print("Preparing statistics DataFrame from df_transactions...")
    stats_df = grouped_transactions.agg({
        # How many times an individual 'msno' showed up in 'df_transactions'
        'msno': {'num_transactions': 'count'},
        'plan_list_price': {'plan_net_worth': 'sum'},
        'actual_amount_paid': {
            'mean_payment': 'mean',
            'total_payments': 'sum',
        },
        'is_cancel': {
            'times_canceled': lambda x: sum(x == 1)
        },
        'is_discount': {
            'num_discounts': lambda x: sum(x == 1)
        },
        'discount': {
            'total_discount': 'sum'
        },
        'membership_duration': {
            'mean_membership_duration': 'mean',
            'total_membership_duration': 'sum',
        },
        'amt_per_day': {
            'mean_amt_per_day': 'mean',
            'total_amt_per_day': 'sum',
        },
    })
    stats_df.columns = stats_df.columns.droplevel(0)
    stats_df.reset_index(inplace=True)
    print("Finished preparing statistics DataFrame")

    stats_df.to_pickle(STATISTICS_DF_PICKLE)
    return stats_df


def get_or_build_training_or_validation_df(validation=False, force_build=False, for_vw=False):
    if not validation:
        csv_path = TRAIN_CSV_PATH
        if not for_vw:
            pickle_path = TRAIN_DF_PICKLE
        else:
            pickle_path = TRAIN_DF_PICKLE_VW
    else:
        csv_path = VALIDATION_CSV_PATH
        if not for_vw:
            pickle_path = VALIDATION_DF_PICKLE
        else:
            pickle_path = VALIDATION_DF_PICKLE_VW
    if os.path.isfile(pickle_path) and not force_build:
        return pd.read_pickle(pickle_path)

    df = pd.read_csv(csv_path)
    if validation:
        df.drop('is_churn', axis=1, inplace=True)
    members_df = get_or_build_members_df(force_build=force_build)

    print("Merging with members_df...")
    df = pd.merge(df, members_df, how='left', on='msno')
    del members_df
    df.gender.fillna('not_specified', inplace=True)
    df.city.fillna(0, inplace=True)
    df.registered_via.fillna(0, inplace=True)
    df.bd = df.bd.clip(0, 100)
    df.bd.fillna(0., inplace=True)
    if not for_vw:
        # If we *are* preparing a DataFrame for VowpalWabbit, we do not want to one-hot encode the
        # categorical variables, as that happens in VowPalWabbit via the hashing trick.
        df = pd.concat(
            [df, pd.get_dummies(df.gender)],
            axis=1,
        ).drop('gender', axis=1)
        df = pd.concat(
            [df, pd.get_dummies(df.city, prefix='city')],
            axis=1,
        ).drop('city', axis=1)
        df = pd.concat(
            [df, pd.get_dummies(df.registered_via, prefix='registered_via')],
            axis=1,
        ).drop('registered_via', axis=1)

    print("Finished merging with members_df")

    stats_df = get_or_build_statistics_df(df, force_build=force_build)

    print("Merging with stats_df...")
    df = pd.merge(df, stats_df, how='left', on='msno')
    del stats_df
    print("Finished merging with stats_df")

    ulog_path = TRAIN_ULOG_PATH if not validation else VALIDATION_ULOG_PATH
    ulog_df = pd.read_csv(ulog_path)
    print("Merging with compiled user log data...")
    df = pd.merge(df, ulog_df, how='left', on='msno')
    df.drop(['registration_init_time'], axis=1, inplace=True)
    na_cols = df.columns[df.isnull().any()].tolist()
    df[na_cols] = df[na_cols].fillna(0.)

    df.to_pickle(pickle_path)
    return df

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
        # 2 prepended for namespace purposes
        '2stddev_ulog': {},
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
        elif key in NUMERICAL_AGG_STDDEV:
            json_obj['2stddev_ulog'][key] = float(value)

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
