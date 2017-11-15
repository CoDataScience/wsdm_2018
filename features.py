LABEL = 'is_churn'

USER_CATEGORICAL = [
    'city',
    'gender',
    'registered_via',
]

NUMERICAL_NON_ULOG = [
    'plan_net_worth',
    'num_transactions',
    'times_canceled',
    'mean_payment',
    'total_payments',
]

NUMERICAL_AGG_AVG = [
    'avg_num_unq',
    'avg_total_secs',
    'avg_num_25',
    'avg_num_50',
    'avg_num_75',
    'avg_num_985',
    'avg_num_100',
]

NUMERICAL_AGG_MAX = [
    'max_num_unq',
    'max_total_secs',
    'max_num_25',
    'max_num_50',
    'max_num_75',
    'max_num_985',
    'max_num_100',
]

NUMERICAL_AGG_MIN = [
    'min_num_unq',
    'min_total_secs',
    'min_num_25',
    'min_num_50',
    'min_num_75',
    'min_num_985',
    'min_num_100',
]

NUMERICAL_AGG_SUM = [
    'sum_num_unq)',
    'sum_total_secs)',
    'sum_num_25',
    'sum_num_50',
    'sum_num_75',
    'sum_num_985',
    'sum_num_100',
]

NUMERICAL_AGG_SUM = [
    'sum_num_unq',
    'sum_total_secs',
    'sum_num_25',
    'sum_num_50',
    'sum_num_75',
    'sum_num_985',
    'sum_num_100',
]

NUMERICAL_AGG_STDDEV = [
    'stddev_num_unq',
    'stddev_total_secs',
    'stddev_num_25',
    'stddev_num_50',
    'stddev_num_75',
    'stddev_num_985',
    'stddev_num_100',
]
