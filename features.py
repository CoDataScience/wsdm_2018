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
    'avg(num_unq)',
    'avg(total_secs)',
    'avg(num_25)',
    'avg(num_50)',
    'avg(num_75)',
    'avg(num_985)',
    'avg(num_100)',
]

NUMERICAL_AGG_MAX = [
    'max(num_unq)',
    'max(total_secs)',
    'max(num_25)',
    'max(num_50)',
    'max(num_75)',
    'max(num_985)',
    'max(num_100)',
]

NUMERICAL_AGG_MIN = [
    'min(num_unq)',
    'min(total_secs)',
    'min(num_25)',
    'min(num_50)',
    'min(num_75)',
    'min(num_985)',
    'min(num_100)',
]

NUMERICAL_AGG_SUM = [
    'sum(num_unq)',
    'sum(total_secs)',
    'sum(num_25)',
    'sum(num_50)',
    'sum(num_75)',
    'sum(num_985)',
    'sum(num_100)',
]
