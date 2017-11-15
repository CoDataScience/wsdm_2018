import getpass
import os
import subprocess

import pyunpack
import requests

CUR_SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

KAGGLE_BASE_DATA_PATH = 'https://www.kaggle.com/c/kkbox-churn-prediction-challenge/download/'
LOCAL_DATA_PATH = os.path.join(CUR_SCRIPT_PATH, 'data')

KAGGLE_TRAIN_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'train_v2.csv.7z')
TRAIN_DATA_V2_FNAME = 'train_v2.csv'
LOCAL_TRAIN_DATA_ARCHIVE_PATH = os.path.join(LOCAL_DATA_PATH, '{}.7z'.format(TRAIN_DATA_V2_FNAME))

KAGGLE_VALIDATION_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'sample_submission_v2.csv.7z')
VALIDATION_DATA_V2_FNAME = 'sample_submission_v2.csv'
LOCAL_VALIDATION_DATA_ARCHIVE_PATH = os.path.join(
    LOCAL_DATA_PATH,
    '{}.7z'.format(VALIDATION_DATA_V2_FNAME),
)

KAGGLE_MEMBERS_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'members_v3.csv.7z')
MEMBERS_DATA_V3_FNAME = 'members_v3.csv'
LOCAL_MEMBERS_DATA_ARCHIVE_PATH = os.path.join(
    LOCAL_DATA_PATH,
    '{}.7z'.format(MEMBERS_DATA_V3_FNAME),
)

KAGGLE_V1_TRANSACTIONS_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'transactions.csv.7z')
TRANSACTIONS_DATA_V1_FNAME = 'transactions.csv'
LOCAL_V1_TRANSACTIONS_DATA_ARCHIVE_PATH = os.path.join(
    LOCAL_DATA_PATH,
    '{}.7z'.format(TRANSACTIONS_DATA_V1_FNAME),
)

KAGGLE_V2_TRANSACTIONS_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'transactions_v2.csv.7z')
TRANSACTIONS_DATA_V2_FNAME = 'transactions_v2.csv'
LOCAL_V2_TRANSACTIONS_DATA_ARCHIVE_PATH = os.path.join(
    LOCAL_DATA_PATH,
    '{}.7z'.format(TRANSACTIONS_DATA_V2_FNAME),
)

KAGGLE_V1_ULOGS_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'user_logs.csv.7z')
ULOGS_DATA_V1_FNAME = 'user_logs.csv'
LOCAL_V1_ULOGS_DATA_ARCHIVE_PATH = os.path.join(
    LOCAL_DATA_PATH,
    '{}.7z'.format(ULOGS_DATA_V1_FNAME),
)

KAGGLE_V2_ULOGS_DATA_PATH = os.path.join(KAGGLE_BASE_DATA_PATH, 'user_logs_v2.csv.7z')
ULOGS_DATA_V2_FNAME = 'user_logs_v2.csv'
LOCAL_V2_ULOGS_DATA_ARCHIVE_PATH = os.path.join(
    LOCAL_DATA_PATH,
    '{}.7z'.format(ULOGS_DATA_V2_FNAME),
)

# Kaggle archives ending in 'v2' will be extracted to this base directory
KAGGLE_V2_ARCHIVE_BASE_PATH = 'data/churn_comp_refresh'

FILE_CHUNK_SIZE = 512 * 1024
# Set this variable if you don't want to keep typing in your Kaggle username everytime you run
# this script
KAGGLE_USERNAME = ''


def download_kaggle_archive_and_write_to_local_path(kaggle_user_info, kaggle_archive_path,
                                                    local_archive_path):
    print("Downloading Kaggle data to {}...".format(local_archive_path))
    resp = requests.get(kaggle_archive_path)
    # Login to Kaggle and retrieve the data.
    resp = requests.post(resp.url, data=kaggle_user_info)
    # Writes the data to a local file one chunk at a time.
    with open(local_archive_path, 'wb') as local_data_archive_file:
         # Reads CHUNK_SIZE MB at a time into memory
        for chunk in resp.iter_content(chunk_size=FILE_CHUNK_SIZE):
            # filter out keep-alive new chunks
            if chunk:
                local_data_archive_file.write(chunk)
    print("Finished downloading Kaggle data to {}!".format(local_archive_path))


def extract_kaggle_archive_to_local_path(local_archive_path, local_fname):
    # Annoyingly we have to special case Kaggle files that have the string 'v2' in them.
    is_v2_archive = 'v2' in local_archive_path

    print("Extracting {} to {}...".format(local_archive_path, local_fname))
    archive = pyunpack.Archive(local_archive_path)
    archive.extractall(LOCAL_DATA_PATH)
    if is_v2_archive:
        # This is the directory to which the archive was extracted
        extract_dir = os.path.join(LOCAL_DATA_PATH, KAGGLE_V2_ARCHIVE_BASE_PATH)
        extracted_location = os.path.join(extract_dir, local_fname)
        desired_location = os.path.join(LOCAL_DATA_PATH, local_fname)
        os.rename(extracted_location, desired_location)
        print("Removing needless extracted directories...")
        os.rmdir(extract_dir)
        os.rmdir(os.path.join(LOCAL_DATA_PATH, 'data'))
        print("Done removing those needless directories!")

    print("Done with extraction, removing {}...".format(local_archive_path))
    os.remove(local_archive_path)
    print("All done extracting!")


def merge_csvs(csv_path1, csv_path2):
    # Do some tricky Unix stuff to delete the header of the second CSV and append that to the end of
    # the first CSV.
    sed_call = "sed -i '' 1d {}".format(csv_path2)
    cat_call = "cat {} >> {}".format(csv_path2, csv_path1)
    subprocess.call(sed_call, shell=True)
    subprocess.call(cat_call, shell=True)


def main():
    if not KAGGLE_USERNAME:
        uname = input("Enter your Kaggle username: ")
    pw = getpass.getpass("Enter your Kaggle account password: ")
    kaggle_user_info = {
        'UserName': KAGGLE_USERNAME or uname,
        'Password': pw,
    }

    print("Beginning download and extraction of training data...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_TRAIN_DATA_PATH,
        LOCAL_TRAIN_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_TRAIN_DATA_ARCHIVE_PATH,
        TRAIN_DATA_V2_FNAME,
    )
    print("\nAll done downloading and extracting training data!\n")

    print("Beginning download and extraction of validation data...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_VALIDATION_DATA_PATH,
        LOCAL_VALIDATION_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_VALIDATION_DATA_ARCHIVE_PATH,
        VALIDATION_DATA_V2_FNAME,
    )
    print("\nAll done downloading and extracting validation data!\n")

    print("Beginning download and extraction of members data...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_MEMBERS_DATA_PATH,
        LOCAL_MEMBERS_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_MEMBERS_DATA_ARCHIVE_PATH,
        MEMBERS_DATA_V3_FNAME,
    )
    print("\nAll done downloading and extracting members data!\n")

    print("Beginning download and extraction of transactions data (v1)...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_V1_TRANSACTIONS_DATA_PATH,
        LOCAL_V1_TRANSACTIONS_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_V1_TRANSACTIONS_DATA_ARCHIVE_PATH,
        TRANSACTIONS_DATA_V1_FNAME,
    )
    print("\nAll done downloading and extracting transactions data (v1)!\n")

    print("Beginning download and extraction of transactions data (v2)...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_V2_TRANSACTIONS_DATA_PATH,
        LOCAL_V2_TRANSACTIONS_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_V2_TRANSACTIONS_DATA_ARCHIVE_PATH,
        TRANSACTIONS_DATA_V2_FNAME,
    )
    print("\nAll done downloading and extracting transactions data (v2)!\n")

    print("Beginning merge of v1 and v2 transactions data...\n")
    transactions_v1_path = os.path.join(LOCAL_DATA_PATH, TRANSACTIONS_DATA_V1_FNAME)
    transactions_v2_path = os.path.join(LOCAL_DATA_PATH, TRANSACTIONS_DATA_V2_FNAME)
    merge_csvs(transactions_v1_path, transactions_v2_path)
    os.remove(transactions_v2_path)
    print("All done merging v1 and v2 transactions data files!")

    print("Beginning download and extraction of user logs data (v1)...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_V1_ULOGS_DATA_PATH,
        LOCAL_V1_ULOGS_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_V1_ULOGS_DATA_ARCHIVE_PATH,
        ULOGS_DATA_V1_FNAME,
    )
    print("\nAll done downloading and extracting user logs data (v1)!\n")

    print("Beginning download and extraction of user logs data (v2)...\n")
    download_kaggle_archive_and_write_to_local_path(
        kaggle_user_info,
        KAGGLE_V2_ULOGS_DATA_PATH,
        LOCAL_V2_ULOGS_DATA_ARCHIVE_PATH,
    )
    extract_kaggle_archive_to_local_path(
        LOCAL_V2_ULOGS_DATA_ARCHIVE_PATH,
        ULOGS_DATA_V2_FNAME,
    )
    print("\nAll done downloading and extracting user logs data (v2)!\n")

    print("Beginning merge of v1 and v2 user logs data...\n")
    ulogs_v1_path = os.path.join(LOCAL_DATA_PATH, ULOGS_DATA_V1_FNAME)
    ulogs_v2_path = os.path.join(LOCAL_DATA_PATH, ULOGS_DATA_V2_FNAME)
    merge_csvs(ulogs_v1_path, ulogs_v2_path)
    os.remove(ulogs_v2_path)
    print("All done merging v1 and v2 user logs data files!\n")

    print("All done with everything, happy modeling!")

if __name__ == '__main__':
    main()
