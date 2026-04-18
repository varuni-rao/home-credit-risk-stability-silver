# Experimental LightGBM ensemble inference — explored for diversity
# Final silver submission used LightAutoML (see train_lightautoml.py)

"""Import dependencies and set up environment for inference. This includes loading the test data, preprocessing it, and making predictions using the trained LightGBM ensemble model. The code also includes batch prediction to handle large datasets efficiently."""

import numpy as np # linear algebra
import pandas as pd # data processing
import argparse

# library for LightGBM Model
import lightgbm as lgb
from lightgbm import LGBMClassifier

# loading library
from sklearn.preprocessing import MinMaxScaler

# library to read and write date related files
import pickle

# library for saving and loading trained models
import joblib

# display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# library for garbage collection
import gc  # since the data is huge here, regularly cleaning up will free up memory

# import utility 
from src import utils as hcu #version1

# library to catch and ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""Function to predict probabilities in batches to manage memory usage when dealing with large datasets. This function takes a trained model, the data to predict on, and a specified batch size, and returns the predicted probabilities for the entire dataset."""
def predict_proba_in_batches(model, data, batch_size=100000):
    num_samples = len(data)
    num_batches = int(np.ceil(num_samples / batch_size))
    probabilities = np.zeros((num_samples,))

    for batch_idx in range(num_batches):
        print(f"Processing batch: {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        X_batch = data.iloc[start_idx:end_idx]
        batch_probs = model.predict_proba(X_batch)[:, 1]
        probabilities[start_idx:end_idx] = batch_probs
        gc.collect()

    return probabilities

if __name__ == "__main__":

    """Data Loading and Preprocessing"""
    # declare directories for download path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/parquet_files/test")
    parser.add_argument("--model", default="models/lgbm/lgbm_voting.joblib")
    parser.add_argument("--cat_cols", default="data/processed/lgbm_cat_cols.pkl")
    parser.add_argument("--col_file", default="data/processed/train_cols_b4_featengg.pkl")
    parser.add_argument("--cols", default="data/processed/lgbm_features.pkl")
    parser.add_argument("--out", default="submission_lgbm.csv")
    args = parser.parse_args()
    test_path = args.data_dir + "/"

    # dictionary of data paths for train data
    test_dict = {
        "base": hcu.DataLoader.get_dataframe(test_path + "test_base.parquet"),
        "depth_0": [
            hcu.DataLoader.get_dataframe(test_path + "test_static_cb_0.parquet"),
            hcu.DataLoader.get_dataframe(test_path + "test_static_0_*.parquet", chunked= True),
            ],
        "depth_1": [
            hcu.DataLoader.get_dataframe(test_path + "test_applprev_1_*.parquet","prev1", 1, True),
            hcu.DataLoader.get_dataframe(test_path + "test_tax_registry_a_1.parquet","tra", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_tax_registry_b_1.parquet","trb", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_tax_registry_c_1.parquet","trc", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_a_1_*.parquet","cba1", 1, True),
            hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_b_1.parquet","cbb1", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_other_1.parquet","oth", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_person_1.parquet","per1", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_deposit_1.parquet","dep", 1),
            hcu.DataLoader.get_dataframe(test_path + "test_debitcard_1.parquet","deb", 1),
            ],
        "depth_2": [
            hcu.DataLoader.get_dataframe(test_path + "test_applprev_2.parquet","prev2", 2),
            hcu.DataLoader.get_dataframe(test_path + "test_person_2.parquet","per2", 2),
            hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_b_2.parquet","cba2", 2),
            hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_a_2_*.parquet","cbb2", 2, True),
            ]
    }

    col_filename = args.col_file
    cat_filename = args.cat_cols
    test_df = hcu.DataPreprocessor.preprocess_test(test_dict, col_filename)
    hcu.MemoryOptimizer.CleanMemory()

    del test_dict
    hcu.MemoryOptimizer.CleanMemory()

    test_df.info()

    """Data cleaning and feature selection"""
    # select same features as in train
    # load train_df_columns.pkl
    with open(args.cols, 'rb') as f:
        df_columns = pickle.load(f)
    df_columns.remove('target')

    test_df = test_df[df_columns]
    test_df.info()

    # describe() first 5 columns of train_df
    test_df.iloc[:, :4].describe().T

    # split test dataframe
    X_test = test_df.copy()
    X_test.drop(['case_id','WEEK_NUM'], axis = 1, inplace = True)

    X_test = test_df.copy()
    X_test.drop(['case_id','WEEK_NUM'], axis=1, inplace=True)

    with open(args.cat_cols, 'rb') as f:
        cat_cols = pickle.load(f)
    X_test[cat_cols] = X_test[cat_cols].astype('category')

    y_test = test_df[['case_id', 'WEEK_NUM']]

    # display dimensions of X_test and y_test dataframes
    print("Dimensions of X dataframe:", X_test.shape)
    print("Dimensions of y dataframe:", y_test.shape)

    X_test.info()

    hcu.MemoryOptimizer.CleanMemory()
    gc.collect()

    # load the voting classifier model
    vmodel = joblib.load(args.model)
    X_test.info()

    y_pred = predict_proba_in_batches(vmodel, X_test) #, index=X_test.index)
    y_pred


    """Submission File Creation"""

    submission = pd.DataFrame({
        "case_id": y_test["case_id"].to_numpy(),
        "score": y_pred
    }).set_index('case_id')
    submission.to_csv("./submission.csv")