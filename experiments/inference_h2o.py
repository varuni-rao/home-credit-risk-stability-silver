# Experimental H2O inference — explored for diversity
# Final silver submission used LightAutoML (see train_lightautoml.py)

import numpy as np
import pandas as pd
import argparse
import os
import glob

from sklearn.metrics import roc_auc_score

import h2o
from h2o.automl import H2OAutoML

from sklearn.preprocessing import MinMaxScaler

import pickle
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import gc

import warnings
warnings.filterwarnings("ignore")

from src import utils as hcu

def predict_proba_in_batches(model, data, batch_size=100000):
    num_samples = data.nrow
    num_batches = int(np.ceil(num_samples / batch_size))
    probabilities = np.zeros((num_samples,))
    for batch_idx in range(num_batches):
        print(f"Processing batch: {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_h2o = data[start_idx:end_idx, :]
        batch_probs = model.predict(batch_h2o).as_data_frame()['p1'].values
        probabilities[start_idx:end_idx] = batch_probs
        h2o.remove(batch_h2o)
    return probabilities

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/parquet_files/test")
    parser.add_argument("--model_dir", default="models/h2o")
    parser.add_argument("--features", default="data/processed/h2o_features.pkl")
    parser.add_argument("--col_file", default="data/processed/train_cols_b4_featengg.pkl")
    parser.add_argument("--scaler", default="data/processed/mmscaler.pkl")
    parser.add_argument("--out", default="submission_h2o.csv")
    args = parser.parse_args()

    test_path = args.data_dir + "/"
    test_dict = {
        "base": hcu.DataLoader.get_dataframe(test_path + "test_base.parquet"),
        "depth_0": [hcu.DataLoader.get_dataframe(test_path + "test_static_cb_0.parquet"), hcu.DataLoader.get_dataframe(test_path + "test_static_0_*.parquet", chunked=True)],
        "depth_1": [hcu.DataLoader.get_dataframe(test_path + "test_applprev_1_*.parquet","prev1",1,True), hcu.DataLoader.get_dataframe(test_path + "test_tax_registry_a_1.parquet","tra",1), hcu.DataLoader.get_dataframe(test_path + "test_tax_registry_b_1.parquet","trb",1), hcu.DataLoader.get_dataframe(test_path + "test_tax_registry_c_1.parquet","trc",1), hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_a_1_*.parquet","cba1",1,True), hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_b_1.parquet","cbb1",1), hcu.DataLoader.get_dataframe(test_path + "test_other_1.parquet","oth",1), hcu.DataLoader.get_dataframe(test_path + "test_person_1.parquet","per1",1), hcu.DataLoader.get_dataframe(test_path + "test_deposit_1.parquet","dep",1), hcu.DataLoader.get_dataframe(test_path + "test_debitcard_1.parquet","deb",1)],
        "depth_2": [hcu.DataLoader.get_dataframe(test_path + "test_applprev_2.parquet","prev2",2), hcu.DataLoader.get_dataframe(test_path + "test_person_2.parquet","per2",2), hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_b_2.parquet","cba2",2), hcu.DataLoader.get_dataframe(test_path + "test_credit_bureau_a_2_*.parquet","cbb2",2,True)]
    }

    test_df = hcu.DataPreprocessor.preprocess_test(test_dict, args.col_file)
    del test_dict; hcu.MemoryOptimizer.CleanMemory()

    with open(args.features, 'rb') as f: 
        features = pickle.load(f)

    with open(args.scaler, 'rb') as f: 
        scaler = pickle.load(f)

    test_df = test_df[features + ['case_id','WEEK_NUM']]

    num_cols = [c for c in features if test_df[c].dtype != 'object' and str(test_df[c].dtype) != 'category']
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    h2o.init()
    test_hf = h2o.H2OFrame(test_df[features])
    hcu.MemoryOptimizer.CleanMemory(); gc.collect()
    
    model_files = glob.glob(f"{args.model_dir}/*.zip") + glob.glob(f"{args.model_dir}/*")
    
    model_files = [f for f in model_files if os.path.isdir(f) or f.endswith('.zip')]
    model_path = sorted(model_files)[-1]
    h2omodel = h2o.load_model(model_path)
    print(f"Loaded model: {model_path}")
    
    y_pred = predict_proba_in_batches(h2omodel, test_hf)
    submission = pd.DataFrame({"case_id": test_df["case_id"].to_numpy(), "score": y_pred}).set_index('case_id')
    submission.to_csv(args.out)
    print(f"Saved {args.out}")
