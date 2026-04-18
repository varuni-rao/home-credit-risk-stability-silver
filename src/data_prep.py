"""This script is for data preparation. It includes loading the data, preprocessing the data, and feature engineering the data. The final output of this script is a cleaned and preprocessed dataframe that is ready for modeling"""
import numpy as np # linear algebra
import pandas as pd # data processing

# import further libraries
import matplotlib.pyplot as plt
import seaborn as sns

# library for MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# library to read and write files
import pickle

# library to save and load models
import joblib

# display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# library for garbage collection
import gc  # since the data is huge here, regularly cleaning up will free up memory

# library to catch and ignore warnings
import warnings
warnings.filterwarnings("ignore")

# library for utility script containing various pipelines
from src import utils as hcu

"""Loading data onto dataframes"""
# this code below is inspired by and borrowed from the following notebook by FARUKCAN SAGLAM
# https://www.kaggle.com/code/dksdms4/lb-0-565-improved-baseline-notebook

# declare directories for download path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/parquet_files/train")
args = parser.parse_args()
train_path = args.data_dir + "/"

# dictionary of data paths for train data
train_dict = {
    "base": hcu.DataLoader.get_dataframe(train_path + "train_base.parquet"),
    "depth_0": [
        hcu.DataLoader.get_dataframe(train_path + "train_static_cb_0.parquet"),
        hcu.DataLoader.get_dataframe(train_path + "train_static_0_*.parquet", chunked = True),
    ],
    "depth_1": [
        hcu.DataLoader.get_dataframe(train_path + "train_applprev_1_*.parquet","prev1", 1, True),
        hcu.DataLoader.get_dataframe(train_path + "train_tax_registry_a_1.parquet","tra", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_tax_registry_b_1.parquet","trb", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_tax_registry_c_1.parquet","trc", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_credit_bureau_a_1_*.parquet","cba1", 1, True),
        hcu.DataLoader.get_dataframe(train_path + "train_credit_bureau_b_1.parquet","cbb1", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_other_1.parquet","oth", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_person_1.parquet","per1", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_deposit_1.parquet","dep", 1),
        hcu.DataLoader.get_dataframe(train_path + "train_debitcard_1.parquet","deb", 1),
    ],
    "depth_2": [
        hcu.DataLoader.get_dataframe(train_path + "train_applprev_2.parquet","prev2", 2),
        hcu.DataLoader.get_dataframe(train_path + "train_person_2.parquet","per2", 2),
        hcu.DataLoader.get_dataframe(train_path + "train_credit_bureau_a_2_*.parquet","cba2", 2, True),
        hcu.DataLoader.get_dataframe(train_path + "train_credit_bureau_b_2.parquet","cbb2", 2),
    ]
}

# clear memory
hcu.MemoryOptimizer.CleanMemory()


"""Data Preprocessing and Cleaning"""
train_df = hcu.DataPreprocessor.preprocess_train(train_dict, threshold = 0.95)
hcu.MemoryOptimizer.CleanMemory()

del train_dict
hcu.MemoryOptimizer.CleanMemory()

gc.collect()

train_df.info()

# save the train_df column names in a pkl file
with open('train_cols_b4_featengg_v8.pkl', 'wb') as f:
  pickle.dump(train_df.columns.to_list(), f)

# create a dictionary of categories for each categorical variable
categories_dict = {}
for col in train_df.select_dtypes(include=['category']).columns:
    categories_dict[col] = train_df[col].cat.categories

# save the dictionary into a file for use in test file
with open('categories_v8.pkl', 'wb') as f:
    pickle.dump(categories_dict, f)


"""Feature Engineering"""
train_df = hcu.FeatureEngineer.preparedf(train_df, na_threshold = 0.95)
hcu.MemoryOptimizer.CleanMemory()

train_df.info()

# save the train_df column names in a pkl file
with open('final_train_cols_v8.pkl', 'wb') as f:
  pickle.dump(train_df.columns.to_list(), f)

gc.collect()


"""Exploratory Data Analysis"""

# describe() first 5 columns of train_df
train_df.iloc[:, :4].describe().T

# value counts of target to determine distribution
train_df['target'].value_counts()

# check the percentage of defaults
print("Percentage of defaults: ", (train_df['target'].sum()/train_df.shape[0] * 100).round(4), "%", sep="")

"""Data Scaling"""

# initializing the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# seperating the numerical columns to perform MinMax scaling
num_cols = train_df.select_dtypes(include = np.number).columns.to_list()
num_cols.remove("target")
num_cols.remove("case_id")
num_cols.remove("WEEK_NUM")

# fit and transform the train
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])

# save the MinMaxScaler in a pkl file
with open('mmscaler_v8.pkl', 'wb') as f:
  pickle.dump(scaler, f)

# check train_df sample
train_df.sample(10)

train_df.info()

# save train_df in .parquet file
train_df.to_parquet("data/processed/train.parquet")

# save category columns in pkl file
cat_cols = train_df.select_dtypes(include = 'category').columns.to_list()
with open('train_cat_cols.pkl', 'wb') as f:
  pickle.dump(cat_cols, f)

# garbage collection
hcu.MemoryOptimizer.CleanMemory()
gc.collect()

