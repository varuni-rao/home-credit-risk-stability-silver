"""Utility functions for loading and processing data."""
import glob # locating chunks
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt # datetime features manipulation
import polars as pl # parquet file I/O (e.g. pl.read_parquet)
pl.enable_string_cache()

# display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# library for pickle
import pickle
import ctypes # module that allows a way to call functions from shared libraries
from os import getpid
from psutil import Process

# library for garbage collection
import gc  # since the data is huge here, regularly cleaning up will free up memory

# library to catch and ignore warnings
import warnings
warnings.filterwarnings("ignore")


"""Loading data onto dataframes"""

class DataLoader:
  # this class includes the functions that will be used intially for loading the data into dataframe

    @staticmethod
    def fix_dtypes(df):
      # this function takes in dataframe read from the parquet file
      # and cset the dtype of the columns based on the ending character of the column name

      for column in df.columns:

        # type cast for case_id, WEEK_NUM, num_group1 and num_group2 as polars int64
        if column in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
           df = df.with_columns(pl.col(column).cast(pl.Int64, strict=False))

        # type cast for column names ending with a P or A as polars float64
        elif column[-1] in ("P", "A"):
           df = df.with_columns(pl.col(column).cast(pl.Float64, strict = False))

        # type cast for column names ending with M as polars categorical features
        elif column[-1] in ("M"):
           df = df.with_columns(pl.col(column).cast(pl.Categorical, strict = False))

        # type cast for column names ending with a D or date_decision as polars date type
        elif column in ["date_decision"] or column[-1] in ("D"):
           df = df.with_columns(pl.col(column).cast(pl.Date, strict = False))

      # return polars dataframe
      return df

    @staticmethod
    def get_aggregate(df, suffix):
      # Aggregator functions applied to input dataframe
      print("dataframe shape before aggregation: ", df.shape)
      all_agg = []; # empty list to add aggregations
      df_cols = df.columns; # columns in dataframe
        
      all_agg.extend([pl.col(col).max().alias(f"max_{col}") for col in df_cols if col[-1] in ("P", "A")])
      all_agg.extend([pl.col(col).last().alias(f"last_{col}") for col in df_cols if col[-1] in ("P", "A")])
      all_agg.extend([pl.col(col).mean().alias(f"mean_{col}") for col in df_cols if col[-1] in ("P", "A")]);
      all_agg.extend([pl.col(col).median().alias(f"median_{col}") for col in df_cols if col[-1] in ("P", "A")]);
      all_agg.extend([pl.col(col).var().alias(f"var_{col}") for col in df_cols if col[-1] in ("P", "A")]);

      all_agg.extend([pl.col(col).max().alias(f"max_{col}") for col in df_cols if col[-1] in ("D")])
      all_agg.extend([pl.col(col).last().alias(f"last_{col}") for col in df_cols if col[-1] in ("D")])
      all_agg.extend([pl.col(col).mean().alias(f"mean_{col}") for col in df_cols if col[-1] in ("D")]);
      all_agg.extend([pl.col(col).median().alias(f"median_{col}") for col in df_cols if col[-1] in ("D")]);
    
      all_agg.extend([pl.col(col).max().alias(f"max_{col}") for col in df_cols if col[-1] in ("M")])
      all_agg.extend([pl.col(col).last().alias(f"last_{col}") for col in df_cols if col[-1] in ("M")])
        
      all_agg.extend([pl.col(col).max().alias(f"max_{col}") for col in df_cols if col[-1] in ("T", "L")])
      all_agg.extend([pl.col(col).last().alias(f"last_{col}") for col in df_cols if col[-1] in ("T", "L")])
        
      all_agg.extend([pl.col(col).max().alias(f"max_{col}_{suffix}") for col in df_cols if "num_group" in col])

      df_new = df.sort(by ="num_group1").group_by("case_id").agg(all_agg);

      print("dataframe shape after aggregation: ", df_new.shape)

      return df_new.unique(subset=["case_id"])
    
    @staticmethod
    def get_dataframe(path, suffix=None, depth = None, chunked = False):
      
      if chunked:
        chunks = []
        
        file_paths = glob.glob(path)
        print("Chunk file paths: ", file_paths)
        for path in file_paths:
          df = pl.read_parquet(path).pipe(DataLoader.fix_dtypes)

          if depth in [1,2]:
            df = DataLoader.get_aggregate(df, suffix)
            print("Aggregation done....")

          chunks.append(df)
          print("Chunk collected.....")

        df = pl.concat(chunks, how="vertical_relaxed")
        print("Chunks concatenated....")

        df = df.unique(subset=["case_id"])
        print("Unique case_ids selected....")

      else:
        print("Filepath:", path)
        df = pl.read_parquet(path).pipe(DataLoader.fix_dtypes)

        if depth in [1,2]:
          df = DataLoader.get_aggregate(df, suffix)

      df = MemoryOptimizer.reduce_memory_usage(df.to_pandas())

      print("Dataframe shape:", df.shape)
      print(" ")

      # return pandas dataframe
      return df

"""Memory optimization functions"""
class MemoryOptimizer:

    @staticmethod
    def reduce_memory_usage(df):

      print("Optimizing memory usage.......")

      # display the memory usage before optimization
      start_mem_train = df.memory_usage().sum() / 1024 ** 2;
      print('Memory usage of train dataframe before optimization is {:.2f} MB'.format(start_mem_train))

      # iterate through all columns
      for col in df.columns:
        col_dtype = df[col].dtype # get column dtype

        # treating integer dtypes
        if "int" in str(col_dtype):
          c_min = df[col].min()
          c_max = df[col].max()

          # treating int types
          if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
              df[col] = df[col].astype(np.int8)

          elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
              df[col] = df[col].astype(np.uint8)

          elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
              df[col] = df[col].astype(np.int16)

          elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
              df[col] = df[col].astype(np.uint16)

          elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
              df[col] = df[col].astype(np.int32)

          elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
              df[col] = df[col].astype(np.uint32)

          elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
              df[col] = df[col].astype(np.int64)

          elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
              df[col] = df[col].astype(np.uint64)


        # treating float64 types
        elif "float" in str(col_dtype):
          c_min = df[col].min()
          c_max = df[col].max()
          
          if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
              df[col] = df[col].astype(np.float16)

          elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
              df[col] = df[col].astype(np.float32)

          elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
              df[col] = df[col].astype(np.float64)

        # treating object types
        elif col_dtype == 'object':
          df[col] = df[col].astype('category')

      # collect garbage to free memory
      gc.collect()

      # determine memory usage after optimization
      end_mem_train = df.memory_usage().sum() / 1024 ** 2
      print('Memory usage after optimization is: {:.3f} MB'.format(end_mem_train))
      print('Decreased by {:.2f}%'.format(100 * (start_mem_train - end_mem_train) / start_mem_train))

      # return the dataframe
      return df

    @staticmethod
    def CleanMemory():
      # this code is borrowed from https://www.kaggle.com/code/ravi20076/homecredit-starter-inference-v1
      # This method cleans the memory off unused objects and displays the cleaned state RAM usage
      # first import the libraries

      libc = ctypes.CDLL("libc.so.6") # Load the 'libc.so.6' shared library
      gc.collect(); # helps manage memory by identifying and cleaning up unused objects

      # Call the 'malloc_trim' function to release memory
      # The argument 0 means that all free memory pages are trimmed
      libc.malloc_trim(0);

      pid = getpid(); # retrieves process ID of the current process
      py = Process(pid); # Creates a Process object associated with the specified PID

      # (index 0) represents the resident set size (RSS) in bytes
      # The division by 2. ** 30 converts the RSS from bytes to gigabytes (GB)
      memory_use = py.memory_info()[0] / 2. ** 30;

      return f"RAM usage = {memory_use :.4} GB";


""" Data preprocessing functions like filtering columns, handling dates, creating new categories for categorical features, etc."""

class DataPreprocessor:
  @staticmethod
  def filtercols(df, threshold, freq_threshold):
      print("Dataframe shape before filtering:", df.shape)

      # initiating empty list to store drop columns
      drop_cols = []

      #drop columns that have more than 90% missing data
      for col in df.columns:

        if col not in ["target", "case_id", "WEEK_NUM"]:
           if df[col].isnull().mean() > threshold:
              drop_cols.append(col)

      # drop columns that have cardinality of 1 or more than 100 for categorical columns
      cat_col = df.select_dtypes(['object', 'category', 'bool']).columns
      for col in cat_col:
        if (col not in ["target", "case_id", "WEEK_NUM"]):
          freq = df[col].nunique()

          if (freq == 1) | (freq > freq_threshold):
            drop_cols.append(col)

      # drop columns that have been appended to drop_cols
      df.drop(drop_cols, axis = 1, inplace = True)

      print("Dataframe shape after filtering:", df.shape)
      print("")

      return df

  @staticmethod
  def combine(df_dict):
      print("Combining the dataframes in to a single dataframe.....")
      joined_df = df_dict['base']
      for df_name, df_list in df_dict.items():
        if df_name != 'base':
          for df in df_list:
            joined_df = joined_df.merge(df, on='case_id', how='left')

      print("Combined dataframe shape:", joined_df.shape)
      print("")
      return joined_df

  @staticmethod
  def handle_dates(df):
      print("Handling datetime datatype.....")
      # borrowed from https://www.kaggle.com/code/dksdms4/lb-0-565-improved-baseline-notebook
      for col in df.columns:
          if col[-1] in ("D"):
            # subtracting all dates from decision date which is considered as the baseline date
            # and changing the columns to number of days
            df[col] = (df['date_decision'] - df[col]).dt.days

      print("All datetime datatype treated !\n")
      return df

  @staticmethod
  # create new Categorical dtype for each category by adding 'Unknown' to the list of unique categories of the column
  def createnewcatdtype(train, cat_cols):
    print("Creating new UNKNOWN category in all categorical features of train data")
    for col in cat_cols:
      new_categories = train[col].cat.categories.to_list() + ["UNKNOWN"]
      new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
      train[col] = train[col].astype(new_dtype)

    print("Categorical features treatment complete !\n")

    return train

  @staticmethod
  # this will set the new categorical dtype to train and test
  def setnewcatdtype(test, cat_filename): #cat_cols, 
    print("Creating new UNKNOWN category in all categorical features of test data")
    with open(cat_filename, 'rb') as f:
        categories_dict = pickle.load(f)
        
    cat_cols = list(categories_dict.keys())

    for col in cat_cols:
      train_categories = set(categories_dict[col])
      test[col] = test[col].astype('category')
      test_categories = set(test[col].cat.categories)
      new_categories = test_categories - train_categories

      # Set the categories for the Categorical column
      new_dtype = pd.CategoricalDtype(categories=train_categories, ordered=True)
      test[col] = test[col].astype(new_dtype)

      # Replace new categories with "Unknown" in the test DataFrame
      if len(new_categories) > 0:
        test.loc[test[col].isin(new_categories), col] = "UNKNOWN"

    return test

  @staticmethod
  def preprocess_train(df_dict, threshold = 0.7, freq_threshold = 200):
      # first filter columns and drop the ones that meet filter criterion
      for df_name, df_list in df_dict.items():
        if df_name == 'base':
          df = DataPreprocessor.filtercols(df_list, threshold, freq_threshold)
        else:
          for df in df_list:
            df = DataPreprocessor.filtercols(df, threshold, freq_threshold)

      # join dataframes into one train dataframe
      joined_df = DataPreprocessor.combine(df_dict)

      # handle dates for datetime columns
      joined_df = DataPreprocessor.handle_dates(joined_df)
      joined_df = MemoryOptimizer.reduce_memory_usage(joined_df)

      # seperate categorical columns and add new categories
      cat_cols = joined_df.select_dtypes(include = ['category']).columns.tolist()
      joined_df = DataPreprocessor.createnewcatdtype(joined_df, cat_cols)

      print("Dataframe shape of the preprocessed dataframe:", joined_df.shape)

      return joined_df

  @staticmethod
  def preprocess_test(df_dict, col_filename, cat_filename=None):
      # join dataframes into one train dataframe
      joined_df = DataPreprocessor.combine(df_dict)

      # select same features as in train
      # load train_df_columns.pkl
      with open(col_filename, 'rb') as f:
        df_columns = pickle.load(f)
      df_columns.remove('target')

      joined_df = joined_df[df_columns]

      # handle dates for datetime columns
      joined_df = DataPreprocessor.handle_dates(joined_df)
      joined_df = MemoryOptimizer.reduce_memory_usage(joined_df)

      # seperate categorical columns and add new categories
      #cat_cols = joined_df.select_dtypes(include = 'category').columns.tolist()
      if cat_filename != None:
          joined_df = DataPreprocessor.setnewcatdtype(joined_df, cat_filename) #cat_cols,

      print("Dataframe shape of the preprocessed dataframe:", joined_df.shape)
      print("Dataframe info:\n", joined_df.info())

      return joined_df

""" Feature engineering functions like merging columns, creating new features by aggregating existing features, etc."""
class FeatureEngineer:

  @staticmethod
  def mergecolumns(df, col_list, aggr, newfeatname):
    if aggr == 'min':
      df[newfeatname] = df[col_list].min(axis=1, skipna = True)
    elif aggr == 'max':
      df[newfeatname] = df[col_list].max(axis=1, skipna = True)

    print("Merged {} to create a new feature {}".format(col_list, newfeatname))

    df.drop(col_list, axis = 1, inplace = True)
    print("")

    return df

  @staticmethod
  def group_columns_by_correlation(df, threshold=0.8):
    # code borrowed from https://www.kaggle.com/code/majiaqi111/metric-s-trick-home-credit-lgb-cat-ensemble
    # obtain correlation matrix
    correlation_matrix = df.corr() 

    groups = [] # initiate group list
    remaining_cols = list(df.columns) #initiate remaining columns list
    remaining_cols.remove('case_id')
    remaining_cols.remove('target')
    remaining_cols.remove('WEEK_NUM')
    
    # condition true while there are columns remaining in remaining_cols list
    while remaining_cols:
        col = remaining_cols.pop(0) #consider first column in remaining columns
        group = [col] # add to the group
        correlated_cols = [col] #initiate correlated column list with first column in remaining group
        
        # for every column in remaining_col
        for c in remaining_cols:
            # if corr() value is greater than threshold -> highly correlated
            if correlation_matrix.loc[col, c] >= threshold:
                group.append(c) #add column to group
                correlated_cols.append(c) # add column to correlated columns group
        groups.append(group) # add group to list of correlated columns
     #   print("Correlated columns are:", group)
        remaining_cols = [c for c in remaining_cols if c not in correlated_cols] # remove coreelated columns from remaining columns
    
    return groups

  @staticmethod
  def reduce_group(df, grps):
    use = []
    for grp_list in grps:
        mx = 0; # initiate number of nunique to 0
        feat = grp_list[0] # initiate feature to first feature
        for col in grp_list:
            n = df[col].nunique()
            if n>mx:
                mx = n
                feat = col
        use.append(feat)
    print('Include these cols in dataframe:',use)
    return use  

  @staticmethod
  def preparedf(df, corr_threshold = 0.8, na_threshold=0.7):
    # drop cols not being used by the model or creating issues for modeling
    df.drop(['MONTH', 'date_decision'], axis = 1, inplace = True) 
    
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='category').columns.to_list()
    
    corr_groups = FeatureEngineer.group_columns_by_correlation(df[num_cols], corr_threshold)
    use_num_cols = FeatureEngineer.reduce_group(df, corr_groups)
    final_use_cols = ['case_id', 'target', 'WEEK_NUM'] + use_num_cols + cat_cols
    
    final_df = df[final_use_cols]
    
    drop_cols = []
    for col in final_df.columns:
        if final_df[col].isnull().mean() > na_threshold:
            drop_cols.append(col)
            
    final_df.drop(drop_cols, axis = 1, inplace = True)

    print("Final Dataframe shape is:", final_df.shape)

    return final_df





