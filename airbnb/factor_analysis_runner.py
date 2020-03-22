from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import MinMaxScaler

import sys
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

airbnb_train_data_input_file_path = sys.argv[1]

airbnb_test_data_input_file_path = sys.argv[2]

data_dir = sys.argv[3]

problem_name = sys.argv[4]

num_components = int(sys.argv[5])

airbnb_train_df = pd.read_csv(airbnb_train_data_input_file_path)

airbnb_test_df = pd.read_csv(airbnb_test_data_input_file_path)

airbnb_df = pd.concat([airbnb_train_df, airbnb_test_df])

transformer = FactorAnalysis(n_components=num_components, random_state=0)
airbnb_transformed_data = transformer.fit_transform(airbnb_df.to_numpy())

transformed_airbnb_data_df = pd.DataFrame(airbnb_transformed_data)

transformed_airbnb_data_df.to_csv(data_dir + "factor_analysis/airbnb_data_factor_analysis_transformed.csv", index=False)