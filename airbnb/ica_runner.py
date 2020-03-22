from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis

import sys
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

airbnb_train_data_input_file_path = sys.argv[1]

airbnb_test_data_input_file_path = sys.argv[2]

data_dir = sys.argv[3]

problem_name = sys.argv[4]

#num_components = int(sys.argv[5])

airbnb_train_df = pd.read_csv(airbnb_train_data_input_file_path)

airbnb_test_df = pd.read_csv(airbnb_test_data_input_file_path)

airbnb_df = pd.concat([airbnb_train_df, airbnb_test_df])

best_kurtosis_sum = 0
best_num_components = 0
for num_component in range(1, 13):
	x_transformed = FastICA(n_components=num_component, whiten=True).fit_transform(airbnb_df.to_numpy())
	kurtosis_vals = kurtosis(x_transformed)
	kurtosis_vals = kurtosis_vals**2
	mean_kurtosis_vals = [np.mean(k) for k in kurtosis_vals]
	summed_kurtosis_vals = np.sum(mean_kurtosis_vals)
	print("# of components: " + str(num_component))
	print("Summed Kurtosis: " + str(summed_kurtosis_vals))
	if (summed_kurtosis_vals > best_kurtosis_sum):
		best_kurtosis_sum = summed_kurtosis_vals
		best_num_components = num_component

x_transformed = FastICA(n_components=best_num_components, whiten=True).fit_transform(airbnb_df.to_numpy())

transformed_airbnb_data_df = pd.DataFrame(x_transformed)

transformed_airbnb_data_df.to_csv(data_dir + "ica/airbnb_data_ica_transformed.csv", index=False)