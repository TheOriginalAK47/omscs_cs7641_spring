from sklearn.random_projection import GaussianRandomProjection

import sys
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

airbnb_train_data_input_file_path = sys.argv[1]

airbnb_test_data_input_file_path = sys.argv[2]

plot_dir = sys.argv[3]

data_dir = sys.argv[4]

problem_name = sys.argv[5]

num_iterations = sys.argv[6]

airbnb_train_df = pd.read_csv(airbnb_train_data_input_file_path)

airbnb_test_df = pd.read_csv(airbnb_test_data_input_file_path)

airbnb_df = pd.concat([airbnb_train_df, airbnb_test_df])

rng = np.random.RandomState(42)
X = rng.rand(100, 10000)
transformer = GaussianRandomProjection(random_state=rng)
X_new = transformer.fit_transform(airbnb_df)

for i in range(0, num_iterations - 1):


transformed_data = pd.DataFrame(pca.fit_transform(data_rescaled)[:, :7])

transformed_data.to_csv(data_dir + "pca/airbnb_data_pca_transformed.csv", index=False)