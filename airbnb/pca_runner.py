from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import sys
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

airbnb_train_data_input_file_path = sys.argv[1]

airbnb_test_data_input_file_path = sys.argv[2]

plot_dir = sys.argv[3]

data_dir = sys.argv[4]

problem_name = sys.argv[5]

airbnb_train_df = pd.read_csv(airbnb_train_data_input_file_path)

airbnb_test_df = pd.read_csv(airbnb_test_data_input_file_path)

airbnb_df = pd.concat([airbnb_train_df, airbnb_test_df])

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(airbnb_df.to_numpy())

pca = PCA().fit(data_rescaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.title('Airbnb Dataset Explained Variance')
plt.savefig(plot_dir + problem_name + "_pca_variance_explained_by_num_components.png")

transformed_data = pd.DataFrame(pca.fit_transform(data_rescaled)[:, :7])

transformed_data.to_csv(data_dir + "pca/airbnb_data_pca_transformed.csv", index=False)