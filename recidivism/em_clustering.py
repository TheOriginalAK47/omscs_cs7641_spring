from sklearn.mixture import GaussianMixture

import sys
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

airbnb_train_data_input_file_path = sys.argv[1]

airbnb_test_data_input_file_path = sys.argv[2]

plot_dir = sys.argv[3]

problem_name = sys.argv[4]

airbnb_train_df = pd.read_csv(airbnb_train_data_input_file_path)

airbnb_test_df = pd.read_csv(airbnb_test_data_input_file_path)

airbnb_df = pd.concat([airbnb_train_df, airbnb_test_df])

best_num_components = 2
best_bic_val = float("inf")
for num_components in range(2, 11):
	em_model = GaussianMixture(n_components=num_components)
	em_model.fit(airbnb_df)
	if (em_model.bic(airbnb_df) < best_bic_val):
		best_num_components = num_components
		best_bic_val = em_model.bic(airbnb_df)

opt_components_cluster_model = GaussianMixture(n_components=best_num_components)

y_km = opt_components_cluster_model.fit_predict(airbnb_df)

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'w', 'orange']

cluster_counts = []
for cluster_id in range(0, best_num_components - 1):
	cluster_count = y_km[y_km == cluster_id].shape[0]
	cluster_counts.append(cluster_count)


cluster_ids = np.arange(0, best_num_components - 1)
cluster_str_ids = [str(i) for i in cluster_ids]

plt.bar(cluster_ids, cluster_counts, align='center', alpha=0.5)
plt.xticks(cluster_ids, cluster_str_ids)
plt.xlabel('Cluster ID #')
plt.ylabel('# of points in cluster')
plt.title('Count of points assigned to cluster by ID')
plt.grid()

plt.savefig(plot_dir + problem_name + "_em_clusters_point_counts.png")