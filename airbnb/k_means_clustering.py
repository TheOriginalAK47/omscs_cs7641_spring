from sklearn.cluster import KMeans
import sys
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

print('foo')

airbnb_train_data_input_file_path = sys.argv[1]

airbnb_test_data_input_file_path = sys.argv[2]

plot_dir = sys.argv[3]

problem_name = sys.argv[4]

if (airbnb_test_data_input_file_path != 'foo'):
	airbnb_train_df = pd.read_csv(airbnb_train_data_input_file_path)
	airbnb_test_df = pd.read_csv(airbnb_test_data_input_file_path)
	airbnb_df = pd.concat([airbnb_train_df, airbnb_test_df])
else:
	airbnb_df = pd.read_csv(airbnb_train_data_input_file_path)

best_k_val = 2
best_inertia_val = float("inf")
for num_cluster in range(2, 11):
	k_means_model = KMeans(n_clusters=num_cluster, n_jobs=-1)
	k_means_model.fit(airbnb_df)
	if (k_means_model.inertia_ < best_inertia_val):
		best_k_val = num_cluster
		best_inertia_val = k_means_model.inertia_

opt_k_val_cluster_model = KMeans(n_clusters=best_k_val, n_jobs=-1)

y_km = opt_k_val_cluster_model.fit_predict(airbnb_df)

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'w', 'orange']

cluster_counts = []
for cluster_id in range(0, best_k_val - 1):
	cluster_count = y_km[y_km == cluster_id].shape[0]
	cluster_counts.append(cluster_count)


cluster_ids = np.arange(0, best_k_val - 1)
cluster_str_ids = [str(i) for i in cluster_ids]

plt.bar(cluster_ids, cluster_counts, align='center', alpha=0.5)
plt.xticks(cluster_ids, cluster_str_ids)
plt.xlabel('Cluster ID #')
plt.ylabel('# of points in cluster')
plt.title('Count of points assigned to cluster by ID')
plt.grid()

print(plot_dir + problem_name + "_k_means_clusters_point_counts.png")
plt.savefig(plot_dir + problem_name + "_k_means_clusters_point_counts.png")