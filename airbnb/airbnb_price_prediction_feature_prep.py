import sklearn
import numpy as np
import pandas as pd
import sys

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

def derive_features_and_perform_one_hot_encoding(raw_df):
	raw_df['neighborhood_manhattan'] = raw_df['neighbourhood_group'].apply(lambda x: 1 if x == 'Manhattan' else 0)
	raw_df['neighborhood_brooklyn'] = raw_df['neighbourhood_group'].apply(lambda x: 1 if x == 'Brooklyn' else 0)
	raw_df['neighborhood_queens'] = raw_df['neighbourhood_group'].apply(lambda x: 1 if x == 'Queens' else 0)
	raw_df['neighborhood_staten_island'] = raw_df['neighbourhood_group'].apply(lambda x: 1 if x == 'Staten Island' else 0)
	raw_df['neighborhood_bronx'] = raw_df['neighbourhood_group'].apply(lambda x: 1 if x == 'Bronx' else 0)
	raw_df['room_type_private'] = raw_df['room_type'].apply(lambda x: 1 if x == 'Private room' else 0)
	raw_df['room_type_entire'] = raw_df['room_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
	raw_df['room_type_shared'] = raw_df['room_type'].apply(lambda x: 1 if x == 'Shared room' else 0)
	raw_df['price'] = raw_df['price'].astype(float)
	raw_df['reviews_per_month'] = raw_df['reviews_per_month'].fillna(0.0)
	airbnb_top_quartile_price_thresh = raw_df['price'].quantile(0.75)
	raw_df['expensive'] = raw_df['price'].apply(lambda price: 1 if price > airbnb_top_quartile_price_thresh else 0)
	airbnb_df = raw_df[['neighborhood_manhattan', 'neighborhood_brooklyn', 'neighborhood_bronx', 'neighborhood_queens', 'neighborhood_staten_island', \
						'room_type_private', 'room_type_entire', 'room_type_shared', 'number_of_reviews', 'reviews_per_month', \
								'calculated_host_listings_count', 'price', 'expensive']]
	return airbnb_df

def exploratory_analysis(df):
	plt.figure()
	viz = sns.kdeplot(df[df['price'] < 1000]['price'], shade=True)
	viz.set(xlabel='NYC Airbnb Rental Price ($)', ylabel='Frequency (out of 1.0)', title='Kernel Density Estimate plot of Airbnb Rental Prices')
	plt.savefig("airbnb_price_kde_plot.png")
	plt.figure()
	bar = sns.countplot(x="expensive", data=df)
	bar.set(xlabel="NYC Airbnb Labels (1 expensive, 0 inexpensive)", ylabel="Label Count", title="NYC Airbnb Rental Label Counts")
	plt.savefig("airbnb_price_label_counts.png")
	plt.figure()

def create_oversampled_dataset(features, labels):
	sm = SMOTE(sampling_strategy='minority')
	X_res, y_res = sm.fit_resample(features, labels)
	return X_res, y_res

airbnb_raw_data_path = sys.argv[1]

training_data_output_path = sys.argv[2]

test_data_output_path = sys.argv[3]

airbnb_raw_df = pd.read_csv(airbnb_raw_data_path)

airbnb_features_df = derive_features_and_perform_one_hot_encoding(airbnb_raw_df)

exploratory_analysis(airbnb_features_df)

feature_cols = [x for x in airbnb_features_df.columns if x != 'price' and x != 'expensive']

train, test = train_test_split(airbnb_features_df, test_size=0.25, train_size=0.75)

synthesized_feats, synthesized_labels = create_oversampled_dataset(train[feature_cols], train['expensive'])

airbnb_dataset = pd.concat([synthesized_feats, synthesized_labels], axis=1)

airbnb_dataset.to_csv(training_data_output_path, index=False)

test.to_csv(test_data_output_path, index=False)