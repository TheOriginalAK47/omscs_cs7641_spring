import sklearn
import numpy as np
import pandas as pd
import sys

import seaborn as sns
import matplotlib.pyplot as plt

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
						'room_type_private', 'room_type_entire', 'room_type_shared', 'expensive', 'number_of_reviews', 'reviews_per_month', \
								'calculated_host_listings_count', 'price']]
	return airbnb_df

def exploratory_analysis(df):
	plt.figure()
	viz = sns.kdeplot(df[df['price'] < 1000]['price'], shade=True)
	viz.set(xlabel='NYC Airbnb Rental Price ($)', ylabel='Frequency (out of 1.0)', title='Kernel Density Estimate plot of Airbnb Rental Prices')
	plt.savefig("airbnb_price_kde_plot.png")
	plt.figure()
	airbnb_top_quartile_price_thresh = df['price'].quantile(0.75)
	df['expensive'] = df['price'].apply(lambda price: 1 if price > airbnb_top_quartile_price_thresh else 0)
	bar = sns.countplot(x="expensive", data=df)
	bar.set(xlabel="NYC Airbnb Labels (1 expensive, 0 inexpensive)", ylabel="Label Count", title="NYC Airbnb Rental Label Counts")
	plt.savefig("airbnb_price_label_counts.png")
	plt.figure()
	ax = sns.heatmap(df)
	ax.set(title="NYC Airbnb features pair-wise correlation")
	plt.savefig("airbnb_features_corr_heatmap.png")

airbnb_raw_data_path = sys.argv[1]

airbnb_raw_df = pd.read_csv(airbnb_raw_data_path)

exploratory_analysis(airbnb_features_df)

airbnb_features_df = derive_features_and_perform_one_hot_encoding(airbnb_raw_df)

airbnb_features_df.to_csv("airbnb_features_data.csv", index=False)