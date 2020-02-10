import sklearn
import numpy as np
import pandas as pd
import sys

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

def ethnicity_mapping_func(raw_ethnicity_str):
	if (raw_ethnicity_str == 'White - Non-Hispanic' or raw_ethnicity_str == 'White - Hispanic'):
		return 'White'
	elif (raw_ethnicity_str == 'Black - Non-Hispanic' or raw_ethnicity_str == 'Black - Hispanic'):
		return 'Black'
	elif (raw_ethnicity_str == 'Asian or Pacific Islander - Non-Hispanic' or raw_ethnicity_str == 'A/PI - NH' or \
		  raw_ethnicity_str == 'Asian or Pacific Islander - Hispanic' or raw_ethnicity_str == 'A/PI - H'):
		return 'A/PI'
	elif (raw_ethnicity_str == 'American Indian or Alaska Native - Non-Hispanic' or raw_ethnicity_str == 'AI/AN - NH' or \
		  raw_ethnicity_str == 'American Indian or Alaska Native - Hispanic' or raw_ethnicity_str == 'AI/AN - H'):
		return "AI/AN"
	else:
		return "Unknown"

def recidivism_data_cleanup(df):
	df['ethnicity_white'] = df['Race - Ethnicity'].apply(lambda x: 1 if x == 'White - Non-Hispanic' \
																						or x == 'White - Hispanic' \
																						else 0)
	df['ethnicity_black'] = df['Race - Ethnicity'].apply(lambda x: 1 if x == 'Black - Non-Hispanic' \
																						or x == 'Black - Hispanic' \
																						else 0)
	df['ethnicity_aipi'] = df['Race - Ethnicity'].apply(lambda x: 1 if x == 'Asian or Pacific Islander - Non-Hispanic' \
																						or x == 'A/PI - NH' \
																						or x == 'Asian or Pacific Islander - Hispanic' \
																						or x == 'A/PI - H' \
																						else 0)
	df['ethnicity_aian'] = df['Race - Ethnicity'].apply(lambda x: 1 if x == 'American Indian or Alaska Native - Non-Hispanic' \
																						or x == 'AI/AN - NH' \
																						or x == 'American Indian or Alaska Native - Hispanic' \
																						or x == 'AI/AN - H' \
																						else 0)
	df['ethnicity_unknown'] = df['Race - Ethnicity'].apply(lambda x: 1 if x == 'Unknown' \
																						or x == 'AI/AN - NH' \
																						or x == 'Unknown - Non-Hispanic' \
																						or x == 'Unk - NH ' \
																						else 0)
	df['ethnicity_cleaned'] = df['Race - Ethnicity'].apply(ethnicity_mapping_func)
	df['region_5jd'] = df['Region Code'].apply(lambda r: 1 if r == '5JD' else 0)
	df['region_6jd'] = df['Region Code'].apply(lambda r: 1 if r == '6JD' else 0)
	df['region_1jd'] = df['Region Code'].apply(lambda r: 1 if r == '1JD' else 0)
	df['region_3jd'] = df['Region Code'].apply(lambda r: 1 if r == '3JD' else 0)
	df['region_2jd'] = df['Region Code'].apply(lambda r: 1 if r == '2JD' else 0)
	df['region_8jd'] = df['Region Code'].apply(lambda r: 1 if r == '8JD' else 0)
	df['region_4jd'] = df['Region Code'].apply(lambda r: 1 if r == '4JD' else 0)
	df['region_7jd'] = df['Region Code'].apply(lambda r: 1 if r == '7JD' else 0)
	df['offender_male'] = df['Sex'].apply(lambda s: 1 if s == 'Male' else 0)
	df['offender_female'] = df['Sex'].apply(lambda s: 1 if s == 'Female' else 0)
	df['offender_unknown'] = df['Sex'].apply(lambda s: 1 if s == 'Unknown' else 0)
	df['offense_public_order'] = df['Convicting Offense Type'].apply(lambda o: 1 if o == 'Public Order' else 0)
	df['offense_drug'] = df['Convicting Offense Type'].apply(lambda o: 1 if o == 'Drug' else 0)
	df['offense_property'] = df['Convicting Offense Type'].apply(lambda o: 1 if o == 'Property' else 0)
	df['offense_violent'] = df['Convicting Offense Type'].apply(lambda o: 1 if o == 'Violent' else 0)
	df['offense_other'] = df['Convicting Offense Type'].apply(lambda o: 1 if o == 'Other' else 0)
	df['offender_supervision_high'] = df['Level of Supervision'].apply(lambda s: 1 if s == 'Intensive' or s == 'High Normal' else 0)
	df['offender_supervision_low'] = df['Level of Supervision'].apply(lambda s: 1 if s != 'Intensive' and s != 'High Normal' else 0)
	df['recidivism_target'] = df['Part of Target Population'].apply(lambda r: 0 if r == 'No' else 1)
	df['recidivism_flag'] = df['Recidivism Type'].apply(lambda r: 0 if r == 'No Recidivism' else 1)
	return df

def recidivism_exploratory_analysis(df):
	plt.figure()
	bar = sns.countplot(x="Recidivism Type", data=df)
	bar.set(xlabel="Recidivism Outcome Type", ylabel="Outcome Count", title="Recidivism Outcome Frequency Plot")
	plt.savefig("recidivism_outcome_freq_plot.png")
	plt.figure()
	bar = sns.countplot(x="ethnicity_cleaned", data=df)
	bar.set(xlabel="Offender Race/Ethnicity", ylabel="Ethnicity Count", title="Offender Ethnicity Frequency Plot")
	plt.savefig("offender_ethnicity_freq_plot.png")
	plt.figure()
	bar = sns.countplot(x="Sex", data=df)
	bar.set(xlabel="Offender Sex", ylabel="Gender Count", title="Offender Gender Frequency Plot")
	plt.savefig("offender_gender_freq_plot.png")

def create_oversampled_dataset(features, labels):
	sm = SMOTE(sampling_strategy='minority')
	X_res, y_res = sm.fit_resample(features, labels)
	return X_res, y_res

recidivism_raw_data_path = sys.argv[1]

training_data_output_path = sys.argv[2]

test_data_output_path = sys.argv[3]

recidivism_raw_df = pd.read_csv(recidivism_raw_data_path)

recidivism_features_df = recidivism_data_cleanup(recidivism_raw_df)

recidivism_exploratory_analysis(recidivism_features_df)

#print(recidivism_features_df.columns)

feature_cols = ['ethnicity_white', 'ethnicity_black', 'ethnicity_aipi', 'ethnicity_aian', \
				'ethnicity_unknown', 'offender_male', 'offender_female', 'offender_unknown', \
				'offense_public_order', 'offense_drug', 'offense_property', \
				'offense_violent', 'offense_other', 'offender_supervision_high', \
				'offender_supervision_low', 'recidivism_target']

train, test = train_test_split(recidivism_features_df, test_size=0.25, train_size=0.75)

synthesized_feats, synthesized_labels = create_oversampled_dataset(train[feature_cols], train['recidivism_flag'])

recidivism_dataset = pd.concat([synthesized_feats, synthesized_labels], axis=1).sample(frac=0.5)

recidivism_dataset.to_csv(training_data_output_path, index=False)

feature_cols.append('recidivism_flag')

test[feature_cols].to_csv(test_data_output_path, index=False)