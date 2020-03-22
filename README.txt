# Assignment 3

Like before all of the code is written in Python and store output into respective data and plot folders. For the Airbnb dataset, the raw data is contained in airbnb/data/raw with both the training and test sets there. To execute the base clustering algorithms you can run the following bash commands:

python3 k_means_clustering.py data/raw/airbnb_training_data.csv data/raw/airbnb_test_data.csv plots/ airbnb_clustering

python3 em_clustering.py data/raw/airbnb_training_data.csv data/raw/airbnb_test_data.csv plots/ airbnb_clustering

python3 k_means_clustering.py data/raw/recidivism_training_data.csv data/raw/recidivism_test_data.csv plots/ recidivism_clustering

python3 em_clustering.py data/raw/recidivism_training_data.csv data/raw/recidivism_test_data.csv plots/ airbnb_clustering

Each of the different decomposition methods have separate wrapper scripts in each use-cases' directory, those different scripts being:

factor_analysis_runner.py
ica_runner.py
pca_runner.py
random_proj_runner.py

