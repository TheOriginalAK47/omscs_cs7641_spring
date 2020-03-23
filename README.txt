# Assignment 3

Git Repo: https://github.com/TheOriginalAK47/omscs_cs7641_spring/tree/assignment3

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

Invokation of these for example is as follows (with each varying on problem domain where the problem name as a parameter is changed depending on which directory you're in. For simplicity's sake I've included the generated dataset in the Git repo because the file-sizes allow):

python3 pca_runner.py data/raw/recidivism_training_data.csv data/raw/recidivism_test_data.csv plots/ data/ recidivism_clustering
python3 ica_runner.py data/raw/airbnb_training_data.csv data/raw/airbnb_test_data.csv data/ airbnb_clusterin
python3 factor_analysis_runner.py data/raw/airbnb_training_data.csv data/raw/airbnb_test_data.csv data/ airbnb_clustering 7
python3 random_proj_runner.py data/raw/airbnb_training_data.csv data/raw/airbnb_test_data.csv plots/ data/ airbnb_clustering 5

If you have any mis-understandings I would simply open each of these decomposition scripts as they're quite simple in nature. If you have any queries still, feel free to email me at akogler3@gatech.edu.