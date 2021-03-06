README for OMSCS_7641 ML Project 1.

Github Link: https://github.com/TheOriginalAK47/omscs_cs7641_spring
Branch: Master

I've separated the code, data, and plots for each of the two datasets into separate directories: airbnb/ and recidivism/. I've also included the base dataset and the model datasets that result from the feature prep scripts I've written as well for ease of re-creation. Both directories have a plots/ and models/ sub-directory where the experiment results and final trained model files are stored. I've also created a bash wrapper script in each directory that should execute
the data preparation and model construction process from end to end which are named respectively:
recidivism/build_models_for_recidivism_prediction_task.sh
airbnb/build_models_for_airbnb_price_prediction_task.sh

These can simply be invoked by:
`sh build_models_for_recidivism_prediction_task.sh`
`sh build_models_for_airbnb_price_prediction_task.sh`

Within these scripts I'm passing parameters where the output pre-processed data files are stored for the first script, and then the two input file paths, plots directory path, model files path, label column name, and problem name to the model-specific code which can be observed in the shell script code.

The Neural-Network and K-Nearest Neighbors code takes significantly longer than other model files but should run to completion successfully in a timely manner, even on a basic desktop setup.

If you have any questions, feel free to contact me at akogler3@gatech.edu.
