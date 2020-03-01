README for OMSCS_7641 ML Project 2.

Github Link: https://github.com/TheOriginalAK47/omscs_cs7641_spring
Branch: assignment2

The script to run all the code for the three different optimization problems of the Knapsack problem, Travelling Salesman Problem, and the Continuous Peaks problem, resides in the run_pipeline.sh script which can simply be invoked by:

`sh run_pipeline.sh`

This calls the optimization technique specific scripts in order which are executed using jython whereas the plotting scripts thereafter are run using python3. Plotting consists of the fitness functions in addition to the runtime performance of each of these respective technqiques.

The plotting scripts accept two files, both CSV's the first corresponding to the fitness results file and the second deals with the technique performance time.

The code is all built on top of the ABAGAIL libary and uses the ABIGAIL.jar to call methods and use classes from that package. Also I use the jython2.7 executable.

If you have any questions, feel free to contact me at akogler3@gatech.edu.