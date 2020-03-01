import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys

fitness_input_file = sys.argv[1]

fitness_df = pd.read_csv(fitness_input_file)
ax = sns.lineplot(x="iteration", y="fitness_val", hue="model_name", data=fitness_df)
plt.xlabel('# of Iterations')
plt.ylabel("Fitness Value")
plt.title("Fitness Values for Opt. Techniques on Continuous Peaks Problem")

plt.savefig('continuous_peaks_fitness_plot.png')

clock_times_input_file = sys.argv[2]

plt.clf()

perf_df = pd.read_csv(clock_times_input_file)
ax = sns.lineplot(x="iteration", y="clock_time", hue="model_name", data=perf_df)
plt.xlabel('# of Iterations')
plt.ylabel("Clock Time (s)")
plt.title("Runtime Performance for Opt. Techniques on Continuous Peaks Problem")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.savefig('continuous_peaks_clock_times_plot.png')