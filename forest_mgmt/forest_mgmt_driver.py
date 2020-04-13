from mdptoolbox.example import forest
from mdptoolbox.mdp import PolicyIteration, ValueIteration, PolicyIterationModified, QLearning

from io import StringIO

import sys

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

def parse_progress_output(output_str, param_name, param_val):
  broken_out_lines = [s.strip() for s in output_str.splitlines()]
  overall_value_deltas = []
  for ln in broken_out_lines[1:-1]:
    iteration_output = ln.replace("\t", "").split()
    overall_value_deltas.append([float(iteration_output[0]), float(iteration_output[1]), param_name, float(param_val)])
  return overall_value_deltas

def process_q_learning_results(mean_vector, param_name, param_val):
  results_arr = []
  q_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  for iter_num in np.arange(1, 101, 1):
    #result_arr = []
    result_arr = [[float(iter_num), float(mean_vector[iter_num - 1]), param_name, float(param_val)]]
    row_as_df = pd.DataFrame(result_arr, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
    q_df = q_df.append(row_as_df)
  return q_df


def plot_results(results_df, technique, param_name, plot_dir):
  ax = sns.lineplot(x="iteration_num", y="value_function_delta", hue="param_val", data=results_df)
  plt.xlabel('# of Iterations')
  plt.ylabel("Value Function Delta")
  plt.title("Value Function delta for " + param_name + " using " + technique)
  plt.savefig(plot_dir + "forest_management_problem_value_function_delta_using_" + technique + "_varying_param_" + param_name + ".png")
  plt.clf()

r1_arr = [2, 4, 7, 10, 12]

r2_arr = [1, 2, 4, 7, 10]

wild_fire_probs = np.arange(0.1, 0.8, 0.1)

plots_dir = sys.argv[1]

wild_fire_pi_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
wild_fire_vi_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
wild_fire_q_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
for wild_fire_prob in wild_fire_probs:
  varied_wild_fire_value_deltas = []
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  P, R = forest(S=5000, p=wild_fire_prob)
  pim = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  pim.setVerbose()
  pim.run()
  output = new_stdout.getvalue()
  parsed_param_set_output = parse_progress_output(output, "wild_fire_prob", wild_fire_prob)
  sys.stdout = old_stdout
  output_row_as_df = pd.DataFrame(parsed_param_set_output, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  wild_fire_pi_df = wild_fire_pi_df.append(output_row_as_df)
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  vi = ValueIteration(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  vi.setVerbose()
  vi.run()
  output = new_stdout.getvalue()
  parsed_param_set_output = parse_progress_output(output, "wild_fire_prob", wild_fire_prob)
  sys.stdout = old_stdout
  other_output_row_as_df = pd.DataFrame(parsed_param_set_output, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  wild_fire_vi_df = wild_fire_vi_df.append(other_output_row_as_df)
  q = QLearning(transitions=P, reward=R, discount=0.9)
  q.run()
  deltas_arr = q.mean_discrepancy
  wild_fire_q_df = wild_fire_q_df.append(process_q_learning_results(deltas_arr, 'wild_fire_prob', wild_fire_prob))
plot_results(wild_fire_pi_df, 'policy_iteration', 'wild_fire_prob', plots_dir)
plot_results(wild_fire_vi_df, 'value_iteration', 'wild_fire_prob', plots_dir)
plot_results(wild_fire_q_df, 'q_learning', 'wild_fire_prob', plots_dir)

r1_pi_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
r1_vi_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
r1_q_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
for r1_val in r1_arr:
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  P, R = forest(S=5000, r1=r1_val)
  pim = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  pim.setVerbose()
  pim.run()
  output = new_stdout.getvalue()
  sys.stdout = old_stdout
  parsed_param_set_output = parse_progress_output(output, "r1_value", r1_val)
  sys.stdout = old_stdout
  output_row_as_df = pd.DataFrame(parsed_param_set_output, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  r1_pi_df = r1_pi_df.append(output_row_as_df)
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  vi = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  vi.setVerbose()
  vi.run()
  output = new_stdout.getvalue()
  sys.stdout = old_stdout
  parsed_param_set_output = parse_progress_output(output, "r1_value", r1_val)
  sys.stdout = old_stdout
  other_output_row_as_df = pd.DataFrame(parsed_param_set_output, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  r1_vi_df = r1_vi_df.append(other_output_row_as_df)
  q = QLearning(transitions=P, reward=R, discount=0.9)
  q.run()
  deltas_arr = q.mean_discrepancy
  r1_q_df = r1_q_df.append(process_q_learning_results(deltas_arr, 'r1_value', r1_val))
plot_results(r1_pi_df, 'policy_iteration', 'r1_reward_val', plots_dir)
plot_results(r1_vi_df, 'value_iteration', 'r1_reward_val', plots_dir)
plot_results(r1_q_df, 'q_learning', 'r1_reward_val', plots_dir)

r2_pi_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
r2_vi_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
r2_q_df = pd.DataFrame(columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
for r2_val in r2_arr:
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  P, R = forest(S=5000, r2=r2_val)
  pim = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  pim.setVerbose()
  pim.run()
  output = new_stdout.getvalue()
  parsed_param_set_output = parse_progress_output(output, "r2_value", r2_val)
  sys.stdout = old_stdout
  output_row_as_df = pd.DataFrame(parsed_param_set_output, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  r2_pi_df = r2_pi_df.append(output_row_as_df)
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  vi = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  vi.setVerbose()
  vi.run()
  output = new_stdout.getvalue()
  parsed_param_set_output = parse_progress_output(output, "r2_value", r2_val)
  sys.stdout = old_stdout
  other_output_row_as_df = pd.DataFrame(parsed_param_set_output, columns=['iteration_num', 'value_function_delta', 'param_name', 'param_val'])
  r2_vi_df = r2_vi_df.append(other_output_row_as_df)
  q = QLearning(transitions=P, reward=R, discount=0.9)
  q.run()
  deltas_arr = q.mean_discrepancy
  r2_q_df = r2_q_df.append(process_q_learning_results(deltas_arr, 'r2_value', r2_val))
plot_results(r2_pi_df, 'policy_iteration', 'r2_reward_val', plots_dir)
plot_results(r2_vi_df, 'value_iteration', 'r2_reward_val', plots_dir)
plot_results(r2_q_df, 'q_learning', 'r2_reward_val', plots_dir)