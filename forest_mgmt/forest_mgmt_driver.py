from mdptoolbox.example import forest
from mdptoolbox.mdp import PolicyIteration, ValueIteration, PolicyIterationModified

from io import StringIO

import sys

import numpy as np

def parse_progress_output(output_str):
  broken_out_lines = [s.strip() for s in output_str.splitlines()]
  overall_value_deltas = []
  for ln in broken_out_lines[1:-1]:
    iteration_output = ln.replace("\t", "").split()
    overall_value_deltas.append([int(iteration_output[0]), float(iteration_output[1])])
  return overall_value_deltas

r1_arr = [2, 4, 7, 10, 12]

r2_arr = [1, 2, 4, 7, 10]

wild_fire_probs = np.arange(0.1, 0.8, 0.1)

varied_wild_fire_value_deltas = []
for wild_fire_prob in wild_fire_probs:
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  P, R = forest(S=100, p=wild_fire_prob)
  pim = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  pim.setVerbose()
  pim.run()
  output = new_stdout.getvalue()
  parsed_param_set_output = parse_progress_output(output)
  sys.stdout = old_stdout
  varied_wild_fire_value_deltas.append(parsed_param_set_output)

varied_r1_value_deltas = []
for r1_val in r1_arr:
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  P, R = forest(S=100, r1=r1_val)
  pim = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  pim.setVerbose()
  pim.run()
  output = new_stdout.getvalue()
  sys.stdout = old_stdout
  parsed_param_set_output = parse_progress_output(output)
  sys.stdout = old_stdout
  varied_r1_value_deltas.append(parsed_param_set_output)

varied_r2_value_deltas = []
for r2_val in r2_arr:
  old_stdout = sys.stdout
  new_stdout = StringIO()
  sys.stdout = new_stdout
  P, R = forest(S=100, r2=r2_val)
  pim = PolicyIterationModified(transitions=P, reward=R, discount=0.9, epsilon=0.01)
  pim.setVerbose()
  pim.run()
  output = new_stdout.getvalue()
  parsed_param_set_output = parse_progress_output(output)
  sys.stdout = old_stdout
  varied_r2_value_deltas.append(parsed_param_set_output)