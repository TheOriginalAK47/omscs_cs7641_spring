from mdptoolbox.example import forest
from mdptoolbox.mdp import PolicyIteration, ValueIteration, PolicyIterationModified

from io import StringIO

import sys

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

import gym 

def plot_results(results_df, plot_dir):
  ax = sns.lineplot(x="iteration_num", y="value_function_delta", hue="technique", data=results_df)
  plt.xlabel('# of Iterations')
  plt.ylabel("Value Function Delta")
  plt.title("Value Function delta for Frozen Lake Problem")
  plt.savefig(plot_dir + "frozen_lake_problem_value_function_delta.png")
  plt.clf()

def eval_state_action(V, cur_state, a, gamma=0.99):
    return np.sum([p * (reward + gamma * V[next_state]) for p, next_state, reward, _ in env.P[cur_state][a]])

def value_iteration(eps=0.01):
    V = np.zeros(num_states)
    num_iterations = 0
    deltas = []
    while True:
        delta = 0
        for cur_state in range(num_states):
            old_v = V[cur_state]
            V[cur_state] = np.max([eval_state_action(V, cur_state, a) for a in range(num_actions)])
            delta = max(delta, np.abs(old_v - V[cur_state]))
        if delta < eps:
            break
        else:
            deltas.append([num_iterations, np.round(delta, 4), 'value_iteration'])
        num_iterations += 1
    return V, deltas

def policy_evaluation(V, policy_matrix, deltas, num_iterations, eps=0.01):
    while True:
        delta = 0
        for cur_state in range(num_states):
            old_v = V[cur_state]
            V[cur_state] = eval_state_action(V, cur_state, policy_matrix[cur_state])
            delta = max(delta, np.abs(old_v - V[cur_state]))
            #deltas.append(delta)
        if delta < eps:
            break
        else:
        	deltas.append([num_iterations, np.round(delta, 4), 'policy_iteration'])

def policy_improvement(V, policy_matrix):
    policy_stable = True
    for cur_state in range(num_states):
        old_a = policy_matrix[cur_state]
        policy_matrix[cur_state] = np.argmax([eval_state_action(V, cur_state, a) for a in range(num_actions)])
        if old_a != policy_matrix[cur_state]: 
            policy_stable = False
    return policy_stable
 

plots_dir = sys.argv[1]

env = gym.make('FrozenLake-v0')
env = env.unwrapped

num_actions = env.action_space.n
num_states = env.observation_space.n

V = np.zeros(num_states)
policy_matrix = np.zeros(num_states)

policy_stable = False
num_iterations = 0

deltas_arr = []
while not policy_stable:
    policy_evaluation(V, policy_matrix, deltas_arr, num_iterations)
    #deltas_arr.append(deltas_pi)
    policy_stable = policy_improvement(V, policy_matrix)
    num_iterations += 1

V_i, deltas = value_iteration(eps=0.01)

deltas_pd = pd.DataFrame(deltas, columns=['iteration_num', 'value_function_delta', 'technique'])

print('Procedure has converged after ' + str(num_iterations) + ' policy iterations')
print(V.reshape((4,4)))
print(policy_matrix.reshape((4,4)))

deltas_pi = [i for i in deltas_arr if i[0] == 2]
new_deltas_pi = []
iter_num = 1
for delta in deltas_pi:
	new_deltas_pi.append([iter_num, delta[1], 'policy_iteration'])
	iter_num += 1

deltas_pi_pd = pd.DataFrame(new_deltas_pi, columns=['iteration_num', 'value_function_delta', 'technique'])

deltas_pd = deltas_pd.append(deltas_pi_pd)

plot_results(deltas_pd, plots_dir)