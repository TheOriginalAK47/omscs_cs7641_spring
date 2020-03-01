import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array

import csv

import time

N=60
T=N/10
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

rhc = RandomizedHillClimbing(hcp)

num_iterations = [1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000]
#num_iterations = [1000, 2000, 5000]
rhc_fitness_list = []
sa_fitness_list = []
ga_fitness_list = []
mim_fitness_list = []

fitness_arr = [['iteration', 'model_name', 'fitness_val']]
perf_arr = [['iteration', 'model_name', 'clock_time']]

for iters in num_iterations:
	fit = FixedIterationTrainer(rhc, iters)
	start_time = time.time()
	fit.train()
	rhc_time = (time.time() - start_time)
	#print "RHC: " + str(ef.value(rhc.getOptimal()))
	rhc_fitness = ef.value(rhc.getOptimal())
	sa = SimulatedAnnealing(1E11, .95, hcp)
	fit = FixedIterationTrainer(sa, iters)
	start_time = time.time()
	fit.train()
	sa_time = (time.time() - start_time)
	#print "SA: " + str(ef.value(sa.getOptimal()))
	sa_fitness = ef.value(sa.getOptimal())
	ga = StandardGeneticAlgorithm(200, 100, 10, gap)
	fit = FixedIterationTrainer(ga, iters)
	start_time = time.time()
	fit.train()
	ga_time = (time.time() - start_time)
	#print "GA: " + str(ef.value(ga.getOptimal()))
	ga_fitness = ef.value(ga.getOptimal())
	mimic = MIMIC(200, 20, pop)
	fit = FixedIterationTrainer(mimic, iters)
	start_time = time.time()
	fit.train()
	mim_time = (time.time() - start_time)
	#print "MIMIC: " + str(ef.value(mimic.getOptimal()))
	mim_fitness = ef.value(mimic.getOptimal())
	fitness_arr.append([iters, 'rhc', rhc_fitness])
	fitness_arr.append([iters, 'sa', sa_fitness])
	fitness_arr.append([iters, 'ga', ga_fitness])
	fitness_arr.append([iters, 'mimic', mim_fitness])
	perf_arr.append([iters, 'rhc', rhc_time])
	perf_arr.append([iters, 'sa', sa_time])
	perf_arr.append([iters, 'ga', ga_time])
	perf_arr.append([iters, 'mimic', mim_time])

with open('continuous_peaks_problem_fitness_results.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|')
	for row in fitness_arr:
		writer.writerow(row)

with open('continuous_peaks_problem_clock_times.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|')
	for row in perf_arr:
		writer.writerow(row)