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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array

import csv

import time

# set N value.  This is the number of points
N = 20
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

rhc = RandomizedHillClimbing(hcp)

num_iterations = [250, 500, 1000, 2500, 5000, 10000]
rhc_fitness_list = []
sa_fitness_list = []
ga_fitness_list = []
mim_fitness_list = []

fitness_arr = [['iteration', 'model_name', 'fitness_val']]
perf_arr = [['iteration', 'model_name', 'clock_time']]
for iters in num_iterations:
	print("Iter no: " + str(iters))
	fit = FixedIterationTrainer(rhc, iters)
	start_time = time.time()
	fit.train()
	rhc_time = (time.time() - start_time)
	#print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))
	rhc_fitness = ef.value(rhc.getOptimal())
	sa = SimulatedAnnealing(1E12, .999, hcp)
	fit = FixedIterationTrainer(sa, iters)
	start_time = time.time()
	fit.train()
	sa_time = (time.time() - start_time)
	#print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))
	sa_fitness = ef.value(sa.getOptimal())
	ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
	fit = FixedIterationTrainer(ga, iters)
	start_time = time.time()
	fit.train()
	ga_time = (time.time() - start_time)
	#print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))
	ga_fitness = ef.value(ga.getOptimal())
	# for mimic we use a sort encoding
	ef = TravelingSalesmanSortEvaluationFunction(points);
	fill = [N] * N
	ranges = array('i', fill)
	odd = DiscreteUniformDistribution(ranges);
	df = DiscreteDependencyTree(.1, ranges); 
	pop = GenericProbabilisticOptimizationProblem(ef, odd, df);
	mimic = MIMIC(500, 100, pop)
	fit = FixedIterationTrainer(mimic, iters)
	start_time = time.time()
	fit.train()
	mim_time = (time.time() - start_time)
	mim_fitness = ef.value(mimic.getOptimal())
	fitness_arr.append([iters, 'rhc', rhc_fitness])
	fitness_arr.append([iters, 'sa', sa_fitness])
	fitness_arr.append([iters, 'ga', ga_fitness])
	fitness_arr.append([iters, 'mimic', mim_fitness])
	perf_arr.append([iters, 'rhc', rhc_time])
	perf_arr.append([iters, 'sa', sa_time])
	perf_arr.append([iters, 'ga', ga_time])
	perf_arr.append([iters, 'mimic', mim_time])

with open('travelling_salesman_problem_fitness_results_even_less_iters.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|')
	for row in fitness_arr:
		writer.writerow(row)

with open('travelling_salesman_problem_clock_times_even_less_iters.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|')
	for row in perf_arr:
		writer.writerow(row)