//
// Copyright (c) 2011 Ronaldo Carpio
//                                     
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and   
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.          
// It is provided "as is" without express or implied warranty.
//                                                            
  


import scipy, time, sys, itertools, scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle as pickle
import gzip

import lininterp2 as linterp
import pyublas, _debugMsg, _maximizer as mx, bellman, _myfuncs
import _consumptionSavings
from _consumptionSavings import EVMethodT

g_Grid = []
g_IterList = []
g_Params = None

# optimal consumption fraction for CRRA utility, 1 riskless, 1 risky asset
class ConsumptionSavingsParams(_consumptionSavings.ConsumptionSavingsParams):
	CONTROL_GRID_SIZE = 100
	def __init__(self, stateGrid, gamma, beta, mean1, mean2, var2, evMethod=EVMethodT.EV_MONTECARLO):
		super(ConsumptionSavingsParams,self).__init__(stateGrid, gamma, beta, mean1, mean2, var2, evMethod)
		self.consumptionFracGrid = scipy.linspace(0, 1, self.CONTROL_GRID_SIZE)
		#self.asset1FracGrid = scipy.linspace(-2, 2, self.CONTROL_GRID_SIZE)
		self.asset1FracGrid = scipy.array([0.0])
		self.stateGrid = stateGrid
	def getControlGridList(self, stateVarList):		
		wealth = stateVarList[0]
		cGrid = self.stateGrid[:nonzero(self.stateGrid > wealth)[0][0]];	# grid is part of stateGrid <= wealth
		#cGrid = scipy.linspace(self.stateGrid[0], wealth, 100)
		return [cGrid, self.asset1FracGrid]

def test_consumptionSavings1(useValueIter=True, parallel=True, evMethod=EVMethodT.EV_MONTECARLO2):
	global g_IterList, g_Grid, g_Params
	time1 = time.time()
	localvars = {}
	
	def postVIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter
		g_IterList.append((currentVArray, optControls[0], optControls[1]))

	def postPIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		

	grid = scipy.linspace(0.0001, 10, 500)
	g_Grid = grid
	params = ConsumptionSavingsParams(grid, gamma=1.0, beta=0.75, mean1=0.05, mean2=0.5, var2=1.0, evMethod=evMethod)
	g_Params = params
	initialVArray = grid;								# initial guess for V: a linear fn
	g_IterList.append((initialVArray, None, None))
	if (useValueIter == True):		
		result = bellman.grid_valueIteration([grid], initialVArray, params, postIterCallbackFn=postVIterCallbackFn, parallel=parallel)
		(nIter, currentVArray, newVArray, optControls) = result
	else:
		result = bellman.grid_policyIteration([grid], [initialPolicyArray], initialVArray, params, postIterCallbackFn=postPIterCallbackFn, parallel=False)
		(nIter, currentVArray, currentPolicyArrayList, greedyPolicyList) = result
		newVArray = currentVArray
		optControls = currentPolicyArrayList
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	
	# plot V
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(grid, newVArray)
	ax.set_xlabel("W")
	ax.set_ylabel("V")	
	# plot optimal c
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	ax.plot(grid, optControls[0])	
	ax.set_xlabel("W")
	ax.set_ylabel("fraction consumed")	
	# plot optimal s
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	ax.plot(grid, optControls[1])	
	ax.set_xlabel("W")
	ax.set_ylabel("fraction invested in risky asset")		
	plt.show()
	return result

def saveIters(filename):
	global g_IterList
	output = gzip.open(filename, 'wb')
	pickle.dump(g_IterList, output)
	output.close()
def loadIters(filename):
	global g_IterList
	pk_file = gzip.open(filename, 'rb')
	g_iterList = pickle.load(pk_file)
	pk_file.close()
	
def plotIters(i):
	global g_IterList, g_Grid
	
	(vArray, optCFArray, optSArray) = g_IterList[i]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(g_Grid, vArray)
	ax.set_xlabel("W")
	ax.set_ylabel("V")	
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(g_Grid, optCFArray, g_Grid, optSArray)
	ax.legend(('fc', 's'))
	ax.set_xlabel("W")
	ax.set_ylabel("fraction")	

def plotEU(params, n, W):
	global g_IterList
	fig = plt.figure()
	ax = Axes3D(fig)
	prevVArray = g_IterList[n][0]
	params.setPrevIteration(prevVArray)
	params.setStateVars([W])
	(cf_grid, s_grid) = params.getControlGridList([W])
	vals = map(lambda x: params.objectiveFunction([x[0], x[1]]), itertools.product(cf_grid, s_grid))
	[xlist, ylist] = zip(*itertools.product(cf_grid, s_grid))
	ax.scatter(xlist, ylist, vals)
	ax.set_xlabel('fraction consumed')
	ax.set_ylabel('s fraction')
	ax.set_zlabel('EU')	
	plt.show()

	
