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
  


import pylab
import scipy, time, sys
import matplotlib.pyplot as plt
import pyublas, _debugMsg, _maximizer as mx
import lininterp2 as linterp
from IPython.Debugger import Tracer; BREAKPOINT = Tracer()

def maximizer1d(controlGridList, bellmanParams, parallel):
	def objFn(x):
		return bellmanParams.objectiveFunction([x])
	fArray = scipy.array(map(objFn, controlGridList[0]))
	argmax = scipy.argmax(fArray)
	return (1, [controlGridList[0][argmax]], fArray[argmax])
	
# wArray is a multidimensional array on a grid
# stateGridList is a list of 1d arrays representing the grid coordinates
# bellmanParams is an object containing the problem-specific information
#   objectiveFunction() is implemented in C++
#   should implement these member functions: setStateVars, setPrevIteration, getControlGridList, getNControls
# parallel is a bool, if true, use the parallel grid search algorithm
def grid_bellman(stateGridList, wArray, bellmanParams, parallel=True):
	stateGridLenList = [len(x) for x in stateGridList]
	nStateVars = len(stateGridList)
	nControls = bellmanParams.getNControls()
	vVals = scipy.zeros(stateGridLenList);										# alloc n-dimensional arrays to hold the V values, size is len_0 x len_1 x ... x len_n	
	optControlVals = [];														# arrays for optimal control values, same size as V
	for i in range(nControls):
		optControlVals.append(scipy.zeros(stateGridLenList))	
	
	bellmanParams.setPrevIteration(stateGridList, wArray)
	for (multiIndex, val) in scipy.ndenumerate(vVals):							# iterate over every point in the grid		
		stateVarList = [stateGridList[i][multiIndex[i]] for i in range(nStateVars)];		# the state variables at this grid point
		bellmanParams.setStateVars(stateVarList)		
		controlGridList = bellmanParams.getControlGridList(stateVarList);				# grids for control variables	
		if (nControls > 1 or parallel==True):
			# call the C++ maximizer, which will maximize bellmanParams.objectiveFunction()
			(count, argmaxList, maxval) = mx.maximizer(controlGridList, bellmanParams, parallel);
		else:
			(count, argmaxList, maxval) = maximizer1d(controlGridList, bellmanParams, parallel)
		vVals.__setitem__(multiIndex, maxval);									# store optimal V, control values
		for i in range(nControls):
			optControlVals[i].__setitem__(multiIndex, argmaxList[i])

		# print("stateVar: %f controlGrid" % stateVarList[0], )
		# print(controlGridList[0])
		# plt.plot(controlGridList[0], [bellmanParams.objectiveFunction([x]) for x in controlGridList[0]])
		# plt.plot(argmaxList[0], maxval, marker='x')
		# plt.draw()		
		# print("press return to continue")
		# pylab.waitforbuttonpress()

	return (vVals, optControlVals)

# default criterion: stop if difference between current and new V is < 0.1%
def defaultValueStoppingCriterion(nIter, currentVArray, newVArray, criterion=0.001):
	diff = newVArray - currentVArray
	pct = diff / currentVArray
	a = abs(pct)		
	# when we allow zero utility, sometimes the pct will have NaNs
	maxdiff = scipy.nanmax(a)
	if (scipy.isnan(maxdiff)):
		assert(False)
	return ((maxdiff < criterion), maxdiff)

# result codes:
ITER_RESULT_CONVERGENCE = 0
ITER_RESULT_MAX_ITERS = 1
ITER_RESULT_MAX_TIME = 2
ITER_RESULT_MAX_V = 3
g_ResultCodeStrings = {
  ITER_RESULT_CONVERGENCE :'ITER_RESULT_CONVERGENCE', 
  ITER_RESULT_MAX_ITERS :'ITER_RESULT_MAX_ITERS',
  ITER_RESULT_MAX_TIME : 'ITER_RESULT_MAX_TIME',
  ITER_RESULT_MAX_V : 'ITER_RESULT_MAX_V'
}
def iterResultString(code):
  return g_ResultCodeStrings[code]

# stoppingCriterionFn: takes 3 args: nIter, currentVArray, newVArray)
#   returns a tuple, first element is True if we should stop iteration, False otherwise
#   rest of elements can be user-specific info
# preIterCallbackFn: takes 0 args
#   called before bellman()
# postIterCallbackFn: takes 5 args: nIter, currentVArray, vVals, optControlVals (returned by bellman), stoppingResult (returned by stoppingCriterionFn)
#   called after bellman()
# other stopping conditions:
#   - if nMaxIters iterations are reached
#   - if total time exceeds maxTime
#   - if the maximum V in the VArray exceeds maxV

def grid_valueIteration(stateGridList, initialVArray, bellmanParams, stoppingCriterionFn=defaultValueStoppingCriterion, preIterCallbackFn=None, postIterCallbackFn=None, 
  nMaxIters=None, maxTime=None, maxV=None, parallel=True):
	cont = True	
	currentVArray = initialVArray
	stoppingResult = None
	nIter = 0
	beginTime = time.time()
	result = None
	
	while (cont == True):
		if (preIterCallbackFn != None): preIterCallbackFn()
		(newVArray, optControls) = grid_bellman(stateGridList, currentVArray, bellmanParams, parallel)
		
		# decide if we stop iterating
		if (stoppingCriterionFn != None): 
			stoppingResult = stoppingCriterionFn(nIter, currentVArray, newVArray)
			if (stoppingResult[0]):
				cont = False
				result = ITER_RESULT_CONVERGENCE
		if (nMaxIters != None and nIter > nMaxIters): cont = False; result = ITER_RESULT_MAX_ITERS
		if (maxTime != None and time.time() - beginTime > maxTime): cont = False; result = ITER_RESULT_MAX_TIME
		if (maxV != None and scipy.amax(newVArray) > maxV): cont = False; result = ITER_RESULT_MAX_V
		
		# post-iteration callback
		if (postIterCallbackFn != None): postIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult)
		
		nIter += 1
		currentVArray = newVArray
	return (result, nIter, currentVArray, newVArray, optControls)

# same as above, with variable-size grid.
# intermediateGrids is a list of stateGridLists. interpolate initialVArray to each element, call grid_valueIteration until convergence, 
# then interpolate to the next elt
def grid_valueIteration2(intermediateGrids, initialGridList, initialVArray, bellmanParams, nMaxIters=None, maxTime=None, maxV=None, **kwargs):
	beginTime = time.time()
	(prevGridList, prevVArray) = (initialGridList, initialVArray)
	for currentGridList in intermediateGrids:
		VArray = linterp.interpolateArray(prevGridList, currentGridList, prevVArray);		# interpolate initial guess to new size
		print("calling grid_valueIteration with gridsize ", [len(g) for g in currentGridList])
		(result, nIter, currentVArray, newVArray, optControls) = grid_valueIteration(currentGridList, VArray, bellmanParams,
		  nMaxIters=nMaxIters, maxTime=maxTime, maxV=maxV, **kwargs)
		if (nMaxIters != None): nMaxIters -= nIter
		if (maxTime != None): maxTime -= (time.time() - beginTime)
		(prevGridList, prevVArray) = (currentGridList, currentVArray)
	finalVArray1 = linterp.interpolateArray(prevGridList, initialGridList, currentVArray);		# interpolate to final size
	finalVArray2 = linterp.interpolateArray(prevGridList, initialGridList, newVArray)
	return (result, nIter, finalVArray1, finalVArray2, optControls)

## functions for policy iteration

# L T_sigma operator in p.144 of Stachurski	
# returns the array of the operator applied to W
# policyFn is a function that takes a sequence of state vars, and returns a list of the chosen policy variables
def LT_sigma(policyFnList, stateGridList, wArray, bellmanParams):
	stateGridLenList = [len(x) for x in stateGridList]
	nStateVars = len(stateGridList)
	vVals = scipy.zeros(stateGridLenList);										# alloc n-dimensional arrays to hold the V values, size is len_0 x len_1 x ... x len_n		
	# use the given W function
	bellmanParams.setPrevIteration(stateGridList, wArray)
	# iterate through all grid points, call objectiveFunction
	for (multiIndex, val) in scipy.ndenumerate(vVals):
		stateVarList = [stateGridList[i][multiIndex[i]] for i in range(nStateVars)]
		controlVarList = [policyFn(stateVarList) for policyFn in policyFnList]
		bellmanParams.setStateVars(stateVarList)
		V = bellmanParams.objectiveFunction(controlVarList)
		vVals.__setitem__(multiIndex, V);
	return vVals

# given a policy (state -> control mapping) and a guess for V, return the value (state -> utility), i.e. utility if we followed the policy
# policyArrayList is a list of arrays on stateGrids
def getValueOfPolicy(policyArrayList, stateGridList, initialVArray, bellmanParams):
	bContinue = True
	policyFnList = [linterp.GetLinterpFnObj(stateGridList, policyArray) for policyArray in policyArrayList]
	currentVArray = initialVArray
	criterion = 0.01
	nIter = 0
	while (bContinue):
		newVArray = LT_sigma(policyFnList, stateGridList, currentVArray, bellmanParams)
		err = scipy.amax(abs(newVArray - currentVArray))
		print("getValueOfPolicy iteration %d err %f" % (nIter, err))
		if (err < criterion):
			bContinue = False
		currentVArray = newVArray
		nIter += 1
	return currentVArray

# calculate a policy that maximizes the given V
def getGreedyPolicy(stateGridList, wArray, bellmanParams, parallel=False):
	(vVals, optControlVals) = grid_bellman(stateGridList, wArray, bellmanParams, parallel);
	return optControlVals	

def defaultPolicyStoppingCriterion(nIter, currentPolicyArrayList, greedyPolicyList, criterion=0.0001):
	diffList = [(greedyPolicyList[i] - currentPolicyArrayList[i]) for i in range(len(greedyPolicyList))]
	pctDiffList = [(diffList[i] / currentPolicyArrayList[i]) for i in range(len(greedyPolicyList))]
	maxdiffList = [scipy.nanmax(abs(diff)) for diff in diffList]
	maxdiff = scipy.amax(maxdiffList)
	return ((maxdiff < criterion), maxdiff)
	
def grid_policyIteration(stateGridList, initialPolicyArrayList, initialVArray, bellmanParams, stoppingCriterionFn=defaultPolicyStoppingCriterion, 
  preIterCallbackFn=None, postIterCallbackFn=None, nMaxIters=None, parallel=False):
	cont = True	
	currentPolicyArrayList = initialPolicyArrayList	
	currentVArray = initialVArray
	stoppingResult = None
	nIter = 0
	while (cont == True and (nMaxIters == None or nIter <= nMaxIters)):
		if (preIterCallbackFn != None): preIterCallbackFn()
		newVArray = getValueOfPolicy(currentPolicyArrayList, stateGridList, currentVArray, bellmanParams)
		greedyPolicyList = getGreedyPolicy(stateGridList, newVArray, bellmanParams, parallel)		
		if (stoppingCriterionFn != None): 
			stoppingResult = stoppingCriterionFn(nIter, currentPolicyArrayList, greedyPolicyList)
			if (stoppingResult[0]):
				cont = False
		if (postIterCallbackFn != None): postIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult)
		nIter += 1
		currentVArray = newVArray
		currentPolicyArrayList = greedyPolicyList
	return (nIter, currentVArray, currentPolicyArrayList, greedyPolicyList)
	
# maximize a static problem in a form that uses value iteration			
class StaticMaxParams(mx.TestParamsArray):
	def __init__(self, controlGridList, fnArray):
		super(StaticMaxParams,self).__init__()
		self.m_beta = 0.5
		self.m_StateVarList = []
		self.m_ControlGridList = controlGridList
		self.m_UArray = fnArray
		for i in range(len(controlGridList)):			# check shape matches grids
			assert(len(controlGridList[i]) == fnArray.shape[i])
			
	def setStateVars(self, stateVarList):
		self.m_StateVarList = stateVarList
	def getControlGridList(self, stateVarList):
		# valid controls are from 0 to the cake size.
		#return [scipy.linspace(0, stateVarList[0], self.m_controlGridSize)]
		# return part of the grid that is <= cake size.
		cakeSize = stateVarList[0]
		dx = self.m_stateVarGrid[1] - self.m_stateVarGrid[0]
		i = int((cakeSize - self.m_stateVarGrid[0]) / dx)		
	def getControlGridList(self, stateVarList):
		return self.m_ControlGridList
	def getNControls(self):
		return len(self.m_ControlGridList)
	def setPrevIteration(self, stateGridList, wArray):		
		# set the array that will be accessed by objectiveFunction().  it will be a function on controlGridList		
		fnArray = self.m_UArray + self.m_beta * wArray.flat[0]
		if (len(self.m_ControlGridList) == 1):
			mx.TestParams.setFunctionArray1d(self, self.m_ControlGridList[0], fnArray)
		elif (len(self.m_ControlGridList) == 2):
			mx.TestParams.setFunctionArray2d(self, self.m_ControlGridList[0], self.m_ControlGridList[1], fnArray)
		elif (len(m_ControlGridList) == 3):
			mx.TestParams.setFunctionArray3d(self, self.m_ControlGridList[0], self.m_ControlGridList[1], self.m_ControlGridList[2], fnArray)
		else:
			assert(false)
			
# fnArray is a multidimensional array defined on controlGrids
# controlGridList is a list of grids
# returns (maxval, argmaxList)
def test_argmax(controlGridList, fnArray):
	time1 = time.time()
	localvars = {}
	def postIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		#print(currentVArray)
		#print(optControls)
		localvars[0] = nIter
		
	stateGrids = [scipy.linspace(0, 10, 10)]
	initialVArray = scipy.random.random_sample(len(stateGrids[0]))
	params = StaticMaxParams(controlGridList, fnArray)
	result = grid_valueIteration(stateGrids, initialVArray, params, postIterCallbackFn=postIterCallbackFn)
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	return result
	
def test1():
	grid1 = scipy.linspace(0, 5, 10)
	fnArray = 2 * scipy.sin(grid1)
	result = test_argmax([grid1], fnArray)
	print(fnArray)
	print("max: %f" % scipy.amax(fnArray))
	print result
	
# keep track of iterations	
g_iterList = [{}]
def resetIters():
	global g_iterList
	g_iterList = [{}]
def getIter(n):
	global g_iterList
	return g_iterList[n]
def getLastIter():
	global g_iterList
	return g_iterList[-1]	
def setInLastIter(key, value):
	getLastIter()[key] = value
def appendIter():
	global g_iterList
	g_iterList.append({})
# for saving and loading
def getItersObj():
	global g_iterList
	return g_iterList
def setItersObj(itersObj):
	global g_iterList
	g_iterList = itersObj

class BParams(mx.BellmanParams):
	def __init__(self):
		super(BParams,self).__init__()
		
	








