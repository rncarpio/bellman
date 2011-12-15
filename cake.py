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
  


# cake.py
# solve the cake eating problem.
#   - deterministic
#   - TODO: add stochastic
#   - with ruin

import scipy, time, sys
import matplotlib.pyplot as plt
import pyublas, debugMsg, maximizer as mx
import lininterp2 as linterp
import bellman

# the deterministic cake eating problem	
class CakeParams1(mx.TestParamsArray):
	# utilityFn is a function that takes one arg and returns utility, u(c)
	def __init__(self, utilityFn, beta, stateVarGrid):
		super(CakeParams1,self).__init__()
		self.m_beta = beta
		self.m_cakeSize = None
		self.m_PrevIterArray = None
		self.m_PrevIterFn = None
		self.m_utilityFn = utilityFn
		self.m_stateVarGrid = stateVarGrid
	def setStateVars(self, stateVarList):
		# state var is the cake size		
		self.m_cakeSize = stateVarList[0]
		# calculate the array of utilities that will be searched for the optimal value
		controlGrid = self.getControlGridList(stateVarList)[0]
		vectorized_utilityFn = scipy.vectorize(self.m_utilityFn)		
		vectorized_nextVFn = scipy.vectorize(lambda c: self.m_PrevIterFn(self.m_cakeSize - c))
		# u(c) + beta * V(cakeSize - c)
		newFnArray = vectorized_utilityFn(controlGrid) + self.m_beta * vectorized_nextVFn(controlGrid)
		mx.TestParamsArray.setFunctionArray1d(self, controlGrid, newFnArray)		
	def getControlGridList(self, stateVarList):
		# return part of the grid that is <= cake size.
		cakeSize = stateVarList[0]
		dx = self.m_stateVarGrid[1] - self.m_stateVarGrid[0]
		i = int((cakeSize - self.m_stateVarGrid[0]) / dx)
		#BREAKPOINT()
		return [self.m_stateVarGrid[:i+1]]
		
	def getNControls(self):
		return 1
	def setPrevIteration(self, wArray):
		self.m_PrevIterArray = wArray
		self.m_PrevIterFn = linterp.LinInterp1D(self.m_stateVarGrid, self.m_PrevIterArray)
		
def test_cake1():
	time1 = time.time()
	localvars = {}	
	cakeSizeGrid = scipy.linspace(0.001, 5, 200);					# state variable grid
	
	def postIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):		
		global g_iterList
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		g_iterList.append((currentVArray, newVArray, optControls))
		localvars[0] = nIter		
	
	initialVArray = cakeSizeGrid;								# initial guess for V: a linear fn
	utilityFn = scipy.log;										# log utility
	beta = 0.8
	params = CakeParams1(utilityFn, beta, cakeSizeGrid)
	result = bellman.grid_valueIteration([cakeSizeGrid], initialVArray, params, postIterCallbackFn=postIterCallbackFn)
	(nIter, currentVArray, newVArray, optControls) = result
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	compareCakeSolution(cakeSizeGrid, beta, currentVArray, optControls[0])
	return result

class CakeParams2(mx.TestParamsFn):
	# utilityFn is a function that takes one arg and returns utility, u(c)
	def __init__(self, utilityFn, beta, stateVarGrid):
		super(CakeParams2,self).__init__()
		self.m_beta = beta
		self.m_cakeSize = None
		self.m_PrevIterArray = None
		self.m_PrevIterFn = None		
		self.m_utilityFn = utilityFn
		self.m_stateVarGrid = stateVarGrid
		self.setObjFn(self.objFn)
		
	def setStateVars(self, stateVarList):
		# state var is the cake size		
		self.m_cakeSize = stateVarList[0]
	# the objective function. will be called from C++
	def objFn(self, argList):
		c = argList[0];		# consumption
		return self.m_utilityFn(c) + self.m_beta * self.m_PrevIterFn(self.m_cakeSize - c)		
	def getControlGridList(self, stateVarList):
		# return part of the grid that is <= cake size.
		cakeSize = stateVarList[0]
		dx = self.m_stateVarGrid[1] - self.m_stateVarGrid[0]
		i = int((cakeSize - self.m_stateVarGrid[0]) / dx)		
		return [self.m_stateVarGrid[:i+1]]
	def getNControls(self):
		return 1
	def setPrevIteration(self, wArray):
		self.m_PrevIterArray = wArray
		self.m_PrevIterFn = linterp.LinInterp1D(self.m_stateVarGrid, self.m_PrevIterArray)
	
def test_cake2():
	time1 = time.time()
	localvars = {}
	cakeSizeGrid = scipy.linspace(0.001, 5, 200);					# state variable grid
	
	def postIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):		
		global g_iterList
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		
	
	initialVArray = cakeSizeGrid;								# initial guess for V: a linear fn
	utilityFn = scipy.log;										# log utility
	beta = 0.9
	params = CakeParams2(utilityFn, beta, cakeSizeGrid);		# don't use parallel search with this, since it makes a callback to Python
	result = bellman.grid_valueIteration([cakeSizeGrid], initialVArray, params, postIterCallbackFn=postIterCallbackFn, parallel=False)
	(nIter, currentVArray, newVArray, optControls) = result
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	compareCakeSolution(cakeSizeGrid, beta, currentVArray, optControls[0])
	return result
	
# use policy iteration		
def test_cake3():
	time1 = time.time()
	localvars = {}
	cakeSizeGrid = scipy.linspace(0.001, 5, 200);					# state variable grid
	
	def postIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		
	
	initialVArray = cakeSizeGrid;								# initial guess for V: a linear fn
	initialPolicyArray = cakeSizeGrid;							# initial guess for policy: eat everything
	utilityFn = scipy.log;										# log utility
	beta = 0.9
	params = CakeParams2(utilityFn, beta, cakeSizeGrid);		# don't use parallel search with this, since it makes a callback to Python
	result = bellman.grid_policyIteration([cakeSizeGrid], [initialPolicyArray], initialVArray, params, postIterCallbackFn=postIterCallbackFn, parallel=False)
	(nIter, currentVArray, currentPolicyArrayList, greedyPolicyList) = result
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	compareCakeSolution(cakeSizeGrid, beta, currentVArray, currentPolicyArrayList[0])
	return result

# compare the solution to the deterministic cake problem with the analytic solution.
def compareCakeSolution(cakeSizeGrid, beta, VArray, policyArray):
	# plot numeric vs. analytic solution
	# see p.19 of Adda & Cooper
	A = (beta * scipy.log(beta) - (1-beta) * scipy.log(1/(1-beta))) / scipy.power(1-beta, 2)
	B = 1/(1-beta)
	analyticV = lambda x: A + B*scipy.log(x)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(cakeSizeGrid, VArray)
	ax.plot(cakeSizeGrid, [analyticV(x) for x in cakeSizeGrid])
	ax.set_xlabel("cake size")
	ax.set_ylabel("V")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(cakeSizeGrid, policyArray)
	ax.plot(cakeSizeGrid, [x*(1-beta) for x in cakeSizeGrid])
	ax.set_xlabel("cake size")
	ax.set_ylabel("optimal consumption")
	plt.show()
	return
	