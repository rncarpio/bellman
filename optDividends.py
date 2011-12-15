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
  


# optDividends.py
# solve the optimal dividends problem

import scipy, time, sys, itertools, scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyublas, _debugMsg, _maximizer as mx, myfuncs
import lininterp2 as linterp
import bellman
import _optDividends

# the original optimal dividends problem, in de Finetti (1957), Shubik & Thompson (1959)
# everything must be an integer
class OptDivParams1(mx.TestParamsFn):
	# utilityFn is a function that takes one arg and returns utility, u(c)
	# zStates, zProbs are sequences of outcomes, probabilities respectively of the income shock
	def __init__(self, utilityFn, beta, zStates, zProbs, stateVarGrid):
		super(OptDivParams1,self).__init__()
		self.m_beta = beta
		self.m_utilityFn = utilityFn
		self.m_stateVarGrid = stateVarGrid
		self.m_zStates = zStates
		self.m_zProbs = zProbs
		self.m_EVArray = None
		assert(len(zStates) == len(zProbs))
		assert(scipy.sum(zProbs) == 1.0)
		self.setObjFn(self.objFn)
		
	def setStateVars(self, stateVarList):
		# state var is the cash reserve
		self.m_cashReserve = stateVarList[0]
	# the objective function. will be called from C++
	def objFn(self, argList):
		d = argList[0];				# div payout
		M = self.m_cashReserve;		# cash reserve		
		return self.m_utilityFn(d) + self.m_beta * self.m_EVFn(M-d)			
	def getControlGridList(self, stateVarList):
		# return part of the grid that is <= cash reserve.
		M = stateVarList[0]
		dx = self.m_stateVarGrid[1] - self.m_stateVarGrid[0]
		i = int((M - self.m_stateVarGrid[0]) / dx)		
		return [self.m_stateVarGrid[:i+1]]
	def getNControls(self):
		return 1
	def setPrevIteration(self, wArray):		
		self.m_PrevIterArray = wArray
		self.m_PrevIterFn = linterp.LinInterp1D(self.m_stateVarGrid, self.m_PrevIterArray)
		# let S be the post-decision state (or the "end-of-period" state), i.e. M-d but before z is realized.
		def calcEV(S):
			EV = 0.0
			for (zState, zProb) in zip(self.m_zStates, self.m_zProbs):
				nextM = S + zState;		
				EV += zProb * (0.0 if nextM < 0.0 else self.m_PrevIterFn(nextM))
			return EV	
		self.m_EVArray = scipy.array(map(calcEV, self.m_stateVarGrid))
		self.m_EVFn = linterp.LinInterp1D(self.m_stateVarGrid, self.m_EVArray)
		
# same as above, but where z can be a continuous random variable
# zRV is a scipy.stats rv_continuous object
class OptDivParams2(mx.TestParamsFn):
	def __init__(self, utilityFn, beta, zRV, stateVarGrid):
		super(OptDivParams2,self).__init__()
		self.m_beta = beta
		self.m_utilityFn = utilityFn
		self.m_stateVarGrid = stateVarGrid
		self.m_zRV = zRV		
		self.setObjFn(self.objFn)
		
	def setStateVars(self, stateVarList):
		# state var is the cash reserve
		self.m_cashReserve = stateVarList[0]
	# the objective function. will be called from C++
	def objFn(self, argList):
		d = argList[0];				# div payout
		M = self.m_cashReserve;		# cash reserve
		return self.m_utilityFn(d) + self.m_beta * self.m_EVFn(M-d)		
	def getControlGridList(self, stateVarList):
		# return part of the grid that is <= cash reserve.
		M = stateVarList[0]
		# dx = self.m_stateVarGrid[1] - self.m_stateVarGrid[0]
		# i = int((M - self.m_stateVarGrid[0]) / dx)		
		# return [self.m_stateVarGrid[:i+1]]
		return [scipy.linspace(0, M, len(self.m_stateVarGrid))]
	def getNControls(self):
		return 1
	def setPrevIteration(self, wArray):
		self.m_PrevIterArray = wArray
		self.m_PrevIterFn = linterp.LinInterp1D(self.m_stateVarGrid, self.m_PrevIterArray)
		# let S be the post-decision state (or the "end-of-period" state), i.e. M-d but before z is realized
		def calcEV(S):			
			# grid
			def nextV1(nextM):				
				return (0.0 if nextM < 0.0 else self.m_PrevIterFn(nextM))					
			vec_nextV = scipy.vectorize(nextV1)
			result = myfuncs.calculateEV_grid(self.m_stateVarGrid, vec_nextV, self.m_zRV, zOffset=S, leftK=0.0, rightK=self.m_PrevIterFn(self.m_stateVarGrid[-1]))			

			# integrate
			def nextV2(z):
				nextM = S + z
				return (0.0 if nextM < 0.0 else self.m_PrevIterFn(nextM))								
			#result = myfuncs.calculateEV_integrate(nextV2, self.m_zRV, a=-S)

			# monte carlo
			#loc = m_zRV.kwds['loc']
			#scale = m_zRV.kwds['scale']
			#grid2 = scipy.zeros(len(self.m_stateVarGrid) + 1)
			
			#result = myfuncs.calculateEV_montecarlo2(

			return result
			
			
		self.m_EVArray = scipy.array(map(calcEV, self.m_stateVarGrid))
		self.m_EVFn = linterp.LinInterp1D(self.m_stateVarGrid, self.m_EVArray)

# use c++ class, Z is calculated with monte carlo
# pass in an array of monte carlo values for Z
class OptDivParams3(_optDividends.OptDividendsParams):
	def __init__(self, stateGrid, beta, randomDraws):
		super(OptDivParams3,self).__init__(stateGrid, beta, randomDraws)
		self.stateGrid = stateGrid
		self.beta = beta
	def getControlGridList(self, stateVarList):
		# return part of the grid that is <= cash reserve.
		M = stateVarList[0]		
		return [self.stateGrid[self.stateGrid <= M]]
		
# for solution, see Schmidli, chapter 1	
# x_0 := sup{x : u(x) = 0}, i.e. the largest starting cash value such that the optimal payout is 0.
# if x_0 = 0, then it is never optimal to save anything; u(x) = x
# Schmidli's e^(-sigma) is our beta -> e^sigma = 1/beta
# returns True if x_0=0 (see Schmidli p.18)
def alwaysPayAll(beta, p):
	return 1.0/beta >= p + scipy.sqrt(p * (1.0-p))
def plot_alwaysPayAll(beta):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	grid = scipy.linspace(0.0, 1.0, 100)
	lhs = map(lambda p: 1.0/beta, grid)
	rhs = map(lambda p: p + scipy.sqrt(p * (1.0-p)), grid)	
	ax.plot(grid, lhs)
	ax.plot(grid, rhs)
	ax.set_xlabel("p (beta = %f)" % beta)
	plt.show()

# n_0 = inf{n : u(n)=1}, the first point where a dividend is paid
def getRoots1(beta, p):
	root1 = (1.0/beta + scipy.sqrt(scipy.square(1.0/beta) - 1 + scipy.square((2*p - 1)))) / (2*p)
	root2 = (1.0/beta - scipy.sqrt(scipy.square(1.0/beta) - 1 + scipy.square((2*p - 1)))) / (2*p)
	return (root1, root2)
def getn0(beta, p):
	(root1, root2) = getRoots1(beta, p)
	n = scipy.log( (-(1.0 - root2) * scipy.log(root2)) / ((root1 - 1.0) * scipy.log(root1)) ) / scipy.log(root1 / root2)
	return n
def plot_n0():
	fig = plt.figure()
	ax = Axes3D(fig)
	grid_beta = scipy.linspace(0.8, 0.99, 20)
	grid_p = scipy.linspace(0.5, 0.8, 20)	
	vals = map(lambda x: getn0(x[0], x[1]), itertools.product(grid_beta, grid_p))
	[xlist, ylist] = zip(*itertools.product(grid_beta, grid_p))	
	ax.scatter(xlist, ylist, vals)
	ax.set_xlabel('beta')
	ax.set_ylabel('pHigh')
	ax.set_zlabel('n0')	
	return (xlist, ylist, vals)

# note that for beta=0.99, pHigh=0.6, Schmidli's n0 doesn't match the numeric result.  Or does it? It can be floor(n0) or floor(n0+1)... which one is it?
# default grid is integers only.
def test_optdiv1(beta=0.9, pHigh=0.75, grid=scipy.arange(21.0), useValueIter=True):
	time1 = time.time()
	localvars = {}
	
	def postVIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):		
		global g_iterList
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		

	def postPIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		
		
	initialVArray = grid;								# initial guess for V: a linear fn
	initialPolicyArray = grid;							# initial guess for d: pay out everything
	utilityFn = lambda x: x;							# linear utility
	zStates = [-1.0, 1.0];	
	zProbs = [1.0-pHigh, pHigh];						# income shock	
	params = OptDivParams1(utilityFn, beta, zStates, zProbs, grid);		# don't use parallel search with this, since it makes a callback to Python			
	if (useValueIter == True):		
		result = bellman.grid_valueIteration([grid], initialVArray, params, postIterCallbackFn=postVIterCallbackFn, parallel=False)
		(nIter, currentVArray, newVArray, optControls) = result
	else:
		result = bellman.grid_policyIteration([grid], [initialPolicyArray], initialVArray, params, postIterCallbackFn=postPIterCallbackFn, parallel=False)
		(nIter, currentVArray, currentPolicyArrayList, greedyPolicyList) = result
		newVArray = currentVArray
		optControls = currentPolicyArrayList
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	
	print("x_0 == 0: %d" % alwaysPayAll(beta, pHigh))
	n0 = getn0(beta, pHigh)
	optd_fn = linterp.LinInterp1D(grid, optControls[0])
	print("n0: %f, d(floor(n0)): %f" % (n0, optd_fn(scipy.floor(n0))))
	# plot V
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(grid, newVArray)
	ax.set_xlabel("M")
	ax.set_ylabel("V")	
	# plot optimal d
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	ax.plot(grid, optControls[0])	
	ax.axvline(scipy.floor(n0), color='gray')
	ax.set_xlabel("M")
	ax.set_ylabel("optimal d")	
	plt.show()
	return result
		
# see Gerber & Shiu (2004), p. 4
# mu = drift of income shock
# sigma = volatility
# delta = continuous discount factor		
def getRoots2(mu, sigma, delta):
	r = (-mu + scipy.sqrt(mu*mu + 2*delta*sigma*sigma)) / (sigma*sigma)
	s = (-mu - scipy.sqrt(mu*mu + 2*delta*sigma*sigma)) / (sigma*sigma)
	return (r, s)

# conditions that must be satisfied:
# V(0,b)=0, V'(b,b)=1
# V(x,b) = g(x) / g'(b) for 0 <= x <= b, where g(x) = exp(r*x) - exp(s*x)
# V(b*, b*) = mu/delta

# the optimal barrier value
def calc_opt_b(mu, sigma, delta):
	(r, s) = getRoots2(mu, sigma, delta)
	bstar = (2.0/(r-s)) * scipy.log(-s/r)
	return bstar
	
# continuous time version.
# see Asmussen & Taksar, Gerber & Shiu (2004) for solution.
# delta = continuous discount rate
# mu = drift parameter
# sigma = volatility
# dt = time step
def test_optdiv2(delta=0.1, mu=0.5, sigma=1.0, dt=1.0, grid=scipy.linspace(0, 5, 200), useValueIter=True):
	time1 = time.time()
	localvars = {}
	
	def postVIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):		
		global g_iterList
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		

	def postPIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		
		
	initialVArray = grid;								# initial guess for V: a linear fn
	initialPolicyArray = grid;							# initial guess for d: pay out everything	
	utilityFn = lambda x: x;							# linear utility
	beta = scipy.power(scipy.e, -(delta * dt))
	print("beta = exp(- %f * %f) = %f" % (delta, dt, beta))
	zRV = scipy.stats.norm(loc=mu*dt, scale=sigma*scipy.sqrt(dt))
	print("income shock: mean %f, sd %f" % (mu*dt, sigma*dt))
	bstar = calc_opt_b(mu, sigma, delta)
	print("optimal barrier: %f" % bstar)
	params = OptDivParams2(utilityFn, beta, zRV, grid);		# don't use parallel search with this, since it makes a callback to Python			
	if (useValueIter == True):		
		result = bellman.grid_valueIteration([grid], initialVArray, params, postIterCallbackFn=postVIterCallbackFn, parallel=False)
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
	ax.set_xlabel("M")
	ax.set_ylabel("V")	
	# plot optimal d
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	ax.plot(grid, optControls[0])	
	ax.axvline(bstar, color='gray')
	ax.set_xlabel("M")
	ax.set_ylabel("optimal d")	
	plt.show()
	return

def test_optdiv3(beta=0.9, grid=scipy.arange(21.0), zDraws=scipy.array([-1.0]*25 + [1.0]*75), useValueIter=True):
	time1 = time.time()
	localvars = {}
	
	def postVIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):		
		global g_iterList
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		

	def postPIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		
		
	initialVArray = grid;								# initial guess for V: a linear fn
	initialPolicyArray = grid;							# initial guess for d: pay out everything
	params = OptDivParams3(grid, beta, zDraws);
	if (useValueIter == True):		
		result = bellman.grid_valueIteration([grid], initialVArray, params, postIterCallbackFn=postVIterCallbackFn, parallel=True)
		(nIter, currentVArray, newVArray, optControls) = result
	else:
		result = bellman.grid_policyIteration([grid], [initialPolicyArray], initialVArray, params, postIterCallbackFn=postPIterCallbackFn, parallel=False)
		(nIter, currentVArray, currentPolicyArrayList, greedyPolicyList) = result
		newVArray = currentVArray
		optControls = currentPolicyArrayList
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))
	
	optd_fn = linterp.LinInterp1D(grid, optControls[0])
	# plot V
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(grid, newVArray)
	dx = grid[1] - grid[0]
	deriv = scipy.diff(newVArray) / dx
	ax.plot(grid[:-1], deriv)
	ax.set_xlabel("M")
	ax.set_ylabel("V")		
	# plot optimal d
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	ax.plot(grid, optControls[0])	
	ax.set_xlabel("M")
	ax.set_ylabel("optimal d")	
	plt.show()
	return result

# use simulation to calculate the EV of a payout strategy
# the discrete time +-1 problem
class Firm:
	def __init__(self, initialM, beta, payoutBarrier):
		self.M = initialM
		self.beta = beta
		self.accumulatedUtility = 0.0;			# this is present utility, discounted back to time=0
		self.payoutBarrier = payoutBarrier
		self.age = 0
		self.currentPayout = 0.0
		self.ruined = False
		
	def choosePayout(self, M):
		return (0.0 if M < self.payoutBarrier else M-self.payoutBarrier)
				
	# returns true if not ruined
	def onIncomeShock(self, z):
		assert(not self.ruined)
		d = self.choosePayout(self.M)
		self.accumulatedUtility += (self.beta ** self.age) * d
		nextM = self.M - d + z
		self.age += 1
		if (nextM < 0.0):
			self.ruined = True
			self.finalM = nextM
			return False
		self.M = nextM
		return True
	
def simulateFirm(firm, p, nPeriods):	
	minus_one = scipy.ones(nPeriods) * -1
	plus_one = scipy.ones(nPeriods)
	draws = scipy.stats.bernoulli.rvs(p, size=nPeriods)
	shocks = scipy.where(draws == 1.0, plus_one, minus_one)
	
	for z in shocks:
		survive = firm.onIncomeShock(z)
		if (not survive):
			#print("ruined at %d, V=%f" % (firm.age, firm.accumulatedUtility))
			return (False, firm.age, firm.accumulatedUtility)
	#print("survived until end")
	return (True, firm.age, firm.accumulatedUtility)
	return

def test_sim1():
	p = 0.6
	payoutBarrier = 1.0
	beta = 0.99
	nPeriods = 1000
	nRuns = 1000
	M_grid = scipy.arange(21.0)
	V_array = scipy.zeros((len(M_grid), len(M_grid)))
	
	for (iM, M) in enumerate(M_grid):
		#print("M=%f" % M)
		b_grid = M_grid[M_grid <= M]
		for (ib, b) in enumerate(b_grid):
			V_simulated_list = []
			for j in range(nRuns):
				firm = Firm(M, beta, b)
				(survived, age, V) = simulateFirm(firm, p, nPeriods)
				V_simulated_list.append(V)
			EV = scipy.mean(V_simulated_list)
			V_array[iM, ib] = EV

	for (iM, M) in enumerate(M_grid):
		V_of_b = V_array[iM,:]
		print(V_of_b)
		optimal_ib = scipy.argmax(V_of_b)
		optimal_b = M_grid[optimal_ib]
		print("for M=%f, optimal b is %f" % (M, optimal_b))
		
	fig = plt.figure()
	ax = Axes3D(fig)
	vals = V_array.flatten()
	[xlist, ylist] = zip(*itertools.product(M_grid, b_grid))	
	ax.scatter(xlist, ylist, vals)
	ax.set_xlabel('M')
	ax.set_ylabel('b')
	ax.set_zlabel('EV')	

			
	