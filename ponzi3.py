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
  

	
import scipy
import numpy as np
from collections import defaultdict
from scipy import linspace, linalg, mean, exp, randn, stats, interpolate, integrate, array, meshgrid
from scipy.optimize import fminbound, tnc
from lininterp import LinInterp, InterpTrilinear
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
#from pylab import *
from IPython.Debugger import Tracer; debug_here = Tracer()
import copy, time
import _ponzi3_fns as p3_fns
import cPickle as pickle
import gzip
import os
import networkx as nx

# my modules. 
import my_bfs
import globals as g
reload(my_bfs)
reload(g)

# interpolation class to use
Interp = LinInterp
class Interp2D:
	# X1, X2 are grids
	def __init__(self, X1, X2, f):
		(self.X1, self.X2, self.f) = (X1, X2, f)
		
	def __call__(self, x):
		return p3_fns.interp2(self.X1, self.X2, self.f, x[0], x[1])
	
# Utility
# exponential utility
def U_exponential(c): return 1 - exp(- g.theta * c) 
def Uprime_exponential(c): return g.theta*exp(- g.theta * c) 
# linear utility
def U_linear(c): return c
def Uprime_linear(c): return (c - c + 1)
# CRRA
def U_crra(c): return c**(1-g.gamma) / (1-g.gamma)
def Uprime_crra(c): return c**(-g.gamma)

U = U_linear
Uprime = Uprime_linear
#U = U_exponential
#Uprime = Uprime_exponential
#U = U_crra
#Uprime = Uprime_crra

# the interest rate that will induce depositor to switch from f=0 to f=1
# only depends on depositor's slope (i.e. mean-SD tradeoff) and probability of survival
def rBar():
	return 1.0 / (g.pHigh - g.depositorSlope * scipy.sqrt(g.pHigh * (1-g.pHigh)))
	

# fraction of household income that is saved, as a function of r_t.
# at r=1, save 0 (hold cash).  Asymptotically approach 1.
def fs(r):
	return  1.0 - exp(1.0 - r)
	
def maximizer1(h, a, b):
	# sometimes fminbound doesn't catch the boundaries.
	result = float(fminbound(lambda x: -h(x), a, b))
	if (h(a) > h(result) and h(a) > h(b)):
		return a
	if (h(b) > h(result) and h(b) > h(a)):
		return b
	return result
	
# crashes 
def maximizer2(h, a, b):
	result = scipy.optimize.fmin_slsqp(lambda x: -h(x), array([(a+b)/2]), bounds=[(a,b)], iprint=0)
	return result[0]
def maximizer2d_slsqp(h, bounds1, bounds2):
	(a1, a2) = (bounds1[0], bounds1[1])
	(b1, b2) = (bounds2[0], bounds2[1])
	init_guess = [(a1+a2)/2, (b1+b2)/2]
	result = scipy.optimize.fmin_slsqp(lambda x: -h(x), array(init_guess), bounds=[bounds1, bounds2], iprint=0)
	return result

def maximizer3(h, a, b):
	result = scipy.optimize.fmin_tnc(lambda x: -h(x[0]), [(a+b)/2.0], bounds=[(a,b)], approx_grad=True, messages = scipy.optimize.tnc.MSg.NONE)
	#debug_here()
	return result[0][0]
def maximizer2d_tnc(h, bounds1, bounds2, guess=None):
	(a1, a2) = (bounds1[0], bounds1[1])
	(b1, b2) = (bounds2[0], bounds2[1])
	init_guess = [(a1+a2)/2, (b1+b2)/2]
	if (guess != None):
		if (a1 <= guess[0] and guess[0] <= a2 and b1 <= guess[1] and guess[1] <= b2):
			init_guess = guess
	(x, nfeval, rc) = scipy.optimize.fmin_tnc(lambda x: -h(x), array(init_guess), bounds=[bounds1, bounds2], approx_grad=True, messages = scipy.optimize.tnc.MSg.NONE)
	if (rc == -1 or rc > 4):
	  print("fmin_tnc error: %s", scipy.optimize.tnc.RCSTRINGS[rc])
	  assert(False)	 
	return x

# if the range is small enough, call optimize.brute
def maximizer4(h, a, b):
	n_points = 20
	if ((a + n_points*gridspace) > b):
		return maximizer1(h, a, b)
	return scipy.optimize.brute(lambda x: -h(x), [(a,b)], Ns=n_points, finish=None)
	
maximizer = maximizer1
maximizer2d = maximizer2d_tnc
###############################
# up to here, these are buit in maximizers/minimizers. don't use them now, since everything's in c++
# might be useful for a simple 1-d problem, compare results with c++ code
################################

# calculate v_{t+1} conditional on z > (-s), 0 otherwise
# w should be a vectorized fn, s scalar
# not vectorized on s, since s + z_draws won't match.
def next_v(w, s):
	nextM = s + z_draws
	val1 = w(nextM)
	val2 = 0
	return scipy.where(nextM > 0, val1, val2)

# v'(M)	
def expUtilPrime(w, wPrime, M, opt_s):
	# all consumption, no savings -> use U'(c)
	if (opt_s == 0):
		return Uprime(M)
	# all savings, no consumption -> use beta * E[w'(s,z)]
	if (opt_s == M):
		return g.beta * mean(next_v(wPrime, M))
	# in between
	return Uprime(M - opt_s)

# calculate E[w(s+z)], with w() =0 if s+z < 0.
# use the grid directly.
# make sure we're not being passed a vector!
def expected_next_v1(w, s, rv):
	assert(grid[0] >= 0)
	assert(scipy.size(s) == 1)
	#debug_here()
	below = rv.cdf(0-s)*0 + (rv.cdf(grid[0]-s) - rv.cdf(0-s)) * w(grid[0])
	#below = rv.cdf(grid[0]-s) * w(grid[0])
	# use w(grid_max) to extrapolate past grid_max
	above = (1 - rv.cdf(grid[gridsize-1]-s)) * w(grid[gridsize-1])
	# element-wise product
	between_fn = w(grid) * rv.pdf(grid - s)
	between = scipy.integrate.trapz(between_fn, grid)
	return below + between + above

# compare expected_next_v to monte carlo
def test_env(w):
	figure()
	fn1 = lambda s: mean(next_v(w, s))
	fn2 = lambda s: expected_next_v(w, s, z_rv)
	plot(grid, map(fn1, list(grid)), grid, map(fn2, list(grid)))

def expected_next_v(w,M,D,d,r):
	sum = 0
	for z in g.z_space:
		next_M = (M + fs(r) * z - D - d) / z
		next_D = r * fs(r)
		if (next_M <= 0):
			sum += 0
		else:
			sum += g.z_rv.pmf(z) * w((next_M, next_D))
	return sum

####################################3
# up to here, this was the old python stuff for expected next value.
# don't think we'll need this 
#####################################

# w is a function that takes (M,D,N) as arg
def bellman1(w, useNew=False, parallel=True):	
	vals = scipy.zeros((g.gridsize_M, g.gridsize_D))
	opt_d = scipy.zeros((g.gridsize_M, g.gridsize_D))
	opt_r = scipy.zeros((g.gridsize_M, g.gridsize_D))

	opt_d2 = scipy.zeros((g.gridsize_M, g.gridsize_D))
	opt_r2 = scipy.zeros((g.gridsize_M, g.gridsize_D))
	
	i = 0
	t1 = time.time()
	argmax = scipy.zeros(2)
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			
			#exp_util = lambda x: p3_fns.expUtil(w.f, M, D, x[0], x[1])
	
			# find max in c
			# d is on g.grid_M, but it's the amount taken out, so make sure 0 is possible (last time it was s, so it was the same)
			grid_d = linspace(0, M, g.grid_d_size)
			if (not useNew):
				useC = 1
				bParallel = 0
				if (parallel): bParallel=1
				(count, argmax[0], argmax[1], maxval) = p3_fns.maximizer2d(grid_d, g.grid_r, w.f, M, D, useC, bParallel, 0)
				opt_d[iM, iD] = argmax[0]
				opt_r[iM, iD] = argmax[1]
			else:
				argGrids = [grid_d, g.grid_r]
				(count, maxval, argmaxList) = p3_fns.maximizer(argGrids, [M,D], w.f, parallel, 0)
				opt_d[iM, iD] = argmaxList[0]
				opt_r[iM, iD] = argmaxList[1]
			vals[iM, iD] = maxval		
		
			i += 1
		t2 = time.time()		
		#print("%i %i: %f sec" % (iM, iD, t2-t1))
		t1 = t2
	
	new_v = Interp2D(g.grid_M, g.grid_D, vals)
	opt_d_fn = Interp2D(g.grid_M, g.grid_D, opt_d)
	opt_r_fn = Interp2D(g.grid_M, g.grid_D, opt_r)
	#opt_d2_fn = Interp2D(g.grid_M, g.grid_D, grid_N, opt_d2)
	#opt_r2_fn = Interp2D(g.grid_M, g.grid_D, grid_N, opt_r2)
	
	addToIter('v', new_v)
	addToIter('opt_d', opt_d_fn)	
	addToIter('opt_r', opt_r_fn)	
	#addToIter('opt_d2', opt_d2_fn)	
	#addToIter('opt_r2', opt_r2_fn)		
	result = (new_v, opt_d_fn, opt_r_fn)
	
	return result

# generalize bellman1 to any dimension of grids, controls
# stateGridList is a list of grids
# controlGridFnList is a list of functions, each takes the state vars as arg and returns the grid for the control variable
#   this is to allow the grids to vary based on the state variables
def bellman2(stateGridList, controlGridFnList, w, parallel=False):
	stateGridLenList = [len(x) for x in stateGridList]
	nStateVars = len(stateGridList)
	nControls = len(controlGridFnList)
	# alloc n-dimensional arrays to hold the V values, size is len_0 x len_1 x ... x len_n
	vVals = scipy.zeros(stateGridLenList)
	# arrays for optimal control values
	optControlVals = []
	for i in range(nControls):
		optControlVals.append(scipy.zeros(stateGridLenList))
	
	iterCount = 0
	t1 = time.time()
	argmax = scipy.zeros(nControls)
	stateGridIndexList = [0] * nStateVars
	bDone = False
	# check for zero size
	if (0 in stateGridLenList): bDone = True
		
	while (not bDone):
		# state vars are args
		argList = [stateGridList[i][stateGridIndexList[i]] for i in range(nStateVars)]	
		# the actual objective fn is done in c++.  we need to pass:
		# 1) list of state vars at this iteration
		# 2) w.f, a parameter that will be used in the objective fn
		# 3) grids for control vars to gridsearch over
		gridList = [gridFn(argList) for gridFn in controlGridFnList]
		result = p3_fns.maximizer(gridList, argList, w.f, parallel, 0)
		count = result[0]
		maxval = result[1]		
		argmax = result[2]
		
		# save argmax, value results
		vVals.__setitem__(tuple(stateGridIndexList), maxval)
		for i in range(nControls):
			optControlVals[i].__setitem__(tuple(stateGridIndexList), argmax[i])
		
		# increment indices, check if done
		# go backwards, add 1, cycle if necessary
		bDone = True
		for i in range(nStateVars-1, -1, -1):
			stateGridIndexList[i] += 1
			if (stateGridIndexList[i] == stateGridLenList[i]):
				stateGridIndexList[i] = 0
			else:
				bDone = False
				break
			# if we make it here, all indices cycle, therefore we're done
			
		iterCount += 1
		t2 = time.time()				
		t1 = t2	
		
	result = (vVals, optControlVals)
	return result

def bellman3(w, **kwargs):
	# grid_d = linspace(0, M, g.grid_d_size)
	grid_d_fn = lambda x: linspace(0, x[0], g.grid_d_size)
	grid_r_fn = lambda x: g.grid_r
	stateGridList = [g.grid_M, g.grid_D]
	controlGridFnList = [grid_d_fn, grid_r_fn]
	(vVals, optControlVals) = bellman2(stateGridList, controlGridFnList, w, **kwargs)

	new_v = Interp2D(g.grid_M, g.grid_D, vVals)
	opt_d_fn = Interp2D(g.grid_M, g.grid_D, optControlVals[0])
	opt_r_fn = Interp2D(g.grid_M, g.grid_D, optControlVals[1])
	
	addToIter('v', new_v)
	addToIter('opt_d', opt_d_fn)	
	addToIter('opt_r', opt_r_fn)	
	
	result = (new_v, opt_d_fn, opt_r_fn)
	return result

#def test1():
#	nStateVars = 3
#	stateGridIndexList = [0, 0, 0]
#	stateGridLenList = [2, 3, 4]
#	bDone = False
#	while (not bDone):
#		print(stateGridIndexList)
#		bDone = True
#		for i in range(nStateVars-1, -1, -1):
#			stateGridIndexList[i] += 1
#			if (stateGridIndexList[i] == stateGridLenList[i]):
#			else:
#				bDone = False
#				break
#			# if we make it here, all indices cycle, therefore we're done

# bellman1 is the c version.
# for some reason, the new c++ version that takes any dimension, outputs different diff numbers when run in parallel and with useNew
# bellman1 calls gridSearch2DParallel, while bellman3 calls gridSearchParallel
# TODO: figure out why they're different		
bellman = bellman1

##################################################################################3
# bellman functions.  definitely going to reuse these.  figure out why the general dimension case returns different values (check the DBL_MAX)
# should work with 1-dimensional insurance problems, brownian, etc
# clean up the API?  What's its inputs and outputs?  Hooks (e.g. addToIter)?
	
def viter_until(v, criterion=0.001, n=None, **kwargs):
	cont = True	
	currentV = v	
	iter = 0	
	t1 = time.time()
	while (cont == True):
		nextIter()
		(newV, opt_d, opt_r) = bellman(currentV, **kwargs)
		d1 = newV.f - currentV.f
		pct = d1 / currentV.f
		a = abs(pct)
		#diff = scipy.amax(a)
		# when we allow zero utility, sometimes the pct will have NaNs
		diff = scipy.nanmax(a)
		if (scipy.isnan(diff)):
			assert(False)
#		diff = scipy.amax(abs((newV.f - currentV.f) / currentV.f))		
		print("iteration %d diff %f" % (iter, diff))
		if (diff < criterion or (n != None and iter >= n)):
			cont = False
		currentV = newV
		#currentVPrime = newVPrime
		iter += 1
	t2 = time.time()
	print("total time: %f sec, %f per iteration" % (t2-t1, (t2-t1)/iter))
	return (currentV, opt_d, opt_r)
	
# functions for policy iteration

def T(sigma, w):
	"Implements the operator L T_sigma."
	vals = []
	env_interp = LinInterp(grid, map(lambda s: expected_next_v(w, s, z_rv), list(grid)))
	for M in grid:
		#Tw_M = U(M - sigma(M)) + g.beta * mean(w(f(sigma(y), W)))
   		Tw_M = U(M - sigma(M)) + g.beta * env_interp(sigma(M))		
		vals.append(Tw_M)
	return LinInterp(grid, vals)

def get_greedy(w):
	"Computes a w-greedy policy."
	vals = []
	env_interp = LinInterp(grid, map(lambda s: expected_next_v(w, s, z_rv), list(grid)))
	for M in grid:
		exp_util = lambda s: U(M-s) + g.beta * env_interp(s)
		argmax = maximizer(exp_util, 0, M)
		vals.append(argmax)
	return LinInterp(grid, vals)

def get_value(sigma, v):	
	"""Computes an approximation to v_sigma, the value
	of following policy sigma. Function v is a guess.
	"""
	tol = 1e-2         # Error tolerance 
	while 1:
		new_v = T(sigma, v)
		err = max(abs(new_v(grid) - v(grid)))
		if err < tol:
			return new_v            
		v = new_v

def piter_until(sigma, criterion, n=None):
	current_v = v0
	current_vprime = Interp(grid, [0] * len(grid))
	current_sigma = sigma
	iter = 0
	t1 = time.time()
	while 1:
		v_sigma = get_value(current_sigma, current_v)
		greedy_sigma = get_greedy(v_sigma)
		diff = amax(greedy_sigma(grid) - current_sigma(grid))
		current_v = v_sigma
		current_sigma = greedy_sigma
		print("policy iter %i: diff %f" % (iter, diff))
		iter += 1
		
		prob_survive = (1 - z_rv.cdf(-current_sigma(grid)))
		result = (current_v, current_vprime, current_sigma, LinInterp(grid, prob_survive), None)
		g_iterList.append(result)
		
		if (diff < criterion or (n != None and iter >= n)):
			t2 = time.time()
			print("total time: %f sec, %f per iteration" % (t2-t1, (t2-t1)/iter))
			return (current_v, current_vprime, current_sigma, LinInterp(grid, prob_survive))
#########################################################3
# functions that call the value, policy iterations.
# can policy iteration work here?  let's try it. what interface should it have?
#########################################################
			
def initIters():	
	# initial v: use utility fn
	global u0, v0_array, v0, g_iterList
	u0 = U(g.grid_M)
	v0_array = scipy.zeros((len(g.grid_M), len(g.grid_D)))
	for i in range(len(g.grid_D)):
		v0_array[:, i] = u0		
	v0 = Interp2D(g.grid_M, g.grid_D, v0_array)
	# global list of iteration results
	g_iterList = [{}]
	
def addToIter(key, value):
	global g_iterList
	g_iterList[-1][key] = value
def nextIter():
	global g_iterList
	g_iterList.append({})
def resetIters1():
	global g_iterList
	g_iterList = [{}]
	addToIter('v', v0)	
	addToIter('opt_d', None)
	addToIter('opt_r', None)	
def saveIters(filename):
	global g_iterList
	params = g.getGlobalParams()
	output = gzip.open(filename, 'wb')
	pickle.dump((params, g_iterList), output)
	output.close()
def loadIters(filename):
	global g_iterList
	pk_file = gzip.open(filename, 'rb')
	(params, g_iterList) = pickle.load(pk_file)
	g.setGlobalParams(True, **params)
	pk_file.close()
def saveItersHD5(filename):
	global g_iterList
	
	import tables
	
	h5file = tables.openFile(filename, 'w')
	root = h5file.createGroup(h5file.root, "Datasets", "Test data sets")
	datasets = root
	
	h5file.createArray(datasets, "g.grid_M", g.grid_M, "M grid")
	h5file.createArray(datasets, "g.grid_D", g.grid_D, "D grid")
	h5file.createArray(datasets, "n", array([len(g_iterList)]), "number of iterations")
	for i in range(len(g_iterList)):		
		h5file.createArray(datasets, "v_%d" % i, g_iterList[i]['v'].f, "value fn %d" % i)
		fn = g_iterList[i]['opt_d']
		if (fn != None):
			h5file.createArray(datasets, "optd_%d" % i, fn.f, "opt d %d" % i)
		fn = g_iterList[i]['opt_r']	
		if (fn != None):		
			h5file.createArray(datasets, "optr_%r" % i, fn.f, "opt r %d" % i)
	h5file.close()
	
# types of items to store in g_iterList
keys = ['v', 'opt_d', 'opt_r']
titles = {'v':'v', 'opt_d':'opt_d', 'opt_r':'opt_r'}

# for policy iteration
def resetIters2():
	global g_iterList
	g_iterList = [(v0, v0prime, sigma0, Interp(grid, [0] * len(grid))) ]
	
# v, vprime, opt_c, rho
titles = ['v', 'vprime', 'optimal c', 'probability of survival']
def plotIters(i, nlist):
	global g_iterList
	
	plt.figure()
	arglist = []
	for n in nlist:
		arglist.append(grid)
		arglist.append(g_iterList[n][i](grid))
	plt.plot(*arglist)
	plt.title(titles[i])
		
def iterate1(n=None, **kwargs):		
	resetIters1()	
	(v1, d1, r1) = viter_until(v0, 0.001, n, **kwargs)
def resume1(n=None):
	global g_iterList
	(v1, d1, r1) = viter_until(g_iterList[-1]['v'], 0.001, n)
def iterate2(n=None):		
	resetIters2()	
	(v1, v1prime, k1, rho1) = piter_until(sigma0, 0.001, n)
	
def plotLast():
	global g_iterList
	
	plt.close('all')
	n = len(g_iterList)
	for i in range(4):
		plotIters(i, [n-1])
		
# plot iterations
def plotRange(nlist):
	close('all')
	
	for i in range(4):
		plotIters(i, nlist)

####################################################################33
# g_iterList stuff.
# should be a separate module?
		
# 3d surface plot of expected utility as a function of d,r
def plot3D_eu(M, D, w, opt_dr_fns=None, bPrint=0):
	fig = plt.figure()
	ax = Axes3D(fig)
	# input to scatter must be 3 lists of coords (so must repeat)
	f_fn = lambda x: p3_fns.expUtil(w.f, M, D, x[0], x[1], bPrint)	
	ax.scatter(g.meshlist_M2, g.meshlist_r, array(map(f_fn, zip(g.meshlist_M2, g.meshlist_r))), c='c', marker='x')
	
	# highlight points
	if (opt_dr_fns != None):
		i=0
		colorList = ['r', 'g', 'y']
		for (d_fn,r_fn) in opt_dr_fns:
			d = d_fn([M,D])
			r = r_fn([M,D])
			ax.scatter([d], [r], [f_fn([d,r])], c=colorList[i%len(colorList)])
			i += 1
	#ax.plot_wireframe(g.meshlist_M2.reshape(len(g.grid_M), len(g.grid_r)), g.meshlist_r.reshape(len(g.grid_M), len(g.grid_r)), array(map(f_fn, zip(g.meshlist_M2, g.meshlist_r))).reshape(len(g.grid_M), len(g.grid_r)))
	ax.set_xlabel('d')
	ax.set_ylabel('r')
	ax.set_zlabel('utility')

	
def plot3D_eu2(M, D, w):
	f_fn = lambda x: p3_fns.expUtil(w.f, M, D, x[0], x[1])	
	mlab.surf(array(map(f_fn, zip(g.meshlist_M2, g.meshlist_r))).reshape(g.mesh_M2.shape))

def plot3D(M, D, w):
	fig = plt.figure()
	ax = Axes3D(fig)
	f_fn = lambda x: w(x[0], x[1])
	ax.scatter(g.meshlist_M2, g.meshlist_r, array(map(f_fn, zip(g.meshlist_M2, g.meshlist_r))))
	#ax.plot_wireframe(g.meshlist_M2.reshape(len(g.grid_M), len(g.grid_r)), g.meshlist_r.reshape(len(g.grid_M), len(g.grid_r)), array(map(f_fn, zip(g.meshlist_M2, g.meshlist_r))).reshape(len(g.grid_M), len(g.grid_r)))
	ax.set_xlabel('d')
	ax.set_ylabel('r')
	ax.set_zlabel('utility')

def plotOptStep(M,D,n):
	#plot3D_eu(M,D,N, g_iterList[n]['v'], opt_dr_fns=[[g_iterList[n+1]['opt_d'], g_iterList[n+1]['opt_r']], [g_iterList[n+1]['opt_d2'], g_iterList[n+1]['opt_r2']]])
	d1 = g_iterList[n+1]['opt_d']([M,D])
	r1 = g_iterList[n+1]['opt_r']([M,D])
	#d2 = g_iterList[n+1]['opt_d2']([M,D,N])
	#r2 = g_iterList[n+1]['opt_r2']([M,D,N])

	eu1 = p3_fns.expUtil(g_iterList[n]['v'].f, M, D, d1, r1, 1)
	#eu2 = p3_fns.expUtil(g_iterList[n]['v'].f, M, D, N, d2, r2, 1)
	#print("grid: M=%f, D=%f, N=%f, d=%f, r=%f, U=%f" % (M,D,N,d1,r1,eu1))
	print("grid: M=%f, D=%f, d=%f, r=%f, U=%f" % (M,D,d1,r1, eu1))
	  
	plot3D_eu(M,D, g_iterList[n]['v'], opt_dr_fns=[[g_iterList[n+1]['opt_d'], g_iterList[n+1]['opt_r']]])

# set viewpoint for 3d plots
def set3DViewpoint(ax, zlabel):
	elev = 21
	azim = -170.53125
	ax.view_init(elev, azim)
	ax.set_xlabel('M (cash reserve)')
	ax.set_ylabel('D (liabilities)')
	ax.set_zlabel(zlabel)

# figure out what mayavi scalars are for red, green, blue, yellow
from enthought.mayavi.core.lut_manager import LUTManager
class MyColorToMayaviColor():
	def __init__(self):
		self.lm = LUTManager()
		self.lut = self.lm.lut.table.to_array()
		self.colorMap = {}
		
	# euclidean distance
	def distanceFn(self, array1, array2):
		diff = array1[0:3] - array2[0:3]
		return scipy.sqrt(scipy.dot(diff, diff))

	# input is a tuple (r, g, b, a) with values between 0 to 1.0.
	def mayaviColor(self, rgbaColor):
		# memoize.  first check if it's stored
		if (rgbaColor in self.colorMap):
			return self.colorMap[rgbaColor]
		# find the closest color
		current_min_dist = -1
		current_i = 0
		for row_i in range(self.lut.shape[0]):
			dist = self.distanceFn(array(rgbaColor) * 255, self.lut[row_i])
			#print(rgbaColor, self.lut[row_i], dist, current_min_dist, current_i)
			if (current_min_dist == -1 or dist <= current_min_dist):
				current_min_dist = dist
				current_i = row_i
		self.colorMap[rgbaColor] = current_i
		return current_i
		
# surface plot of v
# colorFn takes M,D as arguments, returns a color
def plotSurface(w, zlabel="f", aroundPoint=None, aroundN=4, colorFn=None, colorStates=False, filterFn=None, drawEntireRegion=False, useMayavi=False):
	if (aroundPoint == None):
		(mesh_M, mesh_D) = meshgrid(g.grid_M, g.grid_D)
	else:
		M = aroundPoint[0]
		D = aroundPoint[1]
		(mesh_M, mesh_D) = submeshAroundPoint(M,D,aroundN)
	# meshlist_M and meshlist_D are lists that iterate over coordinates of all possible combinations of M and D
	# zip(meshlist_M, meshlist_D) will return a list of all pairs of (M,D)
	meshlist_M = mesh_M.ravel()
	meshlist_D = mesh_D.ravel()
	fArray = array(map(w, zip(meshlist_M, meshlist_D)))

	# if filterFn is given, only plot points for which filterFn returns True
	if (filterFn != None):
		pointList = filter(lambda x: filterFn(x[0:2]), zip(meshlist_M, meshlist_D, fArray))		
		(meshlist_M, meshlist_D, fArray2) = zip(*pointList)
	else:
		fArray2 = fArray
	colorArray = None
	if (colorStates and colorFn != None):
		colorArray = array(map(colorFn, zip(meshlist_M, meshlist_D)))
	
	if (useMayavi):
		mylab.figure(bgcolor=(0, 0, 0), size=(400, 400))
		colorList = []
		s = None
		if (colorArray != None):
			s = []
			cconv = MyColorToMayaviColor()
			for color in colorArray:
				# map my color to mayavi's colormap
				s.append(cconv.mayaviColor(tuple(color)))
		ax = mylab.points3d(meshlist_M, meshlist_D, fArray2, s, mode='point')
#		mylab.draw()
#		mylab.view(40, 85)
#		mylab.show()
	else:
		fig = plt.figure()
		ax = Axes3D(fig)		
		ax.scatter(meshlist_M, meshlist_D, fArray2, color=colorArray)
		if (drawEntireRegion):
			ax.set_xlim3d(g.grid_M[0], g.grid_M[-1])
			ax.set_ylim3d(g.grid_D[0], g.grid_D[-1])
		# set the viewpoint
		set3DViewpoint(ax, zlabel)
		
	return (scipy.transpose(scipy.reshape(fArray, g.mesh_M.shape)), ax)

def iterColorFn(n):
	classifyStateFn = lambda M,D: classifyStateUsingIter(M,D,n+1)
	colorFn = lambda x: stateToRGBA(classifyStateFn(x[0], x[1]))			
	return colorFn
	
def plotV(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotSurface(g_iterList[n]['v'], 'V', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)

def plotOptD(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotSurface(g_iterList[n]['opt_d'], 'optimal d', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotK(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	opt_d = g_iterList[n]['opt_d']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]
	return plotSurface(k_fn, 'k = d + D - M', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotF(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	opt_d = g_iterList[n]['opt_d']
	opt_r = g_iterList[n]['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p3_fns.testf(k_fn(x), opt_r(x))
	return plotSurface(f_fn, 'f', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
# probability of survival
# x = [M, D]
# f_fn: a function that takes (M,D) as arg, returns f, the fraction invested by depositor
# k_fn: a function that takes (M,D) as arg, returns k = d + D - M
def rhoFn(x, f_fn, k_fn):
	inflow_low = f_fn(x) * g.z_space[0]
	inflow_high = f_fn(x) * g.z_space[1]
	k = k_fn(x)
	if (k < inflow_low):
		return 1;
	if (inflow_low <= k and k < inflow_high):
		return g.z_probs[1]
	else:
		return 0

def plotRho(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	opt_d = g_iterList[n]['opt_d']
	opt_r = g_iterList[n]['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p3_fns.testf(k_fn(x), opt_r(x))
	rho_fn = lambda x: rhoFn(x, f_fn, k_fn)
	return plotSurface(rho_fn, 'rho', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)	
def plotOptR(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	# don't plot red states
	stateArray = getStateArray(n)
	def filterFn(x):
		(iM, iD) = getNearestGridPoint(x[0], x[1])
		if (stateArray[iM, iD] != g.STATE_RED):
			return True
		return False
	return plotSurface(g_iterList[n]['opt_r'], 'optimal r', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), filterFn=filterFn, drawEntireRegion=True, **kwargs)
# calculate next period's M, D contingent on zLow, zHigh
#def getNextMD(M, D, optdFn, optrFn, zState):
def getNextMD(M, D, n, zState):
	optdFn = g_iterList[n]['opt_d']
	optrFn = g_iterList[n]['opt_r']
	# k = d + D - M
	k_fn = lambda x: optdFn(x) + x[1] - x[0]	
	f_fn = lambda x: p3_fns.testf(k_fn(x), optrFn(x))
	inflow = lambda x: f_fn(x) * g.z_space[zState]
	M_fn = lambda x: (inflow(x) - k_fn(x))/g.z_space[zState]
	D_fn = lambda x: optrFn(x) * f_fn(x)
	return (M_fn([M,D]), D_fn([M,D]))

# plot next value of M if zLow occurs	
def plotNextM(zState, n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	M_fn = lambda x: getNextMD(x[0], x[1], n, zState)[0]
	return plotSurface(M_fn, 'next M, state=z[%d]' % zState, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextMlow(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextM(0, n, aroundPoint, aroundN, colorStates=colorStates, **kwargs)
def plotNextMhigh(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextM(1, n, aroundPoint, aroundN, colorStates=colorStates, **kwargs)
	
def plotNextD(state, n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	D_fn = lambda x: getNextMD(x[0], x[1], n, zState)[1]
	return plotSurface(D_fn, 'next D, state=z[%d]' % zState, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextDlow(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextD(0, n, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextDhigh(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextD(1, n, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)

# plot only one state
def plotV_state(n, stateList, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	stateArray = getStateArray(n)
	def filterFn(x):
		(iM, iD) = getNearestGridPoint(x[0], x[1])
		if (stateArray[iM, iD] in stateList):
			return True
		return False
	return plotSurface(g_iterList[n]['v'], 'V', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), filterFn=filterFn, drawEntireRegion=True, **kwargs)

def plotOptD_state(n, stateList, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	stateArray = getStateArray(n)
	def filterFn(x):
		(iM, iD) = getNearestGridPoint(x[0], x[1])
		if (stateArray[iM, iD] in stateList):
			return True
		return False
	return plotSurface(g_iterList[n]['opt_d'], 'optimal d', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), filterFn=filterFn, drawEntireRegion=True, **kwargs)

def plotOptR_state(n, stateList, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	stateArray = getStateArray(n)
	def filterFn(x):
		(iM, iD) = getNearestGridPoint(x[0], x[1])
		if (stateArray[iM, iD] in stateList and stateArray[iM, iD] != g.STATE_RED):
			return True
		return False
	return plotSurface(g_iterList[n]['opt_r'], 'optimal r', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), filterFn=filterFn, drawEntireRegion=True, **kwargs)
	
##########################################################################3
# 3d plotting stuff.  Should be in another module.
#######################################################3

# 3d scalar field plot of a value function
import enthought.mayavi.mlab as mylab
def plot3D_v(v):
	s = v.f
	mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='x_axes',
                            slice_index=10,
                        )
	mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='y_axes',
                            slice_index=10,
                        )
	mlab.outline()

# plot f function of ponzi_fns
def testPlotF():
	fig = plt.figure()
	ax = Axes3D(fig)
	f_fn = lambda x: p3_fns.testf(x[0], x[1])
	grid_k = linspace(0, 2, 50)
	grid_r = linspace(1, 3, 50)
	(mesh_k, mesh_r) = meshgrid(grid_k, grid_r)
	meshlist_k = mesh_k.ravel()
	meshlist_r = mesh_r.ravel()
	
	ax.scatter(meshlist_k, meshlist_r, array(map(f_fn, zip(meshlist_k, meshlist_r))))	
	ax.set_xlabel('k')
	ax.set_ylabel('r')
	ax.set_zlabel('f')

# find nearest grid point
# assumes regular grid
def nearestGridPoint_1d(x, grid):	
	dx = grid[1] - grid[0]
	f = scipy.floor((x-grid[0]) / dx)
	if (f < 0):
		return 0
	if (f >= len(grid)):
		return len(grid) - 1
	d1 = x - grid[f]
	d2 = grid[f+1] - x
	if (d1 < d2):
		return int(f)
	else:
		return int(f+1)
		
def nearestGridPoint(M, D):
	return (nearestGridPoint_1d(M, g.grid_M), nearestGridPoint_1d(D, g.grid_D))
def nearestGridValues(M, D):
	(M1, D1) = nearestGridPoint(M, D)
	return (g.grid_M[M1], g.grid_D[D1])
def submeshCoordsAroundPoint(M, D, n=4):
	(x1, y1) = nearestGridPoint(M, D)
	# display +- 4 points
	xlist = scipy.arange(max(0, x1-n), min(len(g.grid_M), x1+n))
	ylist = scipy.arange(max(0, y1-n), min(len(g.grid_D), y1+n))
	return (xlist, ylist)
def submeshAroundPoint(M, D, n=4):
	(xlist, ylist) = submeshCoordsAroundPoint(M,D,n)
	(mesh_M, mesh_D) = meshgrid(g.grid_M[xlist], g.grid_D[ylist])
	return (mesh_M, mesh_D)

# return gradient w.r.t. M, D
# f is array of values
def gradient_M(f):
	dM = g.grid_M[1] - g.grid_M[0]
	dD = g.grid_D[1] - g.grid_D[0]
	g = scipy.gradient(f, dM, dD)	
	result = []
#	for a1 in g:
#		# add empty row, col
#		newcol = scipy.NaN * scipy.ones((1, len(g.grid_D)-1))
#		a2 = scipy.hstack([a1, newcol])
#		newrow = scipy.NaN * scipy.ones((len(g.grid_M), 1))
#		a3 = scipy.vstack([a2, newrow])
#		result.append(a3)
	return g

# add a row or col of NaN to a 2d array
def addNanRow(a):
	newrow = scipy.NaN * scipy.ones((1, a.shape[1]))
	return scipy.vstack([a, newrow])
def addNanCol(a):
	newcol = scipy.NaN * scipy.ones((a.shape[0], 1))
	return scipy.hstack([a, newcol])
	
def gradient2(f):
	dM = g.grid_M[1] - g.grid_M[0]
	dD = g.grid_D[1] - g.grid_D[0]	
	g1 = scipy.diff(f, 1, 0) / dM
	g2 = scipy.diff(f, 1, 1) / dD
	g3 = addNanRow(g1)
	g4 = addNanCol(g2)
	return [g3, g4]

# what is the temp dir?

# use excel to display a 2d array
def displayArrayInExcel(a):
	import os, tempfile
	from win32com.client import Dispatch
	# write to a csv file
	(fileno, filename) = tempfile.mkstemp(suffix=".csv")
	os.close(fileno)
	scipy.savetxt(filename, a, fmt='%f', delimiter=',')
	# start excel
	xl = Dispatch('Excel.Application')
	wb = xl.Workbooks.Open(filename)
	xl.Visible = 1
def displayFnInExcel(w):
	# add grid values as first row & col
	a1 = scipy.hstack([scipy.reshape(g.grid_M, (len(g.grid_M), 1)), w.f])
	row1 = scipy.reshape(array([0] + g.grid_D.tolist()), (1, len(g.grid_D) + 1))
	a2 = scipy.vstack([row1, a1])
	displayArrayInExcel(a2)

#########################################################################3
# export to excel.  maybe I should centralize the exporting? to mathematica too?
###############################################################################

# classify into "states": A, B, C, D
# A: bankrupt with certainty
# B: bankrupt if zLow occurs
# C: not bankrupt with certainty, keep next M just above zero if zLow occurs
# D: not bankrupt with certainty, d=M (withdraw as much as possible, not concerned about inflows)
# inputs: need M, D, a fn that gives rho, a fn that gives opt_d
# returns: 0, 1, 2, 3
g.STATE_RED = 0
g.STATE_BLUE = 1
g.STATE_GREEN = 2
g.STATE_YELLOW = 3
g.STATE_BLACK = 4
g.States = [g.STATE_RED, g.STATE_BLUE, g.STATE_GREEN, g.STATE_YELLOW, g.STATE_BLACK]
g.StateColors = {g.STATE_RED: 'r', g.STATE_BLUE: 'b', g.STATE_GREEN: 'g', g.STATE_YELLOW: 'y', g.STATE_BLACK: 'b'}

#################################################33
# not sure where this states and colors stuff should go
######################################################33

def classifyState(M, D, rhoFn, optdFn):
	rho = rhoFn([M, D])
	if (rho == 0):
		return g.STATE_RED
	if (rho < 1):
		return g.STATE_BLUE
	if (optdFn([M,D]) == M):
		return g.STATE_YELLOW
	return g.STATE_GREEN

def classifyStateUsingIter(M, D, n):
	opt_d = g_iterList[n]['opt_d']
	opt_r = g_iterList[n]['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p3_fns.testf(k_fn(x), opt_r(x))
	rho_fn = lambda x: rhoFn(x, f_fn, k_fn)
	return classifyState(M,D, rho_fn, opt_d)
	
def getStateArray(n):
	classifyStateFn = lambda M,D: classifyStateUsingIter(M,D,n)
	stateArray = scipy.vectorize(classifyStateFn)(g.mesh_M, g.mesh_D)
	return scipy.transpose(stateArray)

# return a dict of lists
def partitionStates(n):
	stateArray = getStateArray(n)
	partition = defaultdict(list)
	# make lists consisting of state=0, 1, 2, 3, 4
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			state = stateArray[iM, iD]
			partition[state].append((iM, iD))
	return partition
	
def getNextStateArray(n, zState):
	opt_d = g_iterList[n]['opt_d']
	opt_r = g_iterList[n]['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p3_fns.testf(k_fn(x), opt_r(x))
	rho_fn = lambda x: rhoFn(x, f_fn, k_fn)

	def nextStateFn(M,D):
		(nextM, nextD) = getNextMD(M, D, n, zState)
		# if current state is A (bankrupt with certainty), then next state has no meaning.  color it black
		currentState = classifyState(M, D, rho_fn, opt_d)
		if (currentState == g.STATE_RED):
			return g.STATE_BLACK
		# if current state is B (bankrupt if zLow occurs) and zLow occurs, next state is in bankruptcy
		if (currentState == g.STATE_BLUE and zState == 0):
			return g.STATE_BLACK
		return classifyState(nextM, nextD, rho_fn, opt_d)
	
	nextStateArray = scipy.vectorize(nextStateFn)(g.mesh_M, g.mesh_D)
	return scipy.transpose(nextStateArray)	

# map state to color
def stateToColor(state):
	return g.StateColors[state]
cconverter = matplotlib.colors.ColorConverter()
def stateToRGBA(state):
	return cconverter.to_rgba(stateToColor(state))
	
# arg: a 2d array, values can be 0, 1, 2, 3, 4
# state 4 is currently bankrupt
def plotStates(stateArray, title=None):	
	partitionM = {}
	partitionD = {}
	for state in g.States:
		partitionM[state] = []
		partitionD[state] = []
	# make lists consisting of state=0, 1, 2, 3, 4
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			state = stateArray[iM, iD]
			partitionM[state].append(M)
			partitionD[state].append(D)
		
	fig = plt.figure()
	ax = fig.add_subplot(111, title=title)	
	for state in g.States:
		if (len(partitionD[state]) > 0):
			plt.scatter(partitionD[state], partitionM[state], c=stateToColor(state))
	ax.set_xlabel('D')
	ax.set_ylabel('M')
		
def plotIterStates(n):
	stateArray = getStateArray(n)
	plotStates(stateArray)

def plotIterNextStates(n, zState):
	stateArray = getNextStateArray(n, zState)
	plotStates(stateArray, title='next state | z[%d]' % zState)

# plot the mapping from current (M,D) to next period's optimally chosen (M,D) conditional on z[0] or z[1] occurring
def plotMappingToNextState(n, zState, currentState, iMList=None, iDList=None, colorByM=True):
	fig = plt.figure()
	ax = Axes3D(fig)
	zlabel = "t"
	stateArray = getStateArray(n)
	nextStateArray = getNextStateArray(n, zState)
	
	colors = ['r', 'b', 'g', 'y', 'k']
	iColor = 0
	iMDict = {}
	iDDict = {}
	if (iMList != None):
		for iM in iMList:
			iMDict[iM] = 1
	if (iDList != None):
		for iD in iDList:
			iDDict[iD] = 1
		
	for (iM, M) in enumerate(g.grid_M):
		if (colorByM):
			iColor = iM % len(colors)
		for (iD, D) in enumerate(g.grid_D):	
			if (not colorByM):
				iColor = iD % len(colors)		
			(nextM, nextD) = getNextMD(M, D, n, zState)
			if (currentState == stateArray[iM, iD]):
				if ((iMList == None or (iM in iMDict)) and (iDList == None or (iD in iDDict))):	
					iColor = nextStateArray[iM, iD]
					ax.plot([M, nextM], [D, nextD], [0, 1], c=colors[iColor])
					
	# set the viewpoint
	set3DViewpoint(ax, zlabel)

class MyNode():
    def __init__(self, iM, M, iD, D):
		self.iM = iM
		self.iD = iD		
		self.M = M
		self.D = D		

# get node from iM, iD
def getNode(iM, iD):
	global g_NodeArray
	return g_NodeArray[iM, iD]

# assume a regular grid
def getNearestGridPoint1D(x, grid):	
	dx = grid[1] - grid[0]
	if (x <= grid[0]):
		return 0
	if (x >= grid[-1]):
		return len(grid)-1
	i = (x - grid[0]) / dx
	i1 = int(np.floor(i))
	i2 = int(np.ceil(i))
	if (x - grid[i1] < grid[i2] - x):
		return i1
	else:
		return i2
		
# return index of nearest grid point
def getNearestGridPoint(M, D):
	iM = getNearestGridPoint1D(M, g.grid_M)
	iD = getNearestGridPoint1D(D, g.grid_D)
	return (iM, iD)

g.BANKRUPT_NODE = 0	
def createGraph(n):
	global g_NodeArray
	g_NodeArray = [[None]*g.gridsize_D]*g.gridsize_M
	stateArray = getStateArray(n)
	opt_d = g_iterList[n]['opt_d']
	opt_r = g_iterList[n]['opt_r']

	G = nx.DiGraph()
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			#newNode = MyNode(iM, M, iD, D)
			#g_NodeArray[iM][iD] = newNode
			G.add_node((iM,iD), M=M, D=D, state=stateArray[iM, iD], opt_d=opt_d([M,D]), opt_r=opt_r([M,D]))
	# bankruptcy node is 0
	G.add_node(g.BANKRUPT_NODE, state=g.STATE_BLACK)
	G.add_edge(g.BANKRUPT_NODE, g.BANKRUPT_NODE)
		
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			(nextMLow, nextDLow) = getNextMD(M, D, n, 0)
			(nextMHigh, nextDHigh) = getNextMD(M, D, n, 1)
			if (nextMLow < 0):
				#G.add_edge(g_NodeArray[iM][iD], g.BANKRUPT_NODE, zState=0)
				G.add_edge((iM,iD), g.BANKRUPT_NODE, zState=0)
			else:
				(next_iMLow, next_iDLow) = getNearestGridPoint(nextMLow, nextDLow)
				#G.add_edge(g_NodeArray[iM][iD], g_NodeArray[next_iMLow][next_iDLow], zState=0)
				G.add_edge((iM,iD), (next_iMLow, next_iDLow), zState=0)
			if (nextMHigh < 0):
				#G.add_edge(g_NodeArray[iM][iD], g.BANKRUPT_NODE, zState=1)
				G.add_edge((iM, iD), g.BANKRUPT_NODE, zState=1)
			else:
				(next_iMHigh, next_iDHigh) = getNearestGridPoint(nextMHigh, nextDHigh)			
				#G.add_edge(g_NodeArray[iM][iD], g_NodeArray[next_iMHigh][next_iDHigh], zState=1)
				G.add_edge((iM, iD), (next_iMHigh, next_iDHigh), zState=1)
	return G

# plot a 2d graph of nodes that satisfy some predicate
# predicateFn(graph, node) returns True or False	
def plotGraphNodes(G, predicateFn, title=None):	
	fig = plt.figure()
	ax = fig.add_subplot(111, title=title)	
	MByState = defaultdict(list)
	DByState = defaultdict(list)	
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			if (predicateFn(G, (iM, iD))):
				MByState[G.node[(iM, iD)]['state']].append(M)
				DByState[G.node[(iM, iD)]['state']].append(D)				
	for state in MByState.keys():
		plt.scatter(DByState[state], MByState[state], c=stateToColor(state))	
	ax.set_xlabel('D')
	ax.set_ylabel('M')

# mark nodes in a graph that will be bankrupt with certainty next period.	
def markNodesLeadingToBankruptcy(G):
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			if (G.successors((iM, iD)) == [g.BANKRUPT_NODE]):
				G[(iM,iD)]['bankruptWithCertainty'] = True
			else:
				G[(iM,iD)]['bankruptWithCertainty'] = False
# returns a set.
def nodesLeadingToBankruptcy(G):
	result = {}
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			if (G.successors((iM, iD)) == [g.BANKRUPT_NODE]):
				result[(iM,iD)] = 1
	return set(result)
	
def plotNodesLeadingToBankruptcy(G):
	S = nodesLeadingToBankruptcy(G)
	plotGraphNodes(G, lambda G,n: n in S, "bankrupt with certainty")

# find all nodes in graph G for which all paths will reach subset S. nodes in S are not necessarily included
# S is a sequence of nodes
# returns a set
def backwardsSearch(G, S):
	currentNodeSet = set()
	S_set = {}
	for node in S:
		S_set.add(node)
	bDone = False
	while (not bDone):
		nAddedThisRound = 0
		allPredecessors = set()
		for node in currentNodeSet | S:
			pre = G.predecessors(node)
			for n in pre:
				# don't need to check nodes that are already in the set
				if (not n in currentNodeDict):
					allPredecessors.add(n)
		#print("allPredecessors %d" % len(allPredecessors))
		for node in allPredecessors:
			# check if all successors are in the set
			nSuccCount = 0
			successors = G.successors(node)
			for n in successors:
				if (n in currentNodeSet or n in S_set):
					nSuccCount += 1
			# if so, add it to the set
			if (nSuccCount == len(successors)):
				currentNode.add(node)
				nAddedThisRound += 1
		#print("added this round: %d" % nAddedThisRound)
		if (nAddedThisRound == 0):
			bDone = True
	return currentNodeSet

# find all nodes reachable from a subset S, not necessarily including S
def findNodesReachableFrom(G, S):
	result = set()
	for s,t in my_bfs.my_bfs_edges(G,S):
		result.add(t)
	return result

def findNodesReverseReachableFrom(G, S):
	reverseG = G.reverse()
	return findNodesReachableFrom(reverseG, S)
	
# find B nodes that are part of a cycle.
# we'll take advantage of the fact that most nodes have no parents.  reverse the graph and see if each individual node can reach itself
def findBCycles(G):
	reverseG = G.reverse()
	blueNodes = set([n for n in reverseG.nodes() if (reverseG.node[n]['state'] == g.STATE_BLUE)])
	result = set([])
	for node in blueNodes:
		reachable = findNodesReachableFrom(reverseG, [node])
		if node in reachable:
			result.add(node)
	return result

# delete nodes that have indegree 0 until there are none left
def pruneZeroIndegree(G):
	g = G.copy()
	bDone = False
	while (not bDone):
		toDelete = []		
		for node in g.nodes():
			if (g.in_degree(node) == 0):
				toDelete.append(node)
		g.remove_nodes_from(toDelete)
		if (len(toDelete) == 0):
			bDone = True
	return set(g.nodes())

# given a set of transient nodes, calculate the expected time to bankruptcy (i.e. the expected hitting time to the bankruptcy absorbing state)
# returns a list of pairs; [(node, expected_time)...]
def calcExpectedTimeToBankruptcy(G, S):
	# make a list of nodes, sort by D, then M.
	def compareNodes(node1, node2):
		(M1, D1) = node1
		(M2, D2) = node2
		if (D1 < D2):
			return -1
		if (D1 > D2):
			return 1
		if (M1 < M2):
			return -1
		if (M1 > M2):
			return 1
		return 0

	nodes = sorted(list(S), cmp=compareNodes)
	k = len(nodes)
	# fill in the transition matrix
	Q = scipy.zeros((k, k))
	for (i_state1, state1) in enumerate(nodes):
		for (i_state2, state2) in enumerate(nodes):
			out_edges = G.out_edges(state1, data=True)
			for (source, dest, attribs) in out_edges:
				if (source == state1 and dest == state2):
					zState = attribs['zState']
					if (zState == 0):
						Q[i_state1, i_state2] = 1 - g.pHigh
					elif (zState == 1):
						Q[i_state1, i_state2] = g.pHigh
					else:
						assert(False, "unknown zState")
	try:
		m = scipy.dot(linalg.inv(scipy.eye(k) - Q), scipy.ones((k, 1)))
		result = zip(nodes, list(m.flatten()))
	except scipy.linalg.LinAlgError:
		# matrix is singular
		result = zip(nodes, [scipy.NaN] * len(nodes))
	return (Q, result)

def plotNodesBankruptInBoundedTime(G):
	S = backwardsSearch(G, [g.BANKRUPT_NODE])
	plotGraphNodes(G, lambda G,n: n in S, "bankrupt in bounded time")
	
def plotNodesThatCanReachBCycle(G, complement=False):
	S = findBCycles(G)
	T = findNodesReverseReachableFrom(G, S)
	if (complement):
		T = set(G.nodes()) - T
	plotNodeSet(G, T, "nodes that can reach recurring B states")
	
# plot a subset of nodes.
def plotNodeSet(G, S, title=None):
	plotGraphNodes(G, lambda G,n: n in S, title)
	
def test1(stateArray, nextStateArray):
	result = []
	for (iM, M) in enumerate(g.grid_M):
		for (iD, D) in enumerate(g.grid_D):
			if (stateArray[iM, iD] == 1 and nextStateArray[iM, iD] == 0):
				result.append((iM, iD))
	return result

##############################################################3
# graph stuff should be in a separate module.
##############################################################3

pHighList = [0.95, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
pHighFilenameFormat = "iters-pHigh-%f"
# run through different values of pHigh, iterate, save output
def test_pHigh():	
	testParams("pHigh", pHighList, pHighFilenameFormat)
def write_graphs_pHigh(outputDirname):
	writeGraphs(outputDirname, "pHigh", "pHigh", pHighList, pHighFilenameFormat)
	
betaList = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]		
betaFilenameFormat = "iters-beta-%f"
# run through different values of beta
def test_beta():	
	testParams("beta", betaList, betaFilenameFormat)
def write_graphs_beta(outputDirname):
	writeGraphs(outputDirname, "beta", "beta", betaList, betaFilenameFormat)

#mList = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]		
mList = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7]		
mFilenameFormat = "iters-m-%f"
def test_m():	
	testParams("depositorSlope", mList, mFilenameFormat)
def write_graphs_m(outputDirname):
	writeGraphs(outputDirname, "depositor slope", "depositorSlope", mList, mFilenameFormat)

zLowList = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]		
zLowFilenameFormat = "iters-zLow-%f"
def test_zLow():	
	testParams("zLow", zLowList, zLowFilenameFormat)
def write_graphs_zLow(outputDirname):
	writeGraphs(outputDirname, "zLow", "zLow", zLowList, zLowFilenameFormat)

zHighList = [1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]		
zHighFilenameFormat = "iters-zHigh-%f"
def test_zHigh():	
	testParams("zHigh", zHighList, zHighFilenameFormat)
def write_graphs_zHigh(outputDirname):
	writeGraphs(outputDirname, "zHigh", "zHigh", zHighList, zHighFilenameFormat)

bpList = [0.0, -0.1, -0.2, -0.5, -0.7, -1.0]		
#bpList = [-0.5, -0.7, -1.0]		
bpFilenameFormat = "iters-bp-%f"
def test_bp():	
	testParams("bankruptcyPenalty", bpList, bpFilenameFormat)
def write_graphs_bp(outputDirname):
	writeGraphs(outputDirname, "bankruptcy penalty", "bankruptcyPenalty", bpList, bpFilenameFormat)

# check if parameters lead to a contraction mapping.  if E(z) < beta, then it should be optimal to consume some and not just save everything in every period	
def checkParamsConvergence(**kwargs):
	beta = kwargs['beta']
	zLow = kwargs['zLow']
	zHigh = kwargs['zHigh']
	pHigh = kwargs['pHigh']
	return ((beta * (zLow * (1-pHigh) + zHigh * pHigh)) < 0.95)

from prettytable import *
import markup
def testAll(doIterate=True, doGraphs=True, doHittingTimes=True):
	# benchmark parameters
	kwargs = {}
	kwargs['beta'] = 0.8
	kwargs['zHigh'] = 1.2
	kwargs['zLow'] = 0.8
	kwargs['pHigh'] = 0.9
			
	def predicate(paramVal):
		kwargs2 = kwargs.copy()
		return checkParamsConvergence(**kwargs2)				
				
	paramList = pHighList
	filteredList = filter(predicate, paramList)
	if (doIterate): testParams("pHigh", filteredList, pHighFilenameFormat, initGlobalParams=True, maxIters=200, baseParams=kwargs)
	if (doGraphs): writeGraphs("pHigh", "pHigh", "pHigh", filteredList, pHighFilenameFormat)
	if (doHittingTimes): calcHittingTimes("pHigh", "pHigh", filteredList, pHighFilenameFormat, baseParams=kwargs)
	
	paramList = betaList
	filteredList = filter(predicate, paramList)	
	if (doIterate): testParams("beta", filteredList, betaFilenameFormat, initGlobalParams=True, maxIters=200, baseParams=kwargs)
	if (doGraphs): writeGraphs("beta", "beta", "beta", filteredList, betaFilenameFormat)
	if (doHittingTimes): calcHittingTimes("beta", "beta", filteredList, betaFilenameFormat, baseParams=kwargs)
	
	paramList = mList
	filteredList = filter(predicate, paramList)	
	if (doIterate): testParams("depositorSlope", mList, mFilenameFormat, initGlobalParams=True, maxIters=200, baseParams=kwargs)
	if (doGraphs): writeGraphs("m", "depositorSlope", "depositorSlope", filteredList, mFilenameFormat)
	if (doHittingTimes): calcHittingTimes("m", "depositorSlope", filteredList, mFilenameFormat, baseParams=kwargs)
	
	paramList = zLowList
	filteredList = filter(predicate, paramList)		
	if (doIterate): testParams("zLow", zLowList, zLowFilenameFormat, initGlobalParams=True, maxIters=200, baseParams=kwargs)
	if (doGraphs): writeGraphs("zLow", "zLow", "zLow", filteredList, zLowFilenameFormat)
	if (doHittingTimes): calcHittingTimes("zLow", "zLow", filteredList, zLowFilenameFormat, baseParams=kwargs)
	
	paramList = zHighList
	filteredList = filter(predicate, paramList)	
	if (doIterate): testParams("zHigh", zHighList, zHighFilenameFormat, initGlobalParams=True, maxIters=200, baseParams=kwargs)
	if (doGraphs): writeGraphs("zHigh", "zHigh", "zHigh", filteredList, zHighFilenameFormat)
	if (doHittingTimes): calcHittingTimes("zHigh", "zHigh", filteredList, zHighFilenameFormat, baseParams=kwargs)
	
	paramList = bpList
	filteredList = filter(predicate, paramList)
	if (doIterate): testParams("bankruptcyPenalty", bpList, bpFilenameFormat, initGlobalParams=True, maxIters=200, baseParams=kwargs)
	if (doGraphs): writeGraphs("bp", "bankruptcyPenalty", "bankruptcyPenalty", filteredList, bpFilenameFormat)
	if (doHittingTimes): calcHittingTimes("bp", "bp", filteredList, bpFilenameFormat, baseParams=kwargs)
	
def testParams(paramName, paramList, filenameFormat, initGlobalParams=True, maxIters=None, baseParams={}):
	itemList = []
	kwargs = baseParams.copy()
	for paramVal in paramList:
		kwargs[paramName] = paramVal		
		g.setGlobalParams(initGlobalParams, **kwargs)		
		#print("%s=%f" % (paramName, paramVal))
		print(kwargs)
		iterate1(n=maxIters)
		filename = (filenameFormat % paramVal) + ".out" 
		saveIters(filename)

# run through 2 params: we'll vary zHigh and beta
zHighList = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7]
betaList = [0.9, 0.8, 0.7, 0.6, 0.5]		
filenameFormat = "iters-beta-%f-zHigh-%f"
def test2Params():
	for zHigh in zHighList:
		for beta in betaList:			
			g.setGlobalParams(True, beta=0.7, zHigh=1.4, zLow=0.9);		# this is the baseline
			g.setGlobalParams(False, zHigh=zHigh, beta=beta)		
			print("zHigh=%f beta=%f" % (zHigh, beta))
			iterate1(useNew=False, parallel=True)
			filename = (filenameFormat % (zHigh, beta)) + ".out" 
			saveIters(filename)
			g = createGraph(len(g_iterList) - 1)
			S = pruneZeroIndegree(g)
			if (g.BANKRUPT_NODE in S):
				S.remove(g.BANKRUPT_NODE)
			if (len(S) == 0):
				print("no nodes in cyclic paths")
			else:
				expHittingTimes = calcExpectedTimeToBankruptcy(g, S)
				print(expHittingTimes)
				
def test2ParamsSummary():
	table = {}
	for zHigh in zHighList:
		for beta in betaList:			
			g.setGlobalParams(True, beta=0.7, zHigh=1.4, zLow=0.9);		# this is the baseline
			g.setGlobalParams(False, zHigh=zHigh, beta=beta)		
			print("zHigh=%f beta=%f" % (zHigh, beta))			
			filename = (filenameFormat % (zHigh, beta)) + ".out" 
			loadIters(filename)
			g = createGraph(len(g_iterList) - 1)
			S = pruneZeroIndegree(g)
			if (g.BANKRUPT_NODE in S):
				S.remove(g.BANKRUPT_NODE)
			if (len(S) == 0):
				print("no nodes in cyclic paths")
			else:
				expHittingTimes = calcExpectedTimeToBankruptcy(g, S)
				print(expHittingTimes)
				(maxNode, maxTime) = (None, None)
				# find the blue node with the largest expected hitting time
				for (node, expHittingTime) in expHittingTimes:
					if ((maxTime == None or expHittingTime > maxTime) and g.node[node]['state'] == g.STATE_BLUE):
						(maxNode, maxTime) = (node, expHittingTime)
				if (not maxNode == None):
					opt_r = g_iterList[len(g_iterList) - 1]['opt_r']
					(M, D) = (g.grid_M[maxNode[0]], g.grid_D[maxNode[1]])
					table[(zHigh, beta)] = (maxNode, (M, D), maxTime, g.node[maxNode]['state'], opt_r([M, D]))

	for beta in betaList:			
		print beta,
	print					
	for zHigh in zHighList:
		for beta in betaList:				
			if ((zHigh, beta) in table):
				(node, (M, D), hittingTime, state, r) = table[(zHigh, beta)]
				print r,
			else:
				print "-",
		print
		for beta in betaList:				
			if ((zHigh, beta) in table):
				(node, (M, D), hittingTime, state, r) = table[(zHigh, beta)]
				print hittingTime,
			else:
				print "-",
		print
	return table
	
# run through different values of pHigh, iterate, save output
def test_pHigh2():	
	testParams("pHigh", pHighList, pHighFilenameFormat)
def write_graphs_pHigh2(outputDirname):
	writeGraphs(outputDirname, "pHigh", "pHigh", pHighList, pHighFilenameFormat)
	
betaList = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]		
betaFilenameFormat = "iters-beta-%f"
# run through different values of beta
def test_beta2():	
	testParams("beta", betaList, betaFilenameFormat)
def write_graphs_beta2(outputDirname):
	writeGraphs(outputDirname, "beta", "beta", betaList, betaFilenameFormat)

def calcHittingTimes(outputDirname, paramName, paramList, filenameFormat, baseParams):	
	def colorBullet(state):
		hexcolor = matplotlib.colors.rgb2hex(stateToRGBA(state)[0:3])
		return '<FONT COLOR="%s">&bull;</FONT>' % hexcolor
		
	if (not os.path.isdir(outputDirname)):
		os.mkdir(outputDirname)		

	filename = os.path.join(outputDirname, "bankruptcyTime.html")
	
	page = markup.page()
	page.init(title=paramName)	

	opt_r_list = []
	exp_time_list = []
	
	for paramVal in paramList:
		kwargs = baseParams.copy()
		kwargs[paramName] = paramVal			
		print(kwargs)
		g.setGlobalParams(True, **kwargs)		
		loadIters((filenameFormat % paramVal) + ".out")

		graph = createGraph(len(g_iterList) - 1)
		print("%s = %f" % (paramName, paramVal))
		page.p("%s = %f" % (paramName, paramVal))
		
		# go through all blue states and get the range of opt_r
		opt_r_range = set()
		for node in graph.nodes():
			if (graph.node[node]['state'] == g.STATE_BLUE):
				opt_r = graph.node[node]['opt_r']
				opt_r_range.add(opt_r)
		
		S = pruneZeroIndegree(graph)
		if (g.BANKRUPT_NODE in S):
			S.remove(g.BANKRUPT_NODE)

		# print out r range
		opt_r_range_list = sorted(list(opt_r_range))
		if (len(opt_r_range_list) > 0):
			print("opt_r range: %f, %f" % (opt_r_range_list[0], opt_r_range_list[-1]))
		else:
			print("opt_r range empty")
		opt_r_range_sorted = sorted(list(opt_r_range))
		print(opt_r_range_sorted)		
		if (len(opt_r_range_list) > 0):
			page.p("opt_r range: %f, %f" % (opt_r_range_list[0], opt_r_range_list[-1]))
			opt_r_list.append((paramVal, opt_r_range_sorted[0]))
		else:
			page.p("opt_r range empty")
			opt_r_list.append((paramVal, scipy.NaN))
				
		# print transition matrix and expected hitting times		
		if (len(S) == 0):
			print("no nodes in cyclic paths")
			page.p("no nodes in cyclic paths")
			exp_time_list.append((paramVal, None))
			print("max exp time: None")
		else:
			(transitionMatrix, expHittingTimes) = calcExpectedTimeToBankruptcy(graph, S)
			# nodeList = sorted(expHittingTimes, cmp=compareNodes)
			nodeStrList = []
			for (node, expTime) in expHittingTimes:
				(iM, iD) = node
				nodeStrList.append("%.3f, %.3f" % (g.grid_M[iM], g.grid_D[iD]))										
			table = PrettyTable([""] + nodeStrList + ["expected hitting time"])
			for (i, nodeStr) in enumerate(nodeStrList):
				table.add_row([colorBullet(graph.node[node]['state']) + nodeStr] + list(transitionMatrix[i]) + [expHittingTimes[i][1]])
			print(table)
			print(expHittingTimes)			
			page.add(table.get_html_string())
			max_exp_time = scipy.amax([x[1] for x in expHittingTimes])
			exp_time_list.append((paramVal, max_exp_time))
			print("max exp time: %f" % max_exp_time)

	print("opt_r summary: ",)
	print(opt_r_list)
	page.p("opt_r summary: ")
	for (paramVal, opt_r) in opt_r_list:
		page.p("%f\t%f" % (paramVal, opt_r))	

	print("exp_time summary: ",)
	print(exp_time_list)
	page.p("exp_time summary: ")
	for (paramVal, exp_time) in exp_time_list:
		if (exp_time == None):
			page.p("%f\tNone" % paramVal)
		else:
			page.p("%f\t%f" % (paramVal, exp_time))
		
	f = open(filename, 'w')
	f.write(str(page))
	f.close()
	
		
def writeGraphs(outputDirname, title, paramName, paramList, filenameFormat):
	import markup
	
	if (not os.path.isdir(outputDirname)):
		os.mkdir(outputDirname)		
	plt.ioff()
	itemList = []
	for paramVal in paramList:				
		kwargs = {paramName: paramVal}
		g.setGlobalParams(True, **kwargs)		
		loadIters((filenameFormat % paramVal) + ".out")
		imgList = []
		
		plotV(-2, colorStates=True)
		filename = (filenameFormat % paramVal) + "-V.png"
		path = os.path.join(outputDirname, filename)
		try:
			os.remove(path)
		except OSError:
			pass	
		print(path)
		plt.savefig(path, format='png', dpi=60)
		plt.close()
		imgList.append(filename)
		
		plotOptD(-2, colorStates=True)
		filename = (filenameFormat % paramVal) + "-optD.png"
		path = os.path.join(outputDirname, filename)
		try:
			os.remove(path)
		except OSError:
			pass		
		print(path)
		plt.savefig(path, format='png', dpi=60)
		plt.close()
		imgList.append(filename)		

		plotOptR(-2, colorStates=True)
		filename = (filenameFormat % paramVal) + "-optR.png"
		path = os.path.join(outputDirname, filename)
		try:
			os.remove(path)
		except OSError:
			pass	
		print(path)
		plt.savefig(path, format='png', dpi=60)
		plt.close()
		imgList.append(filename)

		plotNextM(1, -2, colorStates=True)
		filename = (filenameFormat % paramVal) + "-nextM.png"
		path = os.path.join(outputDirname, filename)
		try:
			os.remove(path)
		except OSError:
			pass	
		print(path)
		plt.savefig(path, format='png', dpi=60)
		plt.close()
		imgList.append(filename)
		
		itemList.append(("%s=%f" % (paramName, paramVal), imgList))
	plt.ion()			

	page = markup.page()
	page.init(title=title)
	page.br( )
	
	for item in itemList:
		(caption, imgList) = item
		page.p(caption)
		for imageName in imgList:
			page.img(src=imageName)
	filename = os.path.join(outputDirname, "index.html")
	f = open(filename, 'w')
	f.write(str(page))
	f.close()
######################################################################3
# testing and generating the output.
# 	
		
# init global parameters, c library
# default values are: beta=0.9, m=0.1, bankruptcyPenalty=0, zLow=1.0, zHigh=1.2, pHigh=0.95
#g.setGlobalParams(True)
g.setGlobalParams(True, beta=0.8, zHigh=1.2, zLow=0.8, pHigh=0.9)
# zLow's effect... ?
# beta's effect is to make blue "vertical", i.e. bank becomes impatient, so would rather take more out of M and go to blue, rather than have a higher M and go to green

initIters()


