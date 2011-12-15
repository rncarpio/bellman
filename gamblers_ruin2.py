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
from scipy import linspace, mean, exp, randn, stats, interpolate, integrate
from scipy.optimize import fminbound, tnc
from lininterp import LinInterp
from pylab import *
from IPython.Debugger import Tracer; debug_here = Tracer()
import copy, time

# interpolation class to use
Interp = LinInterp
#Interp = scipy.interpolate.UnivariateSpline

# Parameters
beta = 0.9
theta = 0.5			
gamma = 2.0

# Utility
# exponential utility
def U_exponential(c): return 1 - exp(- theta * c) 
def Uprime_exponential(c): return theta*exp(- theta * c) 
# linear utility
def U_linear(c): return c
def Uprime_linear(c): return (c - c + 1)
# CRRA
def U_crra(c): return c**(1-gamma) / (1-gamma)
def Uprime_crra(c): return c**(-gamma)

U = U_linear
Uprime = Uprime_linear
#U = U_exponential
#Uprime = Uprime_exponential
#U = U_crra
#Uprime = Uprime_crra

# shock (income)
#z_rv = scipy.stats.uniform(loc=-0.5, scale=1.5)
z_rv = scipy.stats.norm(loc=0.5, scale=1)
z_states = [-1, 1]
pHigh = 0.99
z_probs = [1-pHigh, pHigh]
# check probs add up to 1
if (sum(z_probs) != 1.0):
	raise ValueError('probs should add up to 1.0')	
# z_rv = scipy.stats.rv_discrete(name='income', values=(z_states, z_probs))
# z_rv.z_states = z_states
# z_rv.z_probs = z_probs

z_draws = z_rv.rvs(size=1000)                     
# CDF of z
#def z_cdf(x):
#	return z_rv.cdf(x)

gridmin = 0	
gridmax = 10
gridsize = 1000
grid = linspace(gridmin, gridmax, gridsize)
gridspace = grid[1] - grid[0]
# how to do log-spaced grid points?
#grid = linspace(0, gridmax**1e-1, gridsize)**10
	
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

def maximizer3(h, a, b):
	result = scipy.optimize.fmin_tnc(lambda x: -h(x[0]), [(a+b)/2.0], bounds=[(a,b)], approx_grad=True, messages = scipy.optimize.tnc.MSG_NONE)
	#debug_here()
	return result[0][0]

# if the range is small enough, call optimize.brute
def maximizer4(h, a, b):
	n_points = 20
	if ((a + n_points*gridspace) > b):
		return maximizer1(h, a, b)
	return scipy.optimize.brute(lambda x: -h(x), [(a,b)], Ns=n_points, finish=None)

# grid search	
def maximizer5(h, a, b):
	g2 = linspace(a, b, gridsize)
	h_array = h(g2)
	max_i = scipy.argmax(h_array)
	result = g2[max_i]
	if (h(a) > h(result) and h(a) > h(b)):
		return a
	if (h(b) > h(result) and h(b) > h(a)):
		return b
	return result
	
maximizer = maximizer5

# calculate v_{t+1} conditional on z > (-s), 0 otherwise
# w should be a vectorized fn, s scalar
# not vectorized on s, since s + z_draws won't match.
def next_v(w, s):
	nextM = s + z_draws
	val1 = w(nextM)
	val2 = 0
	return scipy.where(nextM > 0, val1, val2)

# v'(M)	
def expUtilPrime(w, wPrime, M, opt_s, env_interp, envprime_interp):
	# all consumption, no savings -> use U'(c)
	if (opt_s == 0):
		return Uprime(M)
	# all savings, no consumption -> use  beta (P(s+z>0) E[w'(s,z)] + dP E[w]]
#	if (opt_s == M):
		#return beta * mean(next_v(wPrime, M))
		#return beta * (envprime_interp(M) + z_rv.pdf(-M)*wPrime(0))
#		return beta * (envprime_interp(M) + z_rv.pdf(-M)*w(0))
	# in between
	return Uprime(M - opt_s)

# calculate E[w(s+z)], with w() =0 if s+z < 0.
# use the grid directly.
# make sure we're not being passed a vector!
def expected_next_v_continuous(w, s, rv):
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

# for discrete RVs.
def expected_next_v_discrete(w, s, rv):
	assert(grid[0] >= 0)
	assert(scipy.size(s) == 1)
	sum = 0.0
	for i in range(len(rv.z_states)):
		z_i = rv.z_states[i]
		z_i_prob = rv.z_probs[i]
		next_s = s + z_i
		if (next_s >= 0):
		  sum += z_i_prob * w(next_s)
	return sum
expected_next_v	= expected_next_v_continuous
	
# calculate E[w(s+z) given s+z > 0]
def expected_next_v2(w, s, rv):
	assert(grid[0] >= 0)
	assert(scipy.size(s) == 1)
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
	
# returns (Tw, (Tw)', optimal_s, prob of survival)	
def bellman1(w, wPrime):
	
	vals = scipy.zeros(len(grid))
	vprime = scipy.zeros(len(grid))
	opt_controls = scipy.zeros(len(grid))
	prob_survive = scipy.zeros(len(grid))
	params_for_opt = []
	prev_opt_s = 0
	# M = starting cash reserve
	# d = dividend (i.e. consumption)
	# s = savings = M-d
	env_interp = LinInterp(grid, map(lambda s: expected_next_v(w, s, z_rv), list(grid)))
	envprime_interp = LinInterp(grid, map(lambda s: expected_next_v(wPrime, s, z_rv), list(grid)))

	i = 0
	for M in grid:		
		# no need to put rho_t in here, since it will be part of the mean.  if we were to put it in, would need to exclude the zeroes from next_v.		
		#exp_util = lambda s: U(M-s) + beta  * mean(next_v(w,s))
		#exp_util = lambda s: U(M-s) + beta  * expected_next_v(w, s, z_rv);		# try replacing mean() with something that uses pdf of rv
		exp_util = lambda s: U(M-s) + beta  * env_interp(s);		# try replacing mean() with something that uses pdf of rv
		
		# assuming optimal s is monotonic in M, remember the previous optimal s -> this round's minimum
		# it changes the solution for the linear utility gambler's ruin.  Don't use until I can figure out why...
		# Why is opt_c so jagged?  Let's save (M,s,w) for levels of opt_c.
		argmax = maximizer(exp_util, 0, M)
		prev_opt_s = argmax
		params_for_opt.append((argmax, M, copy.copy(w)))
		
		vals[i] = exp_util(argmax)
		vprime[i] = expUtilPrime(w, wPrime, M, argmax, env_interp, envprime_interp)
		opt_controls[i] = argmax
		prob_survive[i] = 1 - z_rv.cdf(-argmax)
		
		i += 1
	addToIter('v', Interp(grid, vals))
	addToIter('vprime', LinInterp(grid, vprime))
	addToIter('opt_s', LinInterp(grid, opt_controls))
	addToIter('prob', LinInterp(grid, prob_survive))
	addToIter('params', params_for_opt)
	
	# marginal utility of consuming vs. saving
	mu1 = lambda d: Uprime(d)
	#mu2 = lambda s: beta * (envprime_interp(s) + z_rv.pdf(-s)*w(0))
	addToIter('mu1', LinInterp(grid, mu1(grid)))
	#addToIter('mu2', LinInterp(grid, mu2(grid)))
	
	result = (Interp(grid, vals), LinInterp(grid, vprime), LinInterp(grid, opt_controls), LinInterp(grid, prob_survive), params_for_opt)
	return result

bellman = bellman1	
	
def viter_until(v, vPrime, criterion=0.001, n=None):
	cont = True	
	currentV = v
	currentVPrime = vPrime
	iter = 0
	opt_ks = None
	t1 = time.time()
	while (cont == True):
		nextIter()
		(newV, newVPrime, opt_ks, prob_survive, arg1) = bellman(currentV, currentVPrime)
		diff = scipy.amax(abs(((newV.Y - currentV.Y) / currentV.Y)))
		print("iteration %d diff %f" % (iter, diff))
		if (diff < criterion or (n != None and iter >= n)):
			cont = False
		currentV = newV
		currentVPrime = newVPrime
		iter += 1		
	t2 = time.time()
	print("total time: %f sec, %f per iteration" % (t2-t1, (t2-t1)/iter))
	return (currentV, currentVPrime, opt_ks, prob_survive)

# functions for policy iteration

def T(sigma, w):
	"Implements the operator L T_sigma."
	vals = []
	env_interp = LinInterp(grid, map(lambda s: expected_next_v(w, s, z_rv), list(grid)))
	for M in grid:
		#Tw_M = U(M - sigma(M)) + beta * mean(w(f(sigma(y), W)))
   		Tw_M = U(M - sigma(M)) + beta * env_interp(sigma(M))		
		vals.append(Tw_M)
	return LinInterp(grid, vals)

def get_greedy(w):
	"Computes a w-greedy policy."
	vals = []
	env_interp = LinInterp(grid, map(lambda s: expected_next_v(w, s, z_rv), list(grid)))
	for M in grid:
		exp_util = lambda s: U(M-s) + beta * env_interp(s)
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
			
	
# initial v: use utility fn
v0 = Interp(grid, U(grid))
v0prime = LinInterp(grid, Uprime(grid))
# initial sigma: save half
sigma0 = LinInterp(grid, grid/2)

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
	addToIter('vprime', v0prime)
	addToIter('opt_s', Interp(grid, [0] * len(grid)))
	addToIter('prob', Interp(grid, [0] * len(grid)))
	
# for policy iteration
def resetIters2():
	global g_iterList
	g_iterList = [{}]
	addToIter('v', v0)
	addToIter('vprime', v0prime)
	addToIter('sigma', sigma0)
	addToIter('prob', Interp(grid, [0] * len(grid)))
	
# v, vprime, opt_c, rho
keys = ['v', 'vprime', 'opt_s', 'prob']
titles = {'v':'v', 'vprime':'vprime', 'opt_s':'optimal savings', 'prob':'probability of survival'}
def plotIters(key, nlist):
	global g_iterList
	
	figure()
	arglist = []
	for n in nlist:
		arglist.append(grid)
		arglist.append(g_iterList[n][key](grid))
	plot(*arglist)
	title(titles[key])
		
def iterate1(n=None):		
	resetIters1()	
	(v1, v1prime, k1, rho1) = viter_until(v0, v0prime, 0.001, n)

def iterate2(n=None):		
	resetIters2()	
	(v1, v1prime, k1, rho1) = piter_until(sigma0, 0.001, n)
	
def plotLast():
	global g_iterList
	
	close('all')
	n = len(g_iterList)
	for key in keys:
		plotIters(key, [n-1])
		
# plot iterations
def plotRange(nlist):
	close('all')
	
	for key in keys:
		plotIters(key, nlist)

# return 1d spline interpolation of a function		
def splineInterp(w):
	return scipy.interpolate.UnivariateSpline(w.X, w.Y, s=0)
	
# plot MU of consumption vs. MU of saving	
def plotMU(nlist):
	global g_iterList
	
	figure()
	arglist = []
	for n in nlist:
		mu1 = g_iterList[n]['mu1']
		mu2 = g_iterList[n]['mu2']
		arglist.append(grid)
		arglist.append(mu1(grid))
		arglist.append(grid)
		arglist.append(mu2(grid[-1] - grid))
		plot(*arglist)
		

