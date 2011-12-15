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
from IPython.Debugger import Tracer; debug_here = Tracer()
import pyublas, _myfuncs
import lininterp2 as linterp

# calculate the expected value of f(z) via Monte Carlo, where f(z) is an arbitrary function that takes a value which sill be randomized
# and zRV is a scipy.stats random variable
# memoize RVs
g_montecarloDraws = {}
def calculateEV_montecarlo(fFn, zRV, nDraws=10000):
	global g_montecarloDraws	
	if (not (zRV, nDraws) in g_montecarloDraws):
		g_montecarloDraws[(zRV, nDraws)] = scipy.sort(zRV.rvs(size=nDraws))
	draws = g_montecarloDraws[(zRV, nDraws)]		
	vals = map(fFn, draws)
	EV = scipy.mean(vals)
	return EV

# uses 1D interpolation code
def calculateEV_montecarlo2(grid, fArray, zRV, nDraws=10000):
	global g_montecarloDraws	
	if (not (zRV, nDraws) in g_montecarloDraws):
		g_montecarloDraws[(zRV, nDraws)] = scipy.sort(zRV.rvs(size=nDraws))
	draws = g_montecarloDraws[(zRV, nDraws)]		
	fn = linterp.LinInterp1D(grid, fArray)
	EV = fn.applySorted(draws) / nDraws
	return EV

def test_montecarlo():
	stdnorm = scipy.stats.norm()
	x = scipy.linspace(-5, 5, 100)
	fn = linterp.LinInterp1D(x, x)
	EV1 = calculateEV_montecarlo(fn, stdnorm, nDraws=100000)
	EV2 = calculateEV_montecarlo2(x, x, stdnorm, nDraws=100000)
	print(EV1)
	print(EV2)
	
# use built-in integration
def calculateEV_integrate(fFn, zRV, a=-scipy.integrate.Inf, b=scipy.integrate.Inf):
	def fn(x):
		return fFn(x) * zRV.pdf(x)
	EV = scipy.integrate.quad(fn, a, b)
	return EV[0]

# f is defined on a grid.  assume that f takes on a constant value outside the grid, equal to leftK and rightK respectively.
# the distribution of z can be displaced by zOffset
def calculateEV_grid(gridArray, fFn, zRV, zOffset, leftK, rightK, fArray=None):	
	# use fArray if provided
	if (fArray == None):
		fArray = fFn(gridArray)
	assert(len(gridArray) == len(fArray))
	# below is the integral to the left of the grid
	below = leftK * zRV.cdf(gridArray[0] - zOffset)
	# above is the integral to the right of the grid
	above = rightK * (1.0 - zRV.cdf(gridArray[-1] - zOffset))
	# between is the integral on the grid.  evaluate f*pdf on the grid points and integrate
	betweenArray = fArray * zRV.pdf(gridArray - zOffset)
	assert(len(betweenArray) == len(fArray))
	between = scipy.integrate.trapz(betweenArray, gridArray)
	return below + above + between

def calculateEV_grid2(fGrid, fVals, pdfGrid, pdfVals, inverseFn):
	inverseList = map(inverseFn, pdfGrid)
	fFn = linterp.LinInterp1D(fGrid, fVals)
	fList = map(fFn, inverseList)
	xList = [f*p for (f,p) in zip(fList, pdfVals)]
	EV = scipy.integrate.trapz(xList, pdfGrid)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(pdfGrid, xList)
	plt.title("f*p")
	
	return EV

# partial expectation of a lognormal with (normal) mean, var params
# on an interval [k, infty)
def lognormal_PartialExp(k, mean, sd):
	return scipy.exp(mean + 0.5*sd*sd) * scipy.stats.norm.cdf( (mean + (sd*sd) - scipy.log(k)) / sd )

# this matches with lognormal_PartialExp. OK
def partialExp1(k, mean, sd):
	rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
	return calculateEV_integrate(lambda x: x, rv, a=k)
	
# instead of a y=x line, take the expectation of Ax+B
def lognormal_PartialExp_Affine(k, A, B, mean, sd):
	return A*lognormal_PartialExp(k, mean, sd) + B*(1.0 - scipy.stats.lognorm.cdf(k, sd, scale=scipy.exp(mean)))	

# matches. OK
def partialExp2(k, A, B, mean, sd):
	rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
	def fn(x):
		return B + A*x
	return calculateEV_integrate(fn, rv, a=k)
	
# calculate expectation of a linear fn using lognormal RV
def EV_interval(x0, x1, y0, y1, mean, sd):
	slope = (y1-y0)/(x1-x0)
	y_intercept = y0 - slope*x0
	return lognormal_PartialExp_Affine(x0, slope, y_intercept, mean, sd) - lognormal_PartialExp_Affine(x1, slope, y_intercept, mean, sd)

# OK
def EV_interval2(x0, x1, y0, y1, mean, sd):
	rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
	def fn(x):
		slope = (y1-y0)/(x1-x0)
		return y0 + slope*(x-x0)
	return calculateEV_integrate(fn, rv, a=x0, b=x1)
	
# inverseFn maps fGrid-space to rv-space
def calculateEV_lognorm(fGrid, fVals, inverseFn, mean, sd):
	grid2 = map(inverseFn, fGrid)
	rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
	below = fVals[0] * rv.cdf(grid2[0])
	above = fVals[-1] * (1.0 - rv.cdf(grid2[-1]))
	between = 0.0
	for i in range(len(grid2)-1):
		between += EV_interval(grid2[i], grid2[i+1], fVals[i], fVals[i+1], mean, sd)
	return below + above + between
	
def test_calculateEV_norm(mean=0.0, moment=2):
	# calculate moments of a normal distribution
	def fFn(x):
		return abs(scipy.real(scipy.power(x, moment)))
	gridArray = scipy.linspace(-10, 10, 500)
	fArray = fFn(gridArray)
	sdRange = scipy.linspace(0.1, 2.0, 100)
	
	t1 = time.time()
	result1 = [calculateEV_montecarlo(fFn, scipy.stats.norm(loc=mean, scale=sd)) for sd in sdRange]
	t2 = time.time()
	result2 = [calculateEV_integrate(fFn, scipy.stats.norm(loc=mean, scale=sd)) for sd in sdRange]
	t3 = time.time()
	result3 = [calculateEV_grid(gridArray, fFn, scipy.stats.norm(loc=mean, scale=sd), zOffset=0.0, leftK=fArray[0], rightK=fArray[-1]) for sd in sdRange]
	t4 = time.time()
	result4 = [_myfuncs.mycalcEV_grid(gridArray, fArray, _myfuncs.NormalCDFObj(mean, sd), 
				_myfuncs.NormalPDFObj(mean, sd), 0.0, fArray[0], fArray[-1]) for sd in sdRange]
	t5 = time.time()

	print("montecarlo: %f" % (t2-t1))
	print("integrate: %f" % (t3-t2))
	print("grid: %f" % (t4-t3))
	print("c++ grid: %f" % (t5-t4))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result1)
	plt.title("monte carlo")
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result2)	
	plt.title("integrate")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result3)	
	plt.title("grid")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result4)	
	plt.title("c++ grid")

def test_calculateEV_lognorm(mean=0.0, moment=2):
	# calculate moments of a lognormal distribution
	def fFn(x):
		#return abs(scipy.real(scipy.power(x, moment)))
		return scipy.real(-(1.0/x))
	#gridArray = scipy.linspace(0.001, 20, 1000)		
	sdRange = scipy.linspace(0.1, 1.5, 100)
	
	t1 = time.time()
	result1 = []
	for sd in sdRange:
		rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
		EV = calculateEV_montecarlo(fFn, rv)
		result1.append(EV)
	t2 = time.time()
	result2 = [calculateEV_integrate(fFn, scipy.stats.lognorm(sd, scale=scipy.exp(mean))) for sd in sdRange]
	t3 = time.time()
	result3 = []
	for sd in sdRange:
		rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
		x = rv.ppf(scipy.linspace(0, 1, 200))[1:-1]
		y = fFn(x)		
		EV = calculateEV_grid(x, fFn, rv, zOffset=0.0, leftK=y[0], rightK=y[-1], fArray=y)
		result3.append(EV)
	t4 = time.time()
	result4 = []
	for sd in sdRange:
		rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
		x = rv.ppf(scipy.linspace(0, 1, 200))[1:-1]
		y = fFn(x)
		result4.append(_myfuncs.mycalcEV_grid(x, y, _myfuncs.LognormalCDFObj(mean, sd), 
				_myfuncs.LognormalPDFObj(mean, sd), 0.0, y[0], y[-1]))
	t5 = time.time()	
	result5 = []
	x = scipy.linspace(0, 20, 1000)
	y = fFn(x)
	for sd in sdRange:
		rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
		pdf_x = rv.ppf(scipy.linspace(0, 1, 1000))[1:-1]
		pdf_y = rv.pdf(pdf_x)		
		result5.append(_myfuncs.mycalcEV_grid2(x, y, pdf_x, pdf_y))
	t6 = time.time()	
	result6 = []
	x = scipy.linspace(0, 20, 1000)[1:-1]
	y = fFn(x)		
	for sd in sdRange:
		rv = scipy.stats.lognorm(sd, scale=scipy.exp(mean))
		EV = calculateEV_montecarlo2(x, y, rv)
		result6.append(EV)
	t7 = time.time()	
	result7 = []
	x = scipy.linspace(0, 20, 1000)[1:-1]
	y = fFn(x)		
	for sd in sdRange:
		EV = calculateEV_lognorm(x, y, lambda x: x, mean, sd)
		result7.append(EV)
	t8 = time.time()	
	result8 = []
	x = scipy.linspace(0, 20, 1000)[1:-1]
	y = fFn(x)		
	for sd in sdRange:
		EV = _myfuncs.lognormal_EV_lininterp(x, y, mean, sd)
		result8.append(EV)
	t9 = time.time()	
	
	print("montecarlo: %f" % (t2-t1))
	print("integrate: %f" % (t3-t2))
	print("grid with CDF/PDF: %f" % (t4-t3))
	print("c++ grid with CDF/PDF: %f" % (t5-t4))
	print("c++ grid 2: %f" % (t6-t5))	
	print("monte carlo 2: %f" % (t7-t6))
	print("partial exp: %f" % (t8-t7))
	print("c++ partial exp: %f" % (t9-t8))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result1)
	plt.title("monte carlo")
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result2)	
	plt.title("integrate")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result3)	
	plt.title("grid")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result4)	
	plt.title("c++ grid")
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result5)	
	plt.title("c++ grid 2")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result6)	
	plt.title("monte carlo 2")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result7)	
	plt.title("partial exp")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(sdRange, result8)	
	plt.title("c++ partial exp")
	