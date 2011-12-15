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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
#from pylab import *
from IPython.Debugger import Tracer; debug_here = Tracer()
import copy, time
import cPickle as pickle
import gzip
import os
import networkx as nx

# my modules. 
import lininterp2 as linterp
import pyublas, _debugMsg, _maximizer as mx, bellman
import _ponziProblem as p_fns
import ponziGlobals as g
import my_bfs
import plot3d

reload(my_bfs)
reload(g)

# we only implement getControlGridList in python, the rest are in C++
class PonziParams(p_fns.PonziParams):
	def __init__(self):
		super(PonziParams,self).__init__()
	def getControlGridList(self, stateVarList):
		M = stateVarList[0]
		grid_d = linspace(0, M, g.grid_d_size)
		return [grid_d, g.grid_r]

# value iteration		
def iterate1(n=None, **kwargs):
	localvars = {}	
	def postIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter
		bellman.appendIter()
		bellman.setInLastIter('v', newVArray)
		bellman.setInLastIter('opt_d', optControls[0])
		bellman.setInLastIter('opt_r', optControls[1])

	v0_array = scipy.zeros((len(g.grid_M), len(g.grid_D)));			# initial guess for value function
	bellman.resetIters()
	bellman.appendIter()
	bellman.setInLastIter('v', v0_array)
	bellman.setInLastIter('opt_d', None)
	bellman.setInLastIter('opt_r', None)
	
	params = PonziParams()	
	time1 = time.time()
	result = bellman.grid_valueIteration([g.grid_M, g.grid_D], v0_array, params, postIterCallbackFn=postIterCallbackFn)
	(nIter, currentVArray, newVArray, optControls) = result
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))

# policy iteration
def iterate2(n=None, **kwargs):	
	localvars = {}		
	def postIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter
		bellman.appendIter()
		bellman.setInLastIter('v', newVArray)
		bellman.setInLastIter('opt_d', currentPolicyArrayList[0])
		bellman.setInLastIter('opt_r', currentPolicyArrayList[1])

	v0_array = scipy.zeros((len(g.grid_M), len(g.grid_D)));			# initial guess for value function
	d0_array = scipy.zeros((len(g.grid_M), len(g.grid_D)));			# initial guess for policy function for d
	r0_array = scipy.ones((len(g.grid_M), len(g.grid_D)));			# initial guess for policy function for r
	
	bellman.resetIters()
	bellman.appendIter()
	bellman.setInLastIter('v', v0_array)
	bellman.setInLastIter('opt_d', None)
	bellman.setInLastIter('opt_r', None)
	
	params = PonziParams()	
	time1 = time.time()
	result = bellman.grid_policyIteration([g.grid_M, g.grid_D], [d0_array, r0_array], v0_array, params, postIterCallbackFn=postIterCallbackFn, parallel=True)
	(nIter, currentVArray, currentPolicyArrayList, greedyPolicyList) = result	
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f" % (time2-time1, (time2-time1)/nIters))		
	
# save/load run
def saveRun(filename):
	params = g.getGlobalParams()
	itersObj = bellman.getItersObj()
	output = gzip.open(filename, 'wb')
	pickle.dump((params, itersObj), output)
	output.close()
def loadRun(filename):	
	pk_file = gzip.open(filename, 'rb')
	(params, itersObj) = pickle.load(pk_file)
	pk_file.close()
	g.setGlobalParams(True, **params)
	bellman.setItersObj(itersObj)		
		
def iterColorFn(n):
	classifyStateFn = lambda M,D: classifyStateUsingIter(M,D,n+1)
	colorFn = lambda x: stateToRGBA(classifyStateFn(x[0], x[1]))			
	return colorFn
	
def plotV(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	fnObj = linterp.GetLinterpFnObj(g.stateGridList, bellman.getIter(n)['v'])
	return plot3d.plotSurface(g.grid_M, g.grid_D, fnObj, xlabel="M", ylabel="D", zlabel="V", colorFn=iterColorFn(n), **kwargs)
def plotOptD(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	fnObj = linterp.GetLinterpFnObj(g.stateGridList, bellman.getIter(n)['opt_d'])
	return plot3d.plotSurface(g.grid_M, g.grid_D, fnObj, xlabel="M", ylabel="D", zlabel="opt d", colorFn=iterColorFn(n), **kwargs)
def plotK(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	opt_d = bellman.getIter(n)['opt_d']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]
	return plotSurface(k_fn, 'k = d + D - M', aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotF(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	opt_d = bellman.getIter(n)['opt_d']
	opt_r = bellman.getIter(n)['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p_fns.testf(k_fn(x), opt_r(x))
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
	opt_d = bellman.getIter(n)['opt_d']
	opt_r = bellman.getIter(n)['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p_fns.testf(k_fn(x), opt_r(x))
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
	fnObj = linterp.GetLinterpFnObj(g.stateGridList, bellman.getIter(n)['opt_r'])
	return plot3d.plotSurface(g.grid_M, g.grid_D, fnObj, xlabel="M", ylabel="D", zlabel="opt r", colorFn=iterColorFn(n), filterFn=filterFn, drawEntireRegion=True, **kwargs)		

# calculate next period's M, D contingent on zLow, zHigh
#def getNextMD(M, D, optdFn, optrFn, zState):
def getNextMD(M, D, n, zState):
	optdFn = bellman.getIter(n)['opt_d']
	optrFn = bellman.getIter(n)['opt_r']
	# k = d + D - M
	k_fn = lambda x: optdFn(x) + x[1] - x[0]	
	f_fn = lambda x: p_fns.testf(k_fn(x), optrFn(x))
	inflow = lambda x: f_fn(x) * g.z_space[zState]
	M_fn = lambda x: (inflow(x) - k_fn(x))/g.z_space[zState]
	D_fn = lambda x: optrFn(x) * f_fn(x)
	return (M_fn([M,D]), D_fn([M,D]))

# plot next value of M if zLow occurs	
def plotNextM(zState, n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	M_fn = lambda x: getNextMD(x[0], x[1], n, zState)[0]
	return plotSurface(M_fn, 'next M, state=z[%d]' % zState, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextMlow(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextM(0, n, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextMhigh(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextM(1, n, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
	
def plotNextD(state, n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	D_fn = lambda x: getNextMD(x[0], x[1], n, zState)[1]
	return plotSurface(D_fn, 'next D, state=z[%d]' % zState, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextDlow(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextD(0, n, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
def plotNextDhigh(n, aroundPoint=None, aroundN=None, colorStates=False, **kwargs):
	return plotNextD(1, n, aroundPoint, aroundN, colorStates=colorStates, colorFn=iterColorFn(n), **kwargs)
	
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
	f_fn = lambda x: p_fns.testf(x[0], x[1])
	grid_k = linspace(0, 2, 50)
	grid_r = linspace(1, 3, 50)
	(mesh_k, mesh_r) = meshgrid(grid_k, grid_r)
	meshlist_k = mesh_k.ravel()
	meshlist_r = mesh_r.ravel()
	
	ax.scatter(meshlist_k, meshlist_r, array(map(f_fn, zip(meshlist_k, meshlist_r))))	
	ax.set_xlabel('k')
	ax.set_ylabel('r')
	ax.set_zlabel('f')

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
	opt_d = g.getFnObj(bellman.getIter(n)['opt_d'])
	opt_r = g.getFnObj(bellman.getIter(n)['opt_r'])
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p_fns.f(k_fn(x), opt_r(x))
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
	opt_d = bellman.getIter(n)['opt_d']
	opt_r = bellman.getIter(n)['opt_r']
	# k = d + D - M
	k_fn = lambda x: opt_d(x) + x[1] - x[0]	
	f_fn = lambda x: p_fns.testf(k_fn(x), opt_r(x))
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
	opt_d = bellman.getIter(n)['opt_d']
	opt_r = bellman.getIter(n)['opt_r']

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

bellman.resetIters()


