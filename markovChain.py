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
  


import networkx as nx
import itertools
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.artist as artist

# create a graph representing a Markov chain, given state variable grids and policy functions that take each point in state space
#   to another point in state space, or an absorbing coffin state.			
# - coffinState is the value of the absorbing state (e.g. bankruptcy)
# - isAbsorbedFn takes 2 args: tuple of indices and a shock, returns True if the next state is the coffin state
# - nextIListFn takes a tuple of indices and returns the next state value's tuple of indices
def createGraphFromPolicyFn(gridList, coffinState, shockList, isAbsorbedFn, nextIListFn):
	rangeList = [range(len(grid)) for grid in gridList]
	G = nx.DiGraph(gridList=gridList, coffinState=coffinState, shockList=shockList)
	# absorbing coffin state (e.g. bankruptcy)
	G.add_node(coffinState)
	G.add_edge(coffinState, coffinState, shock=None, controls=None)
	for iList in itertools.product(*rangeList):
		G.add_node(iList)
		for shock in shockList:
			if (isAbsorbedFn(iList, shock)):
				G.add_edge(iList, coffinState, shock=shock)
			else:
				(nextIList, controls) = nextIListFn(iList, shock)
				G.add_edge(iList, nextIList, shock=shock, controls=controls)
				#print(iList, nextIList, shock)
	return G

# assume a regular grid
def getNearestGridPoint1D(grid, x):	
	dx = grid[1] - grid[0]
	if (x <= grid[0]):
		return 0
	if (x >= grid[-1]):
		return len(grid)-1
	i = (x - grid[0]) / dx
	i1 = int(scipy.floor(i))
	i2 = int(scipy.ceil(i))
	if (x - grid[i1] < grid[i2] - x):
		return i1
	else:
		return i2
		
# return index of nearest grid point
def getNearestGridPoint(gridList, x):
	assert(len(x) == len(gridList))
	i_list = [getNearestGridPoint1D(gridList[i], x[i]) for i in range(len(gridList))]
	return i_list
	
# return a pair of functions that can be passed into createGraph	
def policyFn_to_transitionFns(gridList, nextStateVarFn, absorbedTestFn):	
	def isAbsorbedFn(iList, shock):
		x = [gridList[j][iList[j]] for j in range(len(gridList))]
		(next_x, controls) = nextStateVarFn(x, shock)
		if (absorbedTestFn(next_x)):
			return True
		return False
	def nextIListFn(iList, shock):
		x = [gridList[j][iList[j]] for j in range(len(gridList))]
		(next_x, controls) = nextStateVarFn(x, shock)
		next_iList = getNearestGridPoint(gridList, next_x)
		return (tuple(next_iList), controls)
	return (isAbsorbedFn, nextIListFn)
	
# plot a 2d graph of nodes that satisfy some predicate
# predicateFn(graph, node) returns True or False	
def plotGraphNodes2D(G, predicateFn, title=None, xlabel=None, ylabel=None):	
	fig = plt.figure()
	ax = fig.add_subplot(111, title=title)	
	grid1 = G.graph['gridList'][0]
	grid2 = G.graph['gridList'][1]
	x1_list = []
	x2_list = []
	i_to_point_map = {}
	for (i1, x1) in enumerate(grid1):
		for (i2, x2) in enumerate(grid2):
			if (predicateFn(G, (i1, i2))):
				i_to_point_map[(i1, i2)] = len(x1_list)
				x1_list.append(grid1[i1])
				x2_list.append(grid2[i2])				
	if (len(x1_list) > 0):
		points = plt.scatter(x1_list, x2_list, s=6, c=['k']*len(x1_list))
	#ax.set_xlim(grid1[0], grid1[-1])
	#ax.set_ylim(grid2[0], grid2[-1])
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
	# set up events for mouseover
	global g_select_point, g_next_point_1, g_next_point_2
	g_select_point = None
	def setColor(point, rgba):
		if (point == G.graph['coffinState']): return
		(i1, i2) = point
		if (not (i1, i2) in i_to_point_map): return
		index = i_to_point_map[(i1, i2)]
		points.get_facecolors()[index,:] = rgba

	def removeHighlight():
		global g_select_point, g_next_point_1, g_next_point_2
		if (g_select_point != None):			
			setColor(g_select_point, scipy.array([0., 0., 0., 1.]));		# black
			setColor(g_next_point_1, scipy.array([0., 0., 0., 1.]));		
			setColor(g_next_point_2, scipy.array([0., 0., 0., 1.]));		
			g_select_point = None
			g_next_point_1 = None
			g_next_point_2 = None
			return True
		return
	def setHighlight(i1, i2
	):
		newPoint = (i1, i2)
		global g_select_point, g_next_point_1, g_next_point_2
		if (newPoint == g_select_point): return False
		removeHighlight()
		# color highlighted point
		successors = G.successors(newPoint)
		assert(len(successors) == 2)
		if (G[newPoint][successors[0]]['shock'] == 0):
			(next_point_1, next_point_2) = (successors[0], successors[1])
		else:
			(next_point_1, next_point_2) = (successors[1], successors[0])
		setColor(newPoint, scipy.array([1., 0., 0., 1.]))
		setColor(next_point_1, scipy.array([0., 1., 0., 1.]))
		setColor(next_point_2, scipy.array([1., 1., 0., 1.]))
		(g_select_point, g_next_point_1, g_next_point_2) = (newPoint, next_point_1, next_point_2)
		return True
		
	def onMouseMotion(event):
		(xdata, ydata) = (event.xdata, event.ydata)
		[i1, i2] = getNearestGridPoint(G.graph['gridList'], [xdata, ydata])
		#(bInItem, dict_itemlist) = points.contains(event)		
		#if (not bInItem):
		#	removeHighlight()
		#	return
		redraw = False
		if (not (i1, i2) in i_to_point_map):
			redraw = redraw or removeHighlight()
		else:			
			redraw = redraw or setHighlight(i1, i2)
		if (redraw): plt.draw()
		
		
	fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)
	
# delete nodes that have indegree 0 until there are none left
# return the remaining set of nodes
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
	
class Simulation:
	def __init__(self, G, initialNode):
		self.G = G
		self.shockList = G.graph['shockList']
		self.shockSet = set(self.shockList)
		self.initialNode = initialNode
		self.currentNode = initialNode
		self.currentControls = None
		# map (node, shock) to edge
		self.nodeShock_to_nextNode = {}
		for (u, v, edata) in G.edges(data=True):
			shock = edata['shock']
			controls = edata['controls']
			self.nodeShock_to_nextNode[(u, shock)] = (v, controls)
			
	def applyShock(self, shock):		
		(nextNode, controls) = self.nodeShock_to_nextNode[(self.currentNode, shock)]
		self.currentNode = nextNode
		self.currentControls = controls 
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		