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
  


# classes for linear interpolation
# requires a regular grid

import scipy, types
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyublas, _myfuncs
import itertools
import time

class LinInterp1D_2:
	def __init__(self, grid1, fArray):
		assert(len(grid1) == fArray.shape[0])
		(self.m_grid1, self.m_FArray) = (grid1, fArray)
	def __call__(self, z):
		# this is a hack, should find a better way.
		arg = z
		if (not isinstance(z, types.FloatType)):
		  arg = z[0]
		return _myfuncs.interp1d(self.m_grid1, self.m_FArray, arg)

class LinInterp1D(_myfuncs.Interp1D):
	def __init__(self, grid1, fArray):
		super(LinInterp1D,self).__init__(grid1, fArray)
		(self.m_grid1, self.m_FArray) = (grid1, fArray)
		
class LinInterp2D_grid:
	def __init__(self, grid1, grid2, fArray):
		assert(len(grid1) == fArray.shape[0] and len(grid2) == fArray.shape[1])
		(self.m_grid1, self.m_grid2, self.m_FArray) = (grid1, grid2, fArray)
#	def __call__(self, x1, x2):
#		return _myfuncs.interp2d(self.m_grid1, self.m_grid2, self.m_FArray, x1, x2)
	def __call__(self, x):
		return _myfuncs.interp2d(self.m_grid1, self.m_grid2, self.m_FArray, x[0], x[1])

class LinInterp2D_obj(_myfuncs.Interp2D):
	def __init__(self, grid1, grid2, fArray):
		super(LinInterp2D,self).__init__(grid1, grid2, fArray)
		(self.m_grid1, self.m_grid2, self.m_FArray) = (grid1, grid2, fArray)
LinInterp2D = LinInterp2D_obj
		
class LinInterp3D:
	def __init__(self, grid1, grid2, grid3, fArray):
		assert(len(grid1) == fArray.shape[0] and len(grid2) == fArray.shape[1] and len(grid3) == fArray.shape[2])
		(self.m_grid1, self.m_grid2, self.m_grid3, self.m_FArray) = (grid1, grid2, grid3, fArray)
	def __call__(self, x1, x2, x3):
		return _myfuncs.interp3d(self.m_grid1, self.m_grid2, self.m_grid3, self.m_FArray, x1, x2, x3)
	def __call__(self, x):
		return _myfuncs.interp3d(self.m_grid1, self.m_grid2, self.m_grid3, self.m_FArray, x[0], x[1], x[2])
	
# automatically figure out the right object based on the number of dimensions.  only works for 1 to 3	
def GetLinterpFnObj(stateGridList, fArray):
	assert(len(stateGridList) == fArray.ndim);			# check the shapes match
	for i in range(len(stateGridList)):
		assert(len(stateGridList[i]) == fArray.shape[i])
	if (fArray.ndim == 1):
		return LinInterp1D(stateGridList[0], fArray)
	elif (fArray.ndim == 2):
		return LinInterp2D(stateGridList[0], stateGridList[1], fArray)
	elif (fArray.ndim == 3):
		return LinInterp3D(stateGridList[0], stateGridList[1], stateGridList[2], fArray)		
	else:
		assert(false)
		
def test_1D():
	innerGrid = scipy.linspace(-5, 5, 20)
	fArray = scipy.sin(innerGrid)
	outerGrid = scipy.linspace(-10, 10, 66)
	interp1 = LinInterp1D_2(innerGrid, fArray)
	interp2 = LinInterp1D(innerGrid, fArray)
	
	fig = plt.figure()
	plt.plot(outerGrid, map(interp1, outerGrid))
	fig = plt.figure()	
	plt.plot(outerGrid, map(interp2, outerGrid))
	fig = plt.figure()
	plt.plot(outerGrid, interp2(outerGrid))
	
def test_2D():
	def f(x):
		return scipy.sin(x[0] + 2*x[1])
	coarseGrid1 = scipy.linspace(-5, 5, 80)
	coarseGrid2 = scipy.linspace(-5, 5, 100)
	fineGrid1 = scipy.linspace(-6, 6, 200)
	fineGrid2 = scipy.linspace(-6, 6, 220)	
	[c_x1list, c_x2list] = zip(*itertools.product(coarseGrid1, coarseGrid2))
	[f_x1list, f_x2list] = zip(*itertools.product(fineGrid1, fineGrid2))
	fList = map(f, zip(c_x1list, c_x2list))
	fArray = scipy.array(fList).reshape((len(coarseGrid1), len(coarseGrid2)))			
	# original
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(c_x1list, c_x2list, fList)
	# interp1
	fig = plt.figure()
	ax = Axes3D(fig)
	t1 = time.time()
	interp1 = LinInterp2D_grid(coarseGrid1, coarseGrid2, fArray)
	t2 = time.time()
	for i in range(100):
		interp_fList1 = map(interp1, zip(f_x1list, f_x2list))
	t3 = time.time()
	print("interp grid: setup %f sec, interp %f sec" % (t2-t1, t3-t2))
	ax.scatter(f_x1list, f_x2list, interp_fList1)
	# interp2
	fig = plt.figure()
	ax = Axes3D(fig)
	t1 = time.time()
	interp2 = LinInterp2D_obj(coarseGrid1, coarseGrid2, fArray)
	t2 = time.time()
	for i in range(100):
		interp_fList2 = map(interp2, zip(f_x1list, f_x2list))
	t3 = time.time()
	print("interp obj: setup %f sec, interp %f sec" % (t2-t1, t3-t2))
	ax.scatter(f_x1list, f_x2list, interp_fList2)
	
	diff = scipy.array(interp_fList1) - scipy.array(interp_fList2)
	return scipy.sum(diff*diff)

class LinInterp1D_irreg:
	def __init__(self, grid, fArray):
		(self.grid, self.fArray) = grid, fArray
		self.slopes = (fArray[1:] - fArray[:-1]) / (grid[1:] - grid[:-1])
		
	def interp(self, xi):
		if (xi < self.grid[0]):
			return self.fArray[0]
		if (xi > self.grid[-1]):
			return self.fArray[-1]
		cell = binarySearchForCell(self.grid, xi)
		result = self.fArray[cell] + (xi - self.grid[cell]) * self.slopes[cell]
		return result
		
def binarySearchForCell(grid, x):
	assert(len(grid) > 1)
	left = 0
	right = len(grid)-1
	while (left+1 < right):
		mid = int( (left+right)/2 )
		if (x < grid[mid]):
			right = mid
		else:
			left = mid
	return left

# given an n-dim array f on grids, return an interpolated array on grids of a different size.
# f must have same dimensions as grids.
def interpolateArray(gridList1, gridList2, f):
	assert(len(gridList1) == len(gridList2))
	assert([len(g) for g in gridList1] == list(f.shape))
	interpObj = GetLinterpFnObj(gridList1, f)
	# iterate through each grid point, cycle through grid 0 on the innermost loop
	
	#interpList = [interpObj(x) for x in itertools.product(*list(reversed(gridList2)))]	
	interpList = []
	for x in itertools.product(*gridList2):
		x2 = list(x)
		z = interpObj(x2)
		interpList.append(z)
		#print((x, z))
	result = scipy.array(interpList).reshape(tuple( [len(g) for g in gridList2] ))
	return result

# return an array of fn applied to each grid point in gridList. last elt of gridList will be the innermost loop
def applyGrid(gridList, fn):
	z_list = list(itertools.product(*gridList))
	x_list = zip(*z_list)
	f = scipy.array([fn(z) for z in z_list])
	f = f.reshape(tuple( [len(g) for g in gridList] ))
	
	return (x_list, f)
	
def test_interpolateArray():
	grid_x = scipy.linspace(1, 5, 20)
	grid_y = scipy.linspace(-1, 1, 10)
	def fn(x):
		return scipy.sin(x[0] + x[1])
	((xlist, ylist), f) = applyGrid([grid_x, grid_y], fn)
	fig = plt.figure()
	ax = Axes3D(fig)	
	ax.scatter(xlist, ylist, f.ravel())
	
	grid2_x = scipy.linspace(1, 5, 40)
	grid2_y = scipy.linspace(-1, 1, 20)
	f2 = interpolateArray([grid_x, grid_y], [grid2_x, grid2_y], f)
	xy_list = itertools.product(grid2_x, grid2_y)
	(xlist2, ylist2) = zip(*xy_list)	
	fig = plt.figure()	
	ax = Axes3D(fig)
	ax.scatter(xlist2, ylist2, f2.ravel())
	
	grid3_x = grid_x
	grid3_y = grid_y
	f3 = interpolateArray([grid2_x, grid2_y], [grid3_x, grid3_y], f2)
	xy_list = itertools.product(grid3_x, grid3_y)
	(xlist3, ylist3) = zip(*xy_list)	
	fig = plt.figure()	
	ax = Axes3D(fig)	
	ax.scatter(xlist3, ylist3, f3.ravel())	
	
	
	
	
	
	
	