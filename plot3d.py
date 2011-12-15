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
  


import scipy, itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
		
# scatter plot a function on a 2D grid.
# required args:
#   grid1, grid2, fFn
# optional args:
#   xlabel, ylabel, zlabel
#   aroundPoint, aroundN: if given, plot +-N points around aroundPoint
#   colorFn: a function that specifies the color of a point. takes 1 arg, a tuple (gridVal1, gridVal2).  if None, no color
#   filterFn: a function that returns True or False. takes 3 args: (x1, x2, fFn(x1,x2)). if False is returned, don't draw the point. 
#     if None, no effect
#   drawEntireRegion: if True, show the entire region defined by grids, no zooming/cropping
#   useMayavi: if True, use mayavi instead of matplotlib
def plotSurface(grid1_orig, grid2_orig, fFn, xlabel="grid 1", ylabel="grid 2", zlabel="f", colorFn=None, filterFn=None, drawEntireRegion=False, useMayavi=False, aroundPoint=None, aroundN=4):	
	if (aroundPoint != None):
		grid1 = subgridAround(grid1_orig, aroundPoint[0])
		grid2 = subgridAround(grid2_orig, aroundPoint[1])
	else:
		(grid1, grid2) = (grid1_orig, grid2_orig)
	(x1list, x2list) = zip(*itertools.product(grid1, grid2))
	fListOrig = map(fFn, zip(x1list, x2list))	

	# if filterFn is given, only plot points for which filterFn returns True
	if (filterFn != None):
		pointList = filter(lambda x: filterFn(x[0:2]), zip(x1list, x2list, fListOrig))
		(x1list, x2list, fList) = zip(*pointList)
	else:
		fList = fListOrig
		
	# if colorFn is given, calc color of each point
	colorList = None
	if (colorFn != None):
		colorList = map(colorFn, zip(x1list, x2list))
	# if useMayavi is given, plot with Mayavi
	if (useMayavi):
		mylab.figure(bgcolor=(0, 0, 0), size=(400, 400))
		mayaviColorList = None
		if (colorList != None):			
			cconv = MyColorToMayaviColor()
			# map my color to mayavi's colormap
			mayaviColorList = [cconv.mayaviColor(tuple(c)) for c in colorList]
		ax = mylab.points3d(x1list, x2list, fList, s, mode='point')
	else:
		fig = plt.figure()
		ax = Axes3D(fig)		
		ax.scatter(x1list, x2list, fList, color=colorList, s=2)
		if (drawEntireRegion):
			ax.set_xlim3d(grid1[0], grid2[-1])
			ax.set_ylim3d(grid1[0], grid2[-1])
		# set the viewpoint
		set3DViewpoint(ax, xlabel, ylabel, zlabel)
	
	# return fArray
	fArray = scipy.array(fListOrig)
	scipy.reshape(fArray, (len(grid1), len(grid2)))
	return (fArray, ax)	

# given grid1 and x, return a subgrid centered around x of length 2n
# assumes regular grid
def subgridAround(grid, x, n):
	dx = grid[1] - grid[0]
	i_middle = int(scipy.floor((x-grid[0])/dx))
	i_left = max(0, i_middle - n)
	i_right = min(len(grid)-1, i_middle + n)
	return grid[i_left : i_right]
	
# set viewpoint for 3d plots
def set3DViewpoint(ax, xlabel, ylabel, zlabel):
	elev = 21
	azim = -170.53125
	ax.view_init(elev, azim)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
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
	