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
  

# Filename: lininterp.py
# Author: John Stachurski
# Date: August 2009
# Corresponds to: Listing 6.4

from scipy import interp, interpolate, linspace, ogrid

class LinInterp:
    "Provides linear interpolation in one dimension."

    def __init__(self, X, Y):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points.
        """
        self.X, self.Y = X, Y

    def __call__(self, z):
		"""Parameters: z is a number, sequence or array.
		This method makes an instance f of LinInterp callable,
		so f(z) returns the interpolation value(s) at z.
		"""
		return interp(z, self.X, self.Y)

		
# scipy.interpolate.interp2d is a class that provides 2d interpolation.
# construct an obj with args (x, y, z).		

# use scipy.interpolate.Rbf for 3D
# construct with (list of coord1) (list of coord2) (list of coord3) (list of f)
# problem: doesn't work well with many points.
class Interp3DRBF:
	def __init__(self, X1, X2, X3, f):
		(self.X1, self.X2, self.X3, self.f) = (X1, X2, X3, f)
		self.rbf = scipy.interpolate.Rbf(X1, X2, X3, f)

	def __call__(self, x):
		return self.rbf(x[0], x[1], x[2])

# trilinear interpolation
# f is a len(X1) x len(X2) x len(X3) array
class InterpTrilinear:
	def __init__(self, X1, X2, X3, f):
		(self.X1, self.X2, self.X3, self.f) = (X1, X2, X3, f)
		
	def __call__(self, x):
		return self.interp3d(self.X1, self.X2, self.X3, self.f, x[0], x[1], x[2])

	# code from http://beowulf.cheme.cmu.edu/jacapo/9-numerics/9.4-curve-fitting/9.4.0-curve-fitting.html
	def interp3d(self, xv,yv,zv,cd,xi,yi,zi):
		'''
		interpolate a cubic 3D grid defined by x,y,z,cd at the point
		(xi,yi,zi)
		'''

		# assume regular grid.
		# returns the index of the _cell_ (between grid points!)
		# return -1 or len(vector)-2 if outside grid
		def get_cell_index(value,vector):
			dx = vector[1] - vector[0]
			if (value < vector[0]):
				return -1
			if (value >= vector[-1]):
				return len(vector)-2
			return int((value - vector[0]) / dx)

		def force_to_grid(x, vector):
			if (x < vector[0]):
				return vector[0]
			if (x > vector[-1]):
				return vector[-1]
			return x
			
		#xv = x[:,0,0]
		#yv = y[0,:,0]
		#zv = z[0,0,:]

		a = force_to_grid(xi, xv)
		b = force_to_grid(yi, yv)
		c = force_to_grid(zi, zv)
		
		i = get_cell_index(a,xv)
		j = get_cell_index(b,yv)
		k = get_cell_index(c,zv)

		#x1 = x[i,j,k]
		#x2 = x[i+1,j,k]
		x1 = xv[i]
		x2 = xv[i+1]
		#y1 = y[i,j,k]
		#y2 = y[i,j+1,k]
		y1 = yv[j]
		y2 = yv[j+1]	
		#z1 = z[i,j,k]
		#z2 = z[i,j,k+1]
		z1 = zv[k]
		z2 = zv[k+1]


		u1 = cd[i, j, k]
		u2 = cd[i+1, j, k]
		u3 = cd[i, j+1, k]
		u4 = cd[i+1, j+1, k]
		u5 = cd[i, j, k+1]
		u6 = cd[i+1, j, k+1]
		u7 = cd[i, j+1, k+1]
		u8 = cd[i+1, j+1, k+1]

		w1 = u2 + (u2-u1)/(x2-x1)*(a-x2)
		w2 = u4 + (u4-u3)/(x2-x1)*(a-x2)
		w3 = w2 + (w2-w1)/(y2-y1)*(b-y2)
		w4 = u5 + (u6-u5)/(x2-x1)*(a-x1)
		w5 = u7 + (u8-u7)/(x2-x1)*(a-x1)
		w6 = w4 + (w5-w4)/(y2-y1)*(b-y1)
		w7 = w3 + (w6-w3)/(z2-z1)*(c-z1)
		u = w7

		return u		

# test trilinear interpolation
def f3(x, y, z):
	return x+y+z

def test_3(a, b, c):
	n = 10
	grid1 = linspace(0, 1, n)
	grid2 = linspace(0, 2, n)
	grid3 = linspace(0, 3, n)
	x1,x2,x3 = ogrid[0:n, 0:n, 0:n]
	# use an array as index, only works if elts are all integers
	f = f3(grid1[x1], grid2[x2], grid3[x3])
	# x1 is a 1-axis vector, grid1[x1] picks out the corresponding elts of grid1, then inside f3, they are added
	# broadcasting stretches the arrays
	interpObj = InterpTrilinear(grid1, grid2, grid3, f)
	z = interpObj([a, b, c])
	print(z)

import _ponzi2_fns as p2
def test2_3(a, b, c):
	n = 10
	grid1 = linspace(0, 1, n)
	grid2 = linspace(0, 2, n)
	grid3 = linspace(0, 3, n)
	x1,x2,x3 = ogrid[0:n, 0:n, 0:n]
	# use an array as index, only works if elts are all integers
	f = f3(grid1[x1], grid2[x2], grid3[x3])
	# x1 is a 1-axis vector, grid1[x1] picks out the corresponding elts of grid1, then inside f3, they are added
	# broadcasting stretches the arrays	
	z = p2.test3(grid1, grid2, grid3, f, a, b, c)
	print(z)
	
# for splines, use scipy.interpolate.UnivariateSpline








