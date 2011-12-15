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
  


# incremental monte carlo expectations

import scipy

class incrMonteCarlo:
	def __init__(self, grid, fArray, zDrawsSorted):
		assert(len(grid) == len(fArray))
		(self.grid, self.fArray) = (grid, fArray)
		self.zDraws = zDraws		
		self.slopes = scipy.zeros(len(grid) - 1)
		self.dx = grid[1] - grid[0]
		for i in range(len(grid) - 1):
			self.slopes[i] = (fArray[i+1] - fArray[i]) / self.dx
		# set up sums
		self.cellSums = scipy.zeros(len(grid) + 1)
		self.boundaryIndices = [len(zDraws)] * len(grid)
		for (i, x) in enumerate(grid):
			indices = scipy.nonzero(self.zDraws >= x)[0]
			if (len(indices) > 0):
				self.boundaryIndices[i] = indices[0]
		self.cellSums[0] = scipy.sum(self.zDraws[0:self.boundaryIndices[0]])
		for i in range(1, len(self.cellSums)-1):
			self.cellSums[i] = scipy.sum(self.zDraws[self.boundaryIndices[i-1] : self.boundaryIndices[i]])
		self.cellSums[-1] = scipy.sum(self.zDraws[self.boundaryIndices[-1] : ])
		
		diff = scipy.sum(self.zDraws) - scipy.sum(self.cellSums)
		print("diff: %f" % diff)
		for i in range(len(grid)):
			if (self.boundaryIndices[i] < len(self.zDraws)):
				print("grid point %f, boundary %f" % (self.grid[i], self.zDraws[self.boundaryIndices[i]]))
			else:
				print("grid point %f, no draws to right" % self.grid[i])
				
	def interp(self, xi):
		if (xi < self.grid[0]):
			return self.fArray[0]
		if (xi > self.grid[-1]):
			return self.fArray[-1]
		cell = int(scipy.floor( (xi-self.grid[0])/self.dx ))
		if (cell == len(self.grid)-1):
			cell += -1
		result = self.fArray[cell] + (xi - self.grid[cell]) * self.slopes[cell]
		return result

	def getCurrentMC(self):
		sum = 0.0
		# left of grid
		n = self.boundaryIndices[0]
		sum += n * self.fArray[0]
		for i in range(1, len(self.cellSums)-1):
			n = self.boundaryIndices[i] - self.boundaryIndices[i-1]
			cell = i-1
			sum += (n*self.fArray[cell]) + (self.cellSums[i] - n*self.grid[cell])*self.slopes[cell]
		# right of grid
		n = len(self.zDraws) - self.boundaryIndices[-1]
		sum += n * self.fArray[-1]
		return sum
	
	def shiftGrid(self, delta):
		# for now, assume delta is positive.
		newGrid = self.grid + delta
		newCellSums = scipy.array(self.cellSums)
		newBoundaryIndices = list(self.boundaryIndices)
		for i in range(len(newBoundaryIndices)):
			oldIndex = self.boundaryIndices[i]
			if (i>0 and oldIndex < newBoundaryIndices[i-1]):
				oldIndex = newBoundaryIndices[i-1]
			newIndex = findFirstEltGTE(self.zDraws, oldIndex, newGrid[i])
			# move elements from old index up to new index, to previous cell
			moveSum = scipy.sum(self.zDraws[oldIndex:newIndex])
			newCellSums[i] += moveSum
			newCellSums[i+1] -= moveSum
			newBoundaryIndices[i] = newIndex
			# have to change the slopes too...
			
# given an array, a starting position and x, find the index of the first element >= x.			
def findFirstEltGTE(arr, start, x):
	for i in range(start, len(arr)):
		if (arr[i] >= x):
			return i
	return len(arr)
		
		
		
		