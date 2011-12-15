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
  


# global variables

import _ponzi3_fns as p3_fns
import scipy
from scipy import linspace, linalg, mean, exp, randn, stats, interpolate, integrate, array, meshgrid
import sys

# default values for global parameters
g_default = {		
	'beta' : 0.9,
	'theta' : 0.5,
	'gamma' : 2.0,		
	'depositorSlope' : 0.1, 		# slope of depositor's mean-SD indifference curves		
	'bankruptcyPenalty' : -0.0, 	# bankruptcy penalty (fraction of deficit at bankruptcy)
	'zLow' : 1.0,					# low population shock
	'zHigh' : 1.2,				# high population shock
	'pHigh' : 0.95,				# probability of high population shock
	'M_grid' : (0.05, 4, 80),	# min, max, number of grid points for M grid
	'D_grid' : (0.05, 4, 100),	# min, max, number of grid points for D grid
	'r_grid' : (1, 3, 100),		# min, max, number of grid points for r grid
	'grid_d_size' : 100,			# number of grid points for d grid
	}
# these variable names are the ones that will be included when calling getGlobalParams().  The rest are constructed when setGlobalParams() is called
g_names = [
	'beta',
	'theta',
	'gamma',
	'depositorSlope',
	'bankruptcyPenalty',
	'zLow',
	'zHigh',
	'pHigh',
	'M_grid',
	'D_grid',
	'r_grid',
	'grid_d_size',
]
assert(len(g_default.keys()) == len(g_names))

def setGlobalsFromDict(vars):
	# use this to programatically set global variables
	module = sys.modules[__name__]
	for (name, value) in vars.iteritems():
		setattr(module, name, value)

def init_z_shock(zLow, zHigh, pHi):	
	z_space = [zLow, zHigh]
	z_probs = [1-pHi, pHi]
	z_rv = scipy.stats.rv_discrete(name='z_rv', values=[z_space, z_probs])
	z_draws = array(z_space)

	vars = {
		'z_space': z_space, 
		'z_probs': z_probs, 
		'z_rv': z_rv, 
		'z_draws': z_draws,
	}
	setGlobalsFromDict(vars)
		
def setGlobalParams(init=True, **kwargs):
	vars = {}
	
	# use default values
	if (init):
		for (name, value) in g_default.iteritems():
			vars[name] = value
			
	# set specified parameters
	for (name, value) in kwargs.iteritems():
		if (name not in g_names):
			print("setGlobalParams: unknown parameter", (name, value))
		vars[name] = value
	setGlobalsFromDict(vars)
	
	# these variables should have been set by setGlobalsFromDict
	init_z_shock(zLow, zHigh, pHigh)

	# state space grid
	# variables: M_t (cash reserve), N_t (population), D_t (liabilities)
	# gridM, gridD are triples containing (min, max, size)
	(gridmin_M, gridmax_M, gridsize_M) = M_grid
	(gridmin_D, gridmax_D, gridsize_D) = D_grid
	grid_M = linspace(gridmin_M, gridmax_M, gridsize_M);
	grid_D = linspace(gridmin_D, gridmax_D, gridsize_D);
	(mesh_M, mesh_D) = meshgrid(grid_M, grid_D)
	meshlist_M = mesh_M.ravel()
	meshlist_D = mesh_D.ravel()
	gridshape = (gridsize_M, gridsize_D);

	# controls: d_t (dividends), r_t (interest rate offered).
	# d_t is on same grid as M.
	# r_t is between 1 and 2.  Does it need a grid?  we're going to argmax it... yes	
	(gridmin_r, gridmax_r, gridsize_r) = r_grid
	grid_r = linspace(gridmin_r, gridmax_r, gridsize_r)
	(mesh_M2, mesh_r) = meshgrid(grid_M, grid_r)
	meshlist_M2 = mesh_M2.ravel()
	meshlist_r = mesh_r.ravel()
	vars = {
		'gridmin_M': gridmin_M, 
		'gridmax_M': gridmax_M, 
		'gridsize_M': gridsize_M,
		'gridmin_D': gridmin_D,
		'gridmax_D': gridmax_D,
		'gridsize_D': gridsize_D,
		'grid_M': grid_M,
		'grid_D': grid_D,
		'mesh_M': mesh_M,
		'mesh_D': mesh_D,
		'meshlist_M': meshlist_M,
		'meshlist_D': meshlist_D,
		'gridshape': gridshape,		
		'grid_r': grid_r,
		'mesh_M2': mesh_M2,
		'mesh_r': mesh_r,
		'meshlist_M2': meshlist_M2,
		'meshlist_r': meshlist_r,
	}
	setGlobalsFromDict(vars)
	
	# init c library
	p3_fns.setGlobalParams(theta, beta, gamma, "linear", grid_M, grid_D, array(z_space), array(z_probs), depositorSlope, bankruptcyPenalty, [grid_M, grid_D])

# returns a dict of global params
def getGlobalParams():
	result = {}
	module = sys.modules[__name__]	
	for name in g_names:
		result[name] = getattr(module, name)
	return result

	
	
	
	
	
	
	
	
	
	
	
	
	
