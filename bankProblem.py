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
  


from __future__ import print_function	

import operator
import scipy, time, sys, itertools, scipy.stats
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle as pickle
import gzip, os, glob, shutil
import markup
import table
import pyublas
import multiprocessing

# import c++ modules
import _debugMsg, _maximizer as mx, _myfuncs
import _bankProblem

import bellman	
import lininterp2 as linterp
import plot3d
#import markovChain

# returns a dict that groups xlist by f(x) for each element x in xlist
def groupby(xlist, f):
	result = defaultdict(list)
	[result[f(x)].append(x) for x in xlist]
	return result
	
# globals
class g:
	Grid_M = None
	Grid_S = None
	Grid_P = None
	IterList = []
	Params = None
	ParamSettings = {}
	NIters = 0
	IterResult = None
	
	# grid size, boundaries
	DEF_M_MAX = 4
	DEF_S_MAX = 5
	DEF_P_MAX = 2
	DEF_M_GRID_SIZE = 50
	DEF_S_GRID_SIZE = 40
	DEF_P_GRID_SIZE = 30
	DEF_D_GRID_SIZE = 40
	DEF_SLOW_IN_GRID_SIZE = 40
	DEF_SLOW_IN_FRAC_MAX = 4.0
	# M_MAX = 4
	# S_MAX = 5
	# P_MAX = 2
	# M_GRID_SIZE = 50
	# S_GRID_SIZE = 40
	# P_GRID_SIZE = 30
	D_GRID_SIZE = DEF_D_GRID_SIZE
	SLOW_IN_GRID_SIZE = DEF_SLOW_IN_GRID_SIZE
	SLOW_IN_FRAC_MAX = DEF_SLOW_IN_FRAC_MAX
	
	@staticmethod
	def setGridSize(M=None, S=None, P=None, D=None, frac=None):
		if M: 
			#(g.M_MAX, g.M_GRID_SIZE) = M
			g.Grid_M = scipy.linspace(0, M[0], M[1])
		if S: 
			#(g.S_MAX, g.S_GRID_SIZE) = S
			g.Grid_S = scipy.linspace(0, S[0], S[1])
		if P: 
			#(g.P_MAX, g.P_GRID_SIZE) = P
			g.Grid_P = scipy.linspace(0, P[0], P[1])
		if D: g.D_GRID_SIZE = D
		if frac: (g.SLOW_IN_FRAC_MAX, g.SLOW_IN_GRID_SIZE) = frac
	@staticmethod
	def getGridSize():
		#return {'M': (g.M_MAX, g.M_GRID_SIZE), 'S': (g.S_MAX, g.S_GRID_SIZE), 'P': (g.P_MAX, g.P_GRID_SIZE), 'D': g.D_GRID_SIZE, 'frac': (g.SLOW_IN_GRID_SIZE, g.SLOW_IN_FRAC_MAX)}
		return {
		  'M': (g.Grid_M[-1], len(g.Grid_M)),
		  'S': (g.Grid_S[-1], len(g.Grid_S)),
		  'P': (g.Grid_P[-1], len(g.Grid_P)),
		  'D': g.D_GRID_SIZE, 
		  'frac': (g.SLOW_IN_FRAC_MAX, g.SLOW_IN_GRID_SIZE)
		}
	@staticmethod
	def setDefaultGridSize():
		g.setGridSize(M=(g.DEF_M_MAX, g.DEF_M_GRID_SIZE), S=(g.DEF_S_MAX, g.DEF_S_GRID_SIZE), P=(g.DEF_P_MAX, g.DEF_P_GRID_SIZE), D=g.DEF_D_GRID_SIZE, frac=(g.DEF_SLOW_IN_FRAC_MAX, g.DEF_SLOW_IN_GRID_SIZE))		
		
	# colors of states
	# 0 - bankrupt next period (M_t+1 is always <0)
	# 1 - risky (M_t+1 is <0 in bad shock)
	# 2 - safe (M_t+1 is always >0)
	# 4 - bankrupt now 
	STATE_BANKRUPT = 0
	STATE_RISKY = 1
	STATE_SAFE = 2
	STATE_ABSORBED = 4	
	
	StateToColor = {
	  STATE_BANKRUPT: 'r',
	  STATE_RISKY: 'b',
	  STATE_SAFE: 'g',
	  STATE_ABSORBED: 'k'
	}
	
	# max numbers of value function iterations to try before giving up
	MAX_VITERS = 800
	MAX_TIME = 1000
	MAX_V = 10000
	
	@staticmethod
	def reset():
		g.Grid_M = None
		g.Grid_S = None
		g.Grid_P = None
		g.IterList = []
		g.Params = None
		g.ParamSettings = {}
		g.NIters = 0
		g.IterResult = None
	@staticmethod
	def to_dict():
		dict = {
			'Grid_M': g.Grid_M,
			'Grid_S': g.Grid_S,
			'Grid_P': g.Grid_P,
			'IterList': g.IterList,			
			'ParamSettings': g.ParamSettings,
			'NIters': g.NIters,
			'IterResult': g.IterResult
		}
		return dict
	@staticmethod
	def from_dict(dict):
		g.Grid_M = dict['Grid_M']
		g.Grid_S = dict['Grid_S']
		g.Grid_P = dict['Grid_P']
		g.IterList = dict['IterList']
		g.ParamSettings = dict['ParamSettings']
		g.NIters = dict['NIters']
		g.IterResult = dict['IterResult']
g.setDefaultGridSize()

######################################################################
# save/load optimization results
		
def saveRun(filename, allIters=False):
	output = gzip.open(filename, 'wb')
	dict = g.to_dict()
	if (not allIters):		
		dict['IterList'] = g.IterList[-1:]
	pickle.dump(dict, output)
	output.close()
	
def loadRun(filename):
	pk_file = gzip.open(filename, 'rb')
	dict = pickle.load(pk_file)
	pk_file.close()	
	g.from_dict(dict)
	
def loadRun_1(filename):
	pk_file = gzip.open(filename, 'rb')
	dict = pickle.load(pk_file)
	pk_file.close()
	# change ParamSettings from a list to a dict
	[beta, grid_M, grid_S, grid_P, rFast, rSlow, slowOut, fastOut, fastIn, probSpace] = dict['ParamSettings']
	dict['ParamSettings'] = {'beta':beta, 'grid_M':grid_M, 'grid_S':grid_S, 'grid_P':grid_P, 'rFast':rFast, 'rSlow':rSlow, 'slowOut':slowOut, 
	  'fastOut':fastOut, 'fastIn':fastIn, 'probSpace':probSpace}
	g.from_dict(dict)	

def loadRun_2(filename):
	pk_file = gzip.open(filename, 'rb')
	dict = pickle.load(pk_file)
	pk_file.close()	
	if (dict['NIters'] > 200):
		dict['IterResult'] = bellman.ITER_RESULT_MAX_ITERS
	else:
		dict['IterResult'] = bellman.ITER_RESULT_CONVERGENCE
	g.from_dict(dict)
	
g_LoadFns = {1: loadRun_1, 2: loadRun_2}
def convert_prev_version(filename, version):
	print("loading v1 file: %s" % filename)
	g_LoadFns[version](filename)
	print("saving file: %s" % filename)
	saveRun(filename)

def convert_prev_version_dir(dirname, version, filespec="*.out"):
	pattern = os.path.join(dirname, filespec)
	fileList = glob.glob(pattern)
	for filename in fileList:
		convert_prev_version(filename, version)
	
def resaveSmallFiles(dirname, dir2=None):
	pattern = os.path.join(dirname, "*.out")
	fileList = glob.glob(pattern)
	for filename in fileList:
		file = filename
		outFile = file
		if (dir2 != None):
			base = os.path.basename(file)
			outFile = os.path.join(dir2, base)
		print("%s -> %s" % (file, outFile))
		try:
			loadRun(file)
			saveRun(outFile)
			#print(outFile)
		except StandardError as err:
			print("exception on %s: " % file, err)
			
class BankParams(_bankProblem.BankParams4):
	def __init__(self, beta, rFast, rSlow, slowOut, fastOut, fastIn, probSpace, bankruptcyPenalty, popGrowth, **kwargs):
		super(BankParams,self).__init__(beta, rFast, rSlow, slowOut, fastOut, fastIn, probSpace, bankruptcyPenalty, popGrowth)		
	def getControlGridList(self, stateVarList):
		M = stateVarList[0]
		S = stateVarList[1]
		P = stateVarList[2]
		dGrid = scipy.linspace(0, M, g.D_GRID_SIZE)
		slowInFracGrid = scipy.linspace(0, g.SLOW_IN_FRAC_MAX, g.SLOW_IN_GRID_SIZE)
		return [dGrid, slowInFracGrid]

def triangleDistribution(N):
	if (N % 2 == 0):
		# even
		incrs = scipy.array(range(1, (N/2)+1) + range(N/2, 0, -1)) * 1.0
	else:
		# odd
		incrs = scipy.array(range(1, (N/2)+1) + [N/2 + 1] + range(N/2, 0, -1)) * 1.0
	dy = 1.0 / sum(incrs)
	probSpace = incrs * dy
	# ensure the sum adds up to 1.0
	probSpace[-1] += (1.0 - sum(probSpace))
	return probSpace

# class for generating test cases (varying the parameters)
class TestCaseGenerator:
	def getDefaultParamsDict(self): raise NotImplementedError()
	def getOverrideParamsList(self): raise NotImplementedError()
	def getFilenamePrefix(self, testName, paramsDict): raise NotImplementedError()
	# when we vary 2 parameters, get value1 and value 2
	def getXYName(self): raise NotImplementedError()
	def getXYGraphLabel(self): raise NotImplementedError()
	def getXYShortLabel(self): raise NotImplementedError()
	def getXY(self, paramsDict): raise NotImplementedError()
	
# class for varying interest rates
class VaryR_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, rFastRange=scipy.linspace(0, 0.08, 20), rSlowRange=scipy.linspace(0, 0.15, 20), slowOut=scipy.array([0.1, 0.1]), **kwargs):
		(self.rFastRange, self.rSlowRange) = (rFastRange, rSlowRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.15,
		  'rFast': 0.02,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.7, 0.9]),
		  'slowOut': slowOut,
		  'fastIn':  scipy.array([0.8, 0.8]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])		  
		}
		self.defaultParams.update(kwargs)
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		beta = self.defaultParams['beta']
		# rFast from 0 to 1/beta + 1.0
		# rSlow from 0 to 1/beta + 1.0
		for rFast in self.rFastRange:
			for rSlow in self.rSlowRange:
				tests.append( ("test_r", {'rSlow':rSlow, 'rFast':rFast}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		filename = "%s_beta%.2f_rS%.3f_rF%.3f_pHigh%.2f_outHigh%.2f" % (testName, paramsDict['beta'], paramsDict['rSlow'], paramsDict['rFast'], paramsDict['probSpace'][1], paramsDict['fastOut'][1])
		return filename
	def getXYName(self): return ('rSlow', 'rFast')
	def getXYGraphLabel(self): return ('interest rate on loans', 'interest rate on deposits')
	def getXYShortLabel(self): return (r'$r^L$', r'$r^D$')
	def getXY(self, paramsDict):
		return (paramsDict['rSlow'], paramsDict['rFast'])

# use a different probability space.
class VaryR2_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, rFastRange=scipy.linspace(0, 0.08, 20), rSlowRange=scipy.linspace(0, 0.15, 20), slowOut=scipy.array([0.1, 0.1, 0.1, 0.1]), **kwargs):
		(self.rFastRange, self.rSlowRange) = (rFastRange, rSlowRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.15,
		  'rFast': 0.02,
		  'probSpace': scipy.array([0.25, 0.25, 0.25, 0.25]),
		  'fastOut': scipy.array([0.7, 0.7, 0.9, 0.9]),
		  'slowOut': slowOut,
		  'fastIn':  scipy.array([0.8, 0.8, 0.8, 0.8]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])		  
		}
		self.defaultParams.update(kwargs)
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		beta = self.defaultParams['beta']
		# rFast from 0 to 1/beta + 1.0
		# rSlow from 0 to 1/beta + 1.0
		for rFast in self.rFastRange:
			for rSlow in self.rSlowRange:
				tests.append( ("test_r", {'rSlow':rSlow, 'rFast':rFast}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		filename = "%s_beta%.2f_rS%.3f_rF%.3f_pHigh%.2f_outHigh%.2f" % (testName, paramsDict['beta'], paramsDict['rSlow'], paramsDict['rFast'], paramsDict['probSpace'][1], paramsDict['fastOut'][1])
		return filename
	def getXYName(self): return ('rSlow', 'rFast')
	def getXYGraphLabel(self): return ('interest rate on loans', 'interest rate on deposits')
	def getXYShortLabel(self): return (r'$r^L$', r'$r^D$')
	def getXY(self, paramsDict):
		return (paramsDict['rSlow'], paramsDict['rFast'])

		
# vary pHigh and out frac variance
class VaryOutFracPHigh_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, pHighRange=scipy.linspace(0., 1., 20), outFracLowRange=scipy.linspace(0.3, 0.9, 20)):
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.15,
		  'rFast': 0.10,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.7, 0.9]),
		  'slowOut': scipy.array([0.1, 0.1]),
		  'fastIn':  scipy.array([0.8, 0.8]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])
		}
		(self.pHighRange, self.outFracLowRange) = (pHighRange, outFracLowRange)
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		for pHigh in self.pHighRange:
			for outFracLow in self.outFracLowRange:
				probSpace = scipy.array([1.0-pHigh, pHigh])
				fastOut = scipy.array([outFracLow, 0.9])
				tests.append( ("test_var_pHigh", {'probSpace':probSpace, 'fastOut':fastOut}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		filename = "%s_pHigh%.2f_fastOut%.2f" % (testName, paramsDict['probSpace'][1], paramsDict['fastOut'][0])
		return filename
	def getXYName(self): return ('pHigh', 'fastOutFracLow')	
	def getXYGraphLabel(self): return (r'probability of high $\alpha^D_t$', r'low realization of $\alpha^D_t$')
	def getXY(self, paramsDict):
		return (paramsDict['probSpace'][1], paramsDict['fastOut'][0])

# mean_log_growth is the expected log growth of deposits _Without_ considering insurance (i.e. from new deposits and withdrawals only)
# we construct fastOut such that low log growth is (mean_log_growth-delta) in the low case, (mean_log_growth+delta) in the high case
def fastOutFrac(inFrac, mean_log_growth, delta):
	alpha_high = 1.0 + inFrac - scipy.exp(mean_log_growth - delta)
	alpha_low = 1.0 + inFrac - scipy.exp(mean_log_growth + delta)
	g_low = 1.0 - alpha_high + inFrac
	g_high = 1.0 - alpha_low + inFrac
	#print(scipy.sqrt(g_low * g_high))
	return [alpha_low, alpha_high]
	
class VaryDelta_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, rFastRange=scipy.linspace(0, 0.15, 16), mean_log_growth=0.0, delta_log_growth_gridsize=20, fastInFrac=0.8):	
		(self.beta, self.rFastRange, self.mean_log_growth, self.delta_log_growth_gridsize, self.fastInFrac) = (beta, rFastRange, mean_log_growth, delta_log_growth_gridsize, fastInFrac)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.15,
		  'rFast': 0.,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.7, 0.9]),
		  'slowOut': scipy.array([0.1, 0.1]),
		  'fastIn':  scipy.array([fastInFrac, fastInFrac]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])
		}
		# lowest delta is 0
		# highest delta is when alpha (out frac) = 0
		self.delta_log_growth_max = mean_log_growth - scipy.log(1 - 1 + fastInFrac)		
		self.delta_log_growth_range = scipy.linspace(0, self.delta_log_growth_max, self.delta_log_growth_gridsize)
		self.delta_log_growth_range[0] = 0.0001
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		fastInFrac = self.defaultParams['fastIn'][0]
		for rFast in self.rFastRange:
			for delta_log_growth in self.delta_log_growth_range:
				fastOut = scipy.array(fastOutFrac(fastInFrac, self.mean_log_growth, delta_log_growth))
				assert(fastOut[0] <= fastOut[1])
				tests.append( ("test_delta", {'rFast':rFast, 'fastOut':fastOut, 'delta':delta_log_growth}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		rFast = paramsDict['rFast']
		fastOut = paramsDict['fastOut']
		delta = paramsDict['delta']
		filename = "%s_rFast%.3f_delta%.3f_outFrac%.3f_%.3f" % (testName, rFast, delta, fastOut[0], fastOut[1])
		return filename
	def getXYName(self): return ('delta', 'rFast')
	def getXYGraphLabel(self): return (r'$\delta$', 'interest rate on deposits')
	def getXY(self, paramsDict):
		rFast = paramsDict['rFast']
		delta = paramsDict['delta']	
		return (delta, rFast)

# use a simple arithmetic mean. alpha_low = alpha_mean - delta, alpha_high = alpha_mean + delta
class VaryDelta2_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, rSlow=0.15, rFastRange=scipy.linspace(0, 0.15, 5), alphaMean=0.8, deltaRange=scipy.linspace(0, 0.2, 5), fastInFrac=0.8, slowOut=scipy.array([0.1, 0.1]), **kwargs):	
		(self.beta, self.rFastRange, self.alphaMean, self.deltaRange, self.fastInFrac) = (beta, rFastRange, alphaMean, deltaRange, fastInFrac)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': rSlow,
		  'rFast': 0.,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.7, 0.9]),
		  'slowOut': slowOut,
		  'fastIn':  scipy.array([fastInFrac, fastInFrac]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])
		}
		self.defaultParams.update(kwargs)
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		fastInFrac = self.defaultParams['fastIn'][0]
		for rFast in self.rFastRange:
			for delta in self.deltaRange:
				fastOut = scipy.array([self.alphaMean - delta, self.alphaMean + delta])
				assert(fastOut[0] <= fastOut[1])
				tests.append( ("test_delta", {'rFast':rFast, 'fastOut':fastOut, 'delta':delta}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		rFast = paramsDict['rFast']
		fastOut = paramsDict['fastOut']
		delta = paramsDict['delta']
		filename = "%s_rFast%.3f_delta%.3f_outFrac%.3f_%.3f" % (testName, rFast, delta, fastOut[0], fastOut[1])
		return filename
	def getXYName(self): return ('delta', 'rFast')
	def getXYGraphLabel(self): return (r'$\delta$ ($\alpha^D_t = \bar{\alpha^D} \pm \delta$)', 'interest rate on deposits')
	def getXYShortLabel(self): return (r'$\delta$', r'$r^D$')
	def getXY(self, paramsDict):
		rFast = paramsDict['rFast']
		delta = paramsDict['delta']	
		return (delta, rFast)
		
class VaryDuration_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, delta=0.0, rSlow=0, rFast=0, inFracFastRange=scipy.linspace(0.1, 0.9, 20), outFracSlowRange=scipy.linspace(0.1, 0.8, 20)):
		(self.beta, self.delta, self.inFracFastRange, self.outFracSlowRange) = (beta, delta, inFracFastRange, outFracSlowRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': rSlow,
		  'rFast': rFast,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.8, 0.8]),
		  'slowOut': scipy.array([0.1, 0.1]),
		  'fastIn':  scipy.array([0.7, 0.7]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])		   
		}
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		for inFracFast in self.inFracFastRange:
			for outFracSlow in self.outFracSlowRange:
				
				fastIn = scipy.array([inFracFast, inFracFast])
				slowOut = scipy.array([outFracSlow, outFracSlow])
				(alpha_low, alpha_high) = fastOutFrac(inFracFast, mean_log_growth=0.0, delta=self.delta)
				fastOut = scipy.array([alpha_low, alpha_high])
				tests.append( ("test_duration", {'fastIn': fastIn, 'slowOut': slowOut, 'fastOut': fastOut}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		fastIn = paramsDict['fastIn'][0]
		slowOut = paramsDict['slowOut'][0]
		filename = "%s_fastIn%0.3f_slowOut%0.3f" % (testName, fastIn, slowOut)
		return filename
	def getXYName(self): return ('fastIn', 'slowOut')
	def getXYGraphLabel(self): return (r'$\gamma^L$ (new deposits)', r'$\alpha^D$ (fraction of loans repaid)')	
	def getXY(self, paramsDict):
		fastIn = paramsDict['fastIn'][0]
		slowOut = paramsDict['slowOut'][0]
		return (fastIn, slowOut)		

class VaryDuration2_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, slowOutRange=scipy.linspace(0.2, 0.6, 5), rFastRange=scipy.linspace(0.02, 0.15, 5)):
		(self.beta, self.slowOutRange, self.rFastRange) = (beta, slowOutRange, rFastRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.13,
		  'rFast': 0.,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.8, 0.8]),
		  'slowOut': scipy.array([0.4, 0.4]),
		  'fastIn':  scipy.array([0.7, 0.7]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])		   
		}
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		for rFast in self.rFastRange:
			for slowOutFrac in self.slowOutRange:
				slowOut = scipy.array([slowOutFrac, slowOutFrac])
				tests.append( ("test_duration", {'slowOut': slowOut, 'rFast':rFast}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		rFast = paramsDict['rFast']
		slowOut = paramsDict['slowOut'][0]
		filename = "%s_rF%0.3f_slowOut%0.3f" % (testName, rFast, slowOut)
		return filename
	def getXYName(self): return ('slowOutFrac', 'rFast')
	def getXYGraphLabel(self): return (r'$\alpha^L$ (expected duration=$\frac{1}{\alpha^L}$', 'interest on deposits')
	def getXYShortLabel(self): return (r'$\alpha^L$', r'$r^D$')
	def getXY(self, paramsDict):
		rFast = paramsDict['rFast']
		slowOut = paramsDict['slowOut'][0]
		return (slowOut, rFast)		
	
# duration vs. uncertainty
class VaryDuration3_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, slowOutRange=scipy.linspace(0.2, 0.6, 5), alphaMean=0.8, deltaRange=scipy.linspace(0, 0.2, 5), rFast=0.0):
		(self.beta, self.slowOutRange, self.alphaMean, self.deltaRange) = (beta, slowOutRange, alphaMean, deltaRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.13,
		  'rFast': rFast,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.8, 0.8]),
		  'slowOut': scipy.array([0.4, 0.4]),
		  'fastIn':  scipy.array([0.7, 0.7]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])		   
		}
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []
		for delta in self.deltaRange:
			for slowOutFrac in self.slowOutRange:
				fastOut = scipy.array([self.alphaMean - delta, self.alphaMean + delta])
				slowOut = scipy.array([slowOutFrac, slowOutFrac])
				tests.append( ("test_duration2", {'slowOut': slowOut, 'fastOut':fastOut, 'delta':delta}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		fastOut = paramsDict['fastOut']
		delta = paramsDict['delta']		
		slowOut = paramsDict['slowOut'][0]
		filename = "%s_delta%0.3f_slowOut%0.3f_fastOut%0.3f_%0.3f" % (testName, delta, slowOut, fastOut[0], fastOut[1])
		return filename
	def getXYName(self): return ('slowOutFrac', 'delta')
	def getXYGraphLabel(self): return (r'$\alpha^L$ (expected duration=$\frac{1}{\alpha^L}$', r'$\delta$')
	def getXYShortLabel(self): return (r'$\alpha^L$', r'$\delta$')
	def getXY(self, paramsDict):
		delta = paramsDict['delta']
		slowOut = paramsDict['slowOut'][0]
		return (slowOut, delta)		

# vary pHigh and out frac variance
class VaryP_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, pHighRange=scipy.linspace(0.5, 1., 5), alphaMean=0.8, deltaRange=scipy.linspace(0, 0.2, 5), rFast=0.0, rSlow=0.13, slowOut=scipy.array([0.4, 0.4])):
		(self.beta, self.pHighRange, self.alphaMean, self.deltaRange) = (beta, pHighRange, alphaMean, deltaRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': rSlow,
		  'rFast': rFast,
		  'probSpace': scipy.array([0.5, 0.5]),
		  'fastOut': scipy.array([0.7, 0.9]),
		  'slowOut': slowOut,
		  'fastIn':  scipy.array([0.8, 0.8]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])
		}		
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []		
		for delta in self.deltaRange:	
			for pHigh in self.pHighRange:
				fastOut = scipy.array([self.alphaMean - delta, self.alphaMean + delta])
				probSpace = scipy.array([1.0-pHigh, pHigh])				
				tests.append( ("test_pHigh", {'probSpace':probSpace, 'fastOut':fastOut, 'delta':delta}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		delta = paramsDict['delta']
		filename = "%s_pHigh%.3f_delta%0.3f_fastOut%.3f_%3f" % (testName, paramsDict['probSpace'][1], delta, paramsDict['fastOut'][0], paramsDict['fastOut'][1])
		return filename
	def getXYName(self): return ('pHigh', 'delta')	
	def getXYGraphLabel(self): return (r'$P(\alpha_{high})$', r'$\delta$')
	def getXYShortLabel(self): return (r'$P(\alpha_{high})$', r'$\delta$')
	def getXY(self, paramsDict):
		delta = paramsDict['delta']
		pHigh = paramsDict['probSpace'][1]
		return (pHigh, delta)

# add uncertainty in loans
class VaryL_Generator(TestCaseGenerator):
	def __init__(self, beta=0.9, L_alphaMean=0.4, L_deltaRange=scipy.linspace(0, 0.2, 5), D_alphaMean=0.8, D_deltaRange=scipy.linspace(0, 0.2, 5), rFast=0.0):
		(self.beta, self.L_alphaMean, self.L_deltaRange, self.D_alphaMean, self.D_deltaRange) = (beta, L_alphaMean, L_deltaRange, D_alphaMean, D_deltaRange)
		self.defaultParams = {
		  'beta': beta,
		  'rSlow': 0.13,
		  'rFast': rFast,
		  'probSpace': scipy.array([0.25, 0.25, 0.25, 0.25]),
		  'fastOut': scipy.array([0.7, 0.9]),
		  'slowOut': scipy.array([0.1, 0.1]),
		  'fastIn':  scipy.array([0.8, 0.8, 0.8, 0.8]),
		  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0])
		}		
	def getDefaultParamsDict(self): return self.defaultParams
	def getOverrideParamsList(self):
		tests = []		
		for D_delta in self.D_deltaRange:	
			for L_delta in self.L_deltaRange:	
				fastOut = scipy.array([self.D_alphaMean - D_delta, self.D_alphaMean + D_delta, self.D_alphaMean - D_delta, self.D_alphaMean + D_delta])
				slowOut = scipy.array([self.L_alphaMean - L_delta, self.L_alphaMean - L_delta, self.L_alphaMean + L_delta, self.L_alphaMean + L_delta])
				tests.append( ("test_Ldelta", {'slowOut':slowOut, 'fastOut':fastOut, 'L_delta':L_delta, 'D_delta':D_delta}) )
		return tests
	def getFilenamePrefix(self, testName, paramsDict):	
		filename = "%s_Ldelta_%0.3f_Ddelta_%0.3f" % (testName, paramsDict['L_delta'], paramsDict['D_delta'])
		return filename
	def getXYName(self): return ('L_delta', 'D_delta')	
	
	r'$\delta^L$ ($\alpha^L_t = \bar{\alpha^L} \pm \delta^L$)'
	r'$\delta^D$ ($\alpha^D_t = \bar{\alpha^D} \pm \delta^D$)'
	
	def getXYGraphLabel(self): return (r'$\delta^L$ ($\alpha^L_t =\ \bar{\alpha}^L \pm\ \delta^L$)', r'$\delta^D$ ($\alpha^D_t =\ \bar{\alpha}^D \pm\ \delta^D$)')
	def getXYShortLabel(self): return (r'$\delta^L$', r'$\delta^D$')
	def getXY(self, paramsDict):
		L_delta = paramsDict['L_delta']
		D_delta = paramsDict['D_delta']		
		return (L_delta, D_delta)
		

g_TestR = VaryR_Generator(rFastRange=scipy.linspace(0, 0.15, 5), rSlowRange=scipy.linspace(0, 0.15, 5))
g_TestR2 = VaryR_Generator(rFastRange=scipy.linspace(0, 0.15, 5), rSlowRange=scipy.linspace(0, 0.15, 5), slowOut=scipy.array([0.4, 0.4]))
g_TestR3 = VaryR_Generator(rFastRange=scipy.linspace(0.09, 0.13, 5), rSlowRange=scipy.linspace(0.09, 0.13, 5), slowOut=scipy.array([0.4, 0.4]))
g_TestR4 = VaryR_Generator(rFastRange=scipy.linspace(0.1125, 0.15, 5), rSlowRange=scipy.linspace(0.1125, 0.15, 5), slowOut=scipy.array([0.4, 0.4]))
g_TestR5 = VaryR_Generator(rFastRange=scipy.array([0.03, 0.05, 0.1, 0.12]), rSlowRange=scipy.linspace(0.1, 0.16, 25), slowOut=scipy.array([0.4, 0.4]))
# R6 is same as R5 with a larger grid
g_TestR6 = VaryR_Generator(rFastRange=scipy.array([0.03, 0.05, 0.1, 0.12]), rSlowRange=scipy.linspace(0.1, 0.16, 25), slowOut=scipy.array([0.4, 0.4]), gridSizeDict={'S': (10, 80)})

# uses a different prob. distribution
g_Test2R = VaryR2_Generator(rFastRange=scipy.linspace(0.09, 0.13, 5), rSlowRange=scipy.linspace(0.09, 0.13, 5), slowOut=scipy.array([0.4, 0.4, 0.4, 0.4]))

g_TestDelta = VaryDelta2_Generator()
g_TestDelta2 = VaryDelta2_Generator(rSlow=0.12, rFastRange=scipy.linspace(0.02, 0.12, 11), deltaRange=scipy.linspace(0, 0.2, 11))
g_TestDelta3 = VaryDelta2_Generator(rSlow=0.12, rFastRange=scipy.linspace(0.02, 0.12, 6), deltaRange=scipy.linspace(0, 0.2, 6), slowOut=scipy.array([0.4, 0.4]))
g_TestDelta4 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.linspace(0.00, 0.08, 6), deltaRange=scipy.linspace(0, 0.12, 6), slowOut=scipy.array([0.4, 0.4]))
g_TestDelta5 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.linspace(0.048, 0.064, 5), deltaRange=scipy.linspace(0.04, 0.1, 5), slowOut=scipy.array([0.4, 0.4]))
g_TestDelta6 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.linspace(0.056, 0.064, 5), deltaRange=scipy.linspace(0.07, 0.1, 5), slowOut=scipy.array([0.4, 0.4]))
# delta7 is designed to center on rD=0.06 and delta=0.085
g_TestDelta7 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.array([0.02, 0.04, 0.06, 0.08]), deltaRange=scipy.linspace(0, 0.17, 19), slowOut=scipy.array([0.4, 0.4]))
g_TestDelta8 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.linspace(0.0, 0.14, 15), deltaRange=scipy.linspace(0, 0.17, 19)[2:-2], slowOut=scipy.array([0.4, 0.4]))
# delta9 is the same as delta6, with a larger grid for S
g_TestDelta9 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.linspace(0.056, 0.064, 5), deltaRange=scipy.linspace(0.07, 0.1, 5), slowOut=scipy.array([0.4, 0.4]), gridSizeDict={'S': (10, 80)})
g_TestDelta10 = VaryDelta2_Generator(rSlow=0.13, rFastRange=scipy.linspace(0.056, 0.064, 5), deltaRange=scipy.linspace(0.07, 0.1, 5), slowOut=scipy.array([0.4, 0.4]), gridSizeDict={'S': (15, 120)})

g_TestDuration = VaryDuration2_Generator()

g_Test2Duration = VaryDuration3_Generator()

g_TestP = VaryP_Generator()
g_TestP2 = VaryP_Generator(rSlow=0.15, pHighRange=scipy.linspace(0.1, 0.9, 5))
g_TestP3 = VaryP_Generator(rSlow=0.1, pHighRange=scipy.linspace(0.1, 0.9, 5))
g_TestP4 = VaryP_Generator(rSlow=0.12, pHighRange=scipy.linspace(0.1, 0.9, 5))

g_TestL = VaryL_Generator()
g_TestL2 = VaryL_Generator(L_deltaRange=scipy.linspace(0, 0.2, 10), D_deltaRange=scipy.linspace(0, 0.2, 10))

# What else is there?
#  - defaults?
#  - other prob. distributions
#  - supply & demand curves












SOLUTION_CATEGORY_NOINVEST = 0
SOLUTION_CATEGORY_INVEST = 1
SOLUTION_CATEGORY_NONCONVERGENCE = 2

def write_test_summary_2d(dirname, testGenObj, outFilename="solutions.out"):	
	(xList, yList, outcomeList) = ([], [], [])	
	for (testName, overrideParams) in testGenObj.getOverrideParamsList():			
		newParams = dict(testGenObj.getDefaultParamsDict())
		newParams.update(overrideParams)
		prefix = testGenObj.getFilenamePrefix(testName, newParams)
		outPath = os.path.join(dirname, prefix) + ".out"
		if (not os.path.exists(outPath)):
			print ("not found: %s" % outPath)
			continue
		try:
			loadRun(outPath)
			(x, y) = testGenObj.getXY(newParams)
			print(x,y)
			currentVArray = g.IterList[-1]['V']
			optControl_inFrac = g.IterList[-1]['fracIn']
			# check if optimal in fraction is always zero
			max_inFrac = scipy.amax(optControl_inFrac)
			bZeroFrac = (max_inFrac == 0.0)
			# check nonconvergence						
			bNonConverge = (g.IterResult != bellman.ITER_RESULT_CONVERGENCE)			
			# outcome: 0 - zero inFrac, 1 - nonzero inFrac, 2 - nonconvergence
			if (bNonConverge):
				outcome = SOLUTION_CATEGORY_NONCONVERGENCE
			else:
				if (bZeroFrac):
					outcome = SOLUTION_CATEGORY_NOINVEST
				else:
					outcome = SOLUTION_CATEGORY_INVEST
			xList.append(x)
			yList.append(y)
			outcomeList.append(outcome)
		except AssertionError as err:
			print("exception on %s: " % outPath, err)
	sum_dict = {}	
	g_dict = g.to_dict()
	sum_dict['g'] = g_dict
	(sum_dict['xName'], sum_dict['yName']) =  testGenObj.getXYName()
	sum_dict['yList'] = yList
	sum_dict['xList'] = xList
	sum_dict['outcomeList'] = outcomeList	
	f = gzip.open(os.path.join(dirname, outFilename), 'wb')
	pickle.dump(sum_dict, f)
	f.close()

def plot_test_summary_2d(dirname, testGenObj, summaryFilename="solutions.out", addLines1=False):
	(xlabel, ylabel) = testGenObj.getXYGraphLabel()
	f = gzip.open(os.path.join(dirname, summaryFilename), 'rb')
	dict = pickle.load(f)
	f.close()
	[xName, yName, xList, yList, outcomeList] = [dict[x] for x in ['xName', 'yName', 'xList', 'yList', 'outcomeList']]
	g_dict = dict['g']
	beta = g_dict['ParamSettings']['beta']
	print("beta from %s: %f" % (summaryFilename, beta))
	page = markup.page()
	page.init(title="2d summary of solutions")	
	
	# go from xList, yList to table.
	xValues = sorted(list(set(xList)))
	yValues = reversed(sorted(list(set(yList))))
	xy_to_outcome = {}
	xy_to_i = {}
	for (i, (x, y, outcome)) in enumerate(zip(xList, yList, outcomeList)):
		xy_to_outcome[(x,y)] = outcome
		xy_to_i[(x,y)] = i
	page.table.open()
	# header row
	page.tr.open()
	page.th("")
	for x in xValues:
		page.th("%0.2f" % x)
	page.tr.close()	
	for y in yValues:
		page.tr.open()
		# header col
		page.th("%0.2f" % y)
		for x in xValues:
			o = xy_to_outcome[(x,y)]
			page.td.open()
			#page.font(markup.oneliner.a("*", href="index.html#%d" % i), color=outcome_to_color(o))
			page.a(markup.oneliner.font("*", color=outcome_to_color(o)), href="index.html#%d" % xy_to_i[(x,y)])
			page.td.close()			
		page.tr.close()
	page.table.close()

	# color graph
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	ax.set_xlabel(xlabel if (xlabel != None) else xName)
	ax.set_ylabel(ylabel if (ylabel != None) else yName)
	colors = [outcome_to_color(o) for o in outcomeList]
	ax.scatter(xList, yList, color=colors, edgecolor=colors)
	page.br()
	imgFilename = "2d_summary.png"
	imgPath = os.path.join(dirname, imgFilename)
	try:                                        
		os.remove(imgPath)
	except OSError:        
		pass   
	plt.savefig(imgPath, format='png', dpi=120)					
	page.img(src=imgFilename)               

	# non-color graph
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	ax.set_xlabel(xlabel if (xlabel != None) else xName)
	ax.set_ylabel(ylabel if (ylabel != None) else yName)
	(markers_labels) = [outcome_to_marker(o) for o in outcomeList]
	by_marker = defaultdict(list)
	[by_marker[marker_label].append((x,y)) for (x,y,marker_label) in zip(xList, yList, markers_labels)]
	for ((marker, label), xyList) in by_marker.items():
		(x_list, y_list) = zip(*xyList)
		ax.scatter(x_list, y_list, marker=marker, label=label, facecolor='none')
	(handles, labels) = ax.get_legend_handles_labels()
	hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
	(handles2, labels2) = zip(*hl)	
	fig.legend(handles2, labels2, 'upper center')	
	if (addLines1): 
		ax.axvline((1.0/beta)-1.0, color='gray')
		ax.plot([xList[0], xList[-1]], [yList[0], yList[-1]], color='gray')
	page.br()
	imgFilename = "2d_summary2.png"
	imgPath = os.path.join(dirname, imgFilename)
	try:                                        
		os.remove(imgPath)
	except OSError:        
		pass   
	plt.savefig(imgPath, format='png', dpi=120)					
	page.img(src=imgFilename)               
	
	filename = os.path.join(dirname, "2d_summary.html")
	f = open(filename, 'w')                             
	f.write(str(page))
	f.close()
	
	return (xList, yList, outcomeList)
	
def outcome_to_color(outcome):
	if (outcome == SOLUTION_CATEGORY_NOINVEST): return 'black'
	if (outcome == SOLUTION_CATEGORY_INVEST): return 'green'
	if (outcome == SOLUTION_CATEGORY_NONCONVERGENCE): return 'red'
	assert(false)

def outcome_to_marker(outcome):
	if (outcome == SOLUTION_CATEGORY_NOINVEST): return ('o', "no loans made")
	if (outcome == SOLUTION_CATEGORY_INVEST): return ('+', "loans made")
	if (outcome == SOLUTION_CATEGORY_NONCONVERGENCE): return ('s', "nonconvergence")
	assert(false)
	
def run_all(dirname, testGenObj):
	result1 = run_test_cases(dirname, testGenObj)
	result2 = generate_plots(dirname, testGenObj)
	result3 = write_test_summary_2d(dirname, testGenObj)
	result4 = plot_test_summary_2d(dirname, testGenObj)
	(filename, xName, yName, xList, yList, outcomes) = simulate_summary(dirname, testGenObj)
	result6 = plot_simulation_summary(dirname, testGenObj, filename)
	
def run_test_cases(dirname, testGenObj, skipWrite=False, skipIfExists=True, usePrevVArray=True, nMaxIters=g.MAX_VITERS, maxTime=g.MAX_TIME, maxV=g.MAX_V):	
	currentVArray = None
	prevRunConverged = False
	for (testName, overrideParams) in testGenObj.getOverrideParamsList():			
		newParams = dict(testGenObj.getDefaultParamsDict())
		newParams.update(overrideParams)
		prefix = testGenObj.getFilenamePrefix(testName, newParams)
		path = os.path.join(dirname, prefix) + ".out"
		print(path)
		if (skipIfExists and os.path.exists(path)):
			print("%s exists, skipping" % path)
			continue
		g.reset()
		if ((not usePrevVArray) or (prevRunConverged == False)): currentVArray = None;		# if the previous optimization didn't converge, start from scratch, otherwise, start from previous optimized value
		(iterCode, prevNIter, currentVArray, newVArray, optControls) = test_bank2(plotResult=False, initialVArray=currentVArray, nMaxIters=nMaxIters, maxTime=g.MAX_TIME, maxV=g.MAX_V, overrideParamsDict=newParams)
		prevRunConverged = True if (iterCode == bellman.ITER_RESULT_CONVERGENCE) else False
		if (not skipWrite):
			saveRun(path)

def generate_one_plot(arg):
	(dirname, outPath, prefix, caption, skipIfExists) = arg
	if (not os.path.exists(outPath)):
		print ("not found: %s" % outPath)
		return None
	try:
		plt.ioff()
		loadRun(outPath)
		#G = createGraph(-1)
		# save plots						
		suffixes = ["-V", "-optD", "-inF"]
		plotFns = [plotV, plotOptD, plotFracIn]
		imgList = []
		for (suffix, plotFn) in zip(suffixes, plotFns):
			plotFn(-1, len(g.Grid_P)/2)
			imgFilename = prefix + suffix + ".png"
			imgPath = os.path.join(dirname, imgFilename)
			if (not skipIfExists):
				try:                                        
					os.remove(imgPath)
				except OSError:        
					pass   
				print("writing %s" % imgPath) 
				plt.savefig(imgPath, format='png', dpi=120)				  
			imgList.append(imgFilename)	
		plt.close('all')
		plt.ion()
		return (caption, imgList)
	except AssertionError as err:
		print("exception on %s: " % outPath, err)
		return None
		
def generate_plots(dirname, testGenObj, skipIfExists=False, multiprocess=True, nProcesses=6):
	argList = []
	for (testName, overrideParams) in testGenObj.getOverrideParamsList():			
		newParams = dict(testGenObj.getDefaultParamsDict())
		newParams.update(overrideParams)
		prefix = testGenObj.getFilenamePrefix(testName, newParams)
		outPath = os.path.join(dirname, prefix) + ".out"
		caption = "%s: beta=%f, rSlow=%f, rFast=%f, diff=%f, probSpace=[%f, %f], fastOutFraction=[%f, %f]" % (testName, newParams['beta'], newParams['rSlow'], newParams['rFast'], newParams['rSlow']-newParams['rFast'], newParams['probSpace'][0], newParams['probSpace'][1], newParams['fastOut'][0], newParams['fastOut'][1])
		argList.append((dirname, outPath, prefix, caption, skipIfExists))
		
	if (multiprocess):
		p = multiprocessing.Pool(nProcesses)		
		plotList = p.map(generate_one_plot, argList)
		p.close()
	else:		
		plotList = map(generate_one_plot, argList)
			                                               
	# write html index
	page = markup.page()
	page.init(title="bankProblem")
	page.br( )
						
	for (i, plot) in enumerate(plotList):
		(prefix, imgList) = plot
		page.p(prefix, id="%d" % i)
		for imgFile in imgList:
			page.img(src=imgFile)             
	filename = os.path.join(dirname, "index.html")
	f = open(filename, 'w')                             
	f.write(str(page))
	f.close()
	
# run with current parameters
def test_bank0(**kwargs):
	overrideParamsDict = dict(g.ParamSettings)
	del overrideParamsDict['grid_M']
	del overrideParamsDict['grid_S']
	del overrideParamsDict['grid_P']
	return test_bank2(overrideParamsDict=overrideParamsDict, **kwargs)
	
def test_bank1(beta=0.9, rSlow=0.15, rFast=0.02, probSpace=scipy.array([0.5, 0.5]), fastOut=scipy.array([0.7, 0.9]), 
  slowOut=scipy.array([0.1, 0.1]), fastIn=scipy.array([0.8, 0.8]), bankruptcyPenalty=scipy.array([0.0, 0.0, 0.0]), popGrowth=1.0, gridSizeDict=None, **kwargs):
	overrideParamsDict = {'beta':beta, 'rSlow':rSlow, 'rFast':rFast, 'probSpace':probSpace, 'fastOut':fastOut, 'slowOut':slowOut, 'fastIn':fastIn, 'bankruptcyPenalty':bankruptcyPenalty, 'popGrowth':popGrowth, 'gridSizeDict':gridSizeDict}
	return test_bank2(overrideParamsDict=overrideParamsDict, **kwargs)
	
def test_bank2(useValueIter=True, plotResult=True, nMaxIters=g.MAX_VITERS, initialVArray=None, nMultiGrid=2, overrideParamsDict=None, **kwargs):
	time1 = time.time()
	localvars = {}
	
	def postVIterCallbackFn(nIter, currentVArray, newVArray, optControls, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter
		# append iteration results to g.IterList
		g.IterList.append({'V': currentVArray, 'd': optControls[0], 'fracIn': optControls[1]})
		g.NIters += 1

	def postPIterCallbackFn(nIter, newVArray, currentPolicyArrayList, greedyPolicyList, stoppingResult):				
		(stoppingDecision, diff) = stoppingResult
		print("iter %d, diff %f" % (nIter, diff))
		localvars[0] = nIter		

	g.setDefaultGridSize()
	if ('gridSizeDict' in overrideParamsDict and overrideParamsDict['gridSizeDict'] != None):
		g.setGridSize(**overrideParamsDict['gridSizeDict'])		
		# TODO: fix this
	(grid_M, grid_S, grid_P) = (g.Grid_M, g.Grid_S, g.Grid_P)
	
	defaultParamsDict = {
	  'beta': 0.9,
	  'rSlow': 0.15,
	  'rFast': 0.10,
	  'probSpace': scipy.array([0.5, 0.5]),
	  'fastOut': scipy.array([0.7, 0.9]),
	  'slowOut': scipy.array([0.1, 0.1]),
	  'fastIn':  scipy.array([0.8, 0.8]),
	  'bankruptcyPenalty': scipy.array([0.0, 0.0, 0.0]),
	  'popGrowth': 1.0
	}
	paramsDict = dict(defaultParamsDict)
	paramsDict.update(overrideParamsDict)
	[beta, rSlow, rFast, probSpace, fastOut, slowOut, fastIn, bankruptcyPenalty, popGrowth] = [paramsDict[x] for x in ['beta', 'rSlow', 'rFast', 'probSpace', 'fastOut', 'slowOut', 'fastIn', 'bankruptcyPenalty', 'popGrowth']]
	print("using params: beta=%f rFast=%f rSlow=%f" % (beta, rFast, rSlow))
	print("probSpace: ", probSpace, " slowOut: ", slowOut, " fastOut: ", fastOut, " fastIn: ", fastIn, " bp: ", bankruptcyPenalty, " popGrowth: ", popGrowth)
	print("M grid: ", (grid_M[-1], len(grid_M)), " S grid: ", (grid_S[-1], len(grid_S)), " P grid: ", (grid_P[-1], len(grid_P)), "d grid size: %d inFrac grid: " % g.D_GRID_SIZE, (g.SLOW_IN_FRAC_MAX, g.SLOW_IN_GRID_SIZE))
	g.ParamSettings = {'beta':beta, 'grid_M':grid_M, 'grid_S':grid_S, 'grid_P':grid_P, 'rFast':rFast, 'rSlow':rSlow, 'slowOut':slowOut, 
	  'fastOut':fastOut, 'fastIn':fastIn, 'probSpace':probSpace, 'bankruptcyPenalty':bankruptcyPenalty, 'popGrowth':popGrowth}
	#beta, rFast, rSlow, slowOut, fastOut, fastIn, probSpace
	params = BankParams(**g.ParamSettings)
	g.Params = params
		
	# initial guess for V: V = M
	if (initialVArray == None):
		initialVArray = scipy.zeros((len(grid_M), len(grid_S), len(grid_P)))
		for (iM, M) in enumerate(grid_M):
			for (iS, S) in enumerate(grid_S):
				for (iP, P) in enumerate(grid_P):
					initialVArray[iM, iS, iP] = M
			#initialVArray[iM,:,:] = M		
			
	g.IterList.append({'V':initialVArray, 'd':None, 'fracIn':None})
	if (useValueIter == True):
		if (nMultiGrid == None):			
			result = bellman.grid_valueIteration([grid_M, grid_S, grid_P], initialVArray, params, postIterCallbackFn=postVIterCallbackFn, parallel=True, nMaxIters=nMaxIters, **kwargs)
		else:
			# start with coarse grids and progressively get finer.			
			gridList = [grid_M, grid_S, grid_P]
			coarseGridList = []
			for n in reversed(range(nMultiGrid)):
				coarseGridList.append( [scipy.linspace(grid[0], grid[-1], len(grid)/(2**n)) for grid in gridList] )
			result = bellman.grid_valueIteration2(coarseGridList, gridList, initialVArray, params, postIterCallbackFn=postVIterCallbackFn, parallel=True, nMaxIters=nMaxIters, **kwargs)
		(iterCode, nIter, currentVArray, newVArray, optControls) = result
		g.IterResult = iterCode
	else:
		initialPolicyArrayList = bellman.getGreedyPolicy([grid_M, grid_S, grid_P], initialVArray, params, parallel=True)
		result = bellman.grid_policyIteration([grid_M, grid_S, grid_P], initialPolicyArrayList, initialVArray, params, postIterCallbackFn=postPIterCallbackFn, parallel=True)
		(iterCode, nIter, currentVArray, currentPolicyArrayList, greedyPolicyList) = result
		newVArray = currentVArray
		optControls = currentPolicyArrayList
	time2 = time.time()
	nIters = localvars[0]
	print("total time: %f, avg time: %f iter code: %s" % (time2-time1, (time2-time1)/(1+nIters), bellman.iterResultString(iterCode)))
	
	if (plotResult):
		plotV(-1, 1)
		plotOptD(-1, 1)
		plotFracIn(-1, 1)	
	return result

def plotV(nIter, iP):
	setupData = setupGetNextState(nIter)
	colorFn = lambda x: getColor(setupData, x[0], x[1], g.Grid_P[iP])
	currentVArray = g.IterList[nIter]['V']
	fnObj_V = linterp.GetLinterpFnObj([g.Grid_M, g.Grid_S, g.Grid_P], currentVArray)
	fnObj_V2 = lambda x: fnObj_V([x[0], x[1], g.Grid_P[iP]])
	plot3d.plotSurface(g.Grid_M, g.Grid_S, fnObj_V2, xlabel="M", ylabel="L", zlabel="V", colorFn=colorFn)	

def plotOptD(nIter, iP):
	setupData = setupGetNextState(nIter)
	colorFn = lambda x: getColor(setupData, x[0], x[1], g.Grid_P[iP])
	optControl_d = g.IterList[nIter]['d']
	fnObj_optD = linterp.GetLinterpFnObj([g.Grid_M, g.Grid_S, g.Grid_P], optControl_d)
	fnObj_optD2 = lambda x: fnObj_optD([x[0], x[1], g.Grid_P[iP]])
	plot3d.plotSurface(g.Grid_M, g.Grid_S, fnObj_optD2, xlabel="M", ylabel="L", zlabel="opt d", colorFn=colorFn)	

def plotFracIn(nIter, iP):
	setupData = setupGetNextState(nIter)
	colorFn = lambda x: getColor(setupData, x[0], x[1], g.Grid_P[iP])
	optControl_inFrac = g.IterList[nIter]['fracIn']		
	fnObj_optInFrac = linterp.GetLinterpFnObj([g.Grid_M, g.Grid_S, g.Grid_P], optControl_inFrac)
	fnObj_optInFrac2 = lambda x: fnObj_optInFrac([x[0], x[1], g.Grid_P[iP]])
	plot3d.plotSurface(g.Grid_M, g.Grid_S, fnObj_optInFrac2, xlabel="M", ylabel="L", zlabel="opt frac in", colorFn=colorFn)	

# outcome can be 0 (low) or 1 (high)
# returns (M, S)
def setupGetNextState(n, currentVArray=None, optControl_d=None, optControl_inFrac=None):
	if (n != None):
		currentVArray = g.IterList[n]['V']
		optControl_d = g.IterList[n]['d']
		optControl_inFrac = g.IterList[n]['fracIn']
	fnObj_optD = linterp.GetLinterpFnObj([g.Grid_M, g.Grid_S, g.Grid_P], optControl_d)
	fnObj_optInFrac = linterp.GetLinterpFnObj([g.Grid_M, g.Grid_S, g.Grid_P], optControl_inFrac)		
	params = BankParams(**g.ParamSettings)
	params.setPrevIteration([g.Grid_M, g.Grid_S, g.Grid_P], currentVArray)
	return (fnObj_optD, fnObj_optInFrac, params)
def getNextStateVar(setupData, M, S, P, outcome):
	(fnObj_optD, fnObj_optInFrac, params) = setupData
	d = fnObj_optD([M, S, P])
	inFrac = fnObj_optInFrac([M, S, P])
	params.setStateVars([M, S, P])	
	(ev, nextMList, nextSList, nextPList) = params.calc_EV(d, inFrac)
	return ((nextMList[outcome], nextSList[outcome], nextPList[outcome]), (d, inFrac))

# coloring states:
def getColor(setupData, M, S, P):
	(fnObj_optD, fnObj_optInFrac, params) = setupData
	d = fnObj_optD([M, S, P])
	inFrac = fnObj_optInFrac([M, S, P])
	params.setStateVars([M, S, P])	
	(ev, nextMList, nextSList, nextPList) = params.calc_EV(d, inFrac)
	state = None
	# bankrupt next period
	if (nextMList[0] <= 0.0 and nextMList[1] <= 0.0):
		state = g.STATE_BANKRUPT
	elif (nextMList[0] > 0.0 and nextMList[1] <= 0.0):
		state = g.STATE_RISKY
	else:
		assert(nextMList[0] > 0.0 and nextMList[1] > 0.0)
		state = g.STATE_SAFE
	return g.StateToColor[state]

def plotNextM(n, outcome):
	setupData = setupGetNextState(n)
	colorFn = lambda x: getColor(setupData, x[0], x[1], x[2])
	fnObj_nextM = lambda x: getNextStateVar(setupData, x[0], x[1], x[2], outcome)[0][0]
	#plot3d.plotSurface(g.Grid_M, g.Grid_S, fnObj_nextM, xlabel="M", ylabel="S", zlabel="next M | outcome %d" % outcome, colorFn=colorFn)	
def plotNextS(n, outcome):
	setupData = setupGetNextState(n)
	colorFn = lambda x: getColor(setupData, x[0], x[1], x[2])
	fnObj_nextS = lambda x: getNextStateVar(setupData, x[0], x[1], x[2], outcome)[0][1]
	#plot3d.plotSurface(g.Grid_M, g.Grid_S, fnObj_nextS, xlabel="M", ylabel="S", zlabel="next S | outcome %d" % outcome, colorFn=colorFn)	
	
def plotObjectiveFn(params, nIter, state1, state2, state3):
	prevIterVArray = g.IterList[nIter][0]
	params.setPrevIteration([g.Grid_M, g.Grid_S, g.Grid_P], prevIterVArray)
	params.setStateVars([state1, state2, state3])
	controlGrids = params.getControlGridList([state1, state2, state3])
	objFn = lambda x: params.objectiveFunction([x[0], x[1], x[2]])
	#plot3d.plotSurface(controlGrids[0], controlGrids[1], objFn, xlabel="d", ylabel="frac in", zlabel="objFn")	

# def createGraph(n):	
	# setupData = setupGetNextState(n)
	# def fnObj_nextM(x, shock_i):
		# (next_x, controls) = getNextStateVar(setupData, x[0], x[1], shock_i)
		# return (next_x[0], controls)	
	# def fnObj_nextS(x, shock_i): 
		# (next_x, controls) = getNextStateVar(setupData, x[0], x[1], shock_i)
		# return (next_x[1], controls)	
	# gridList = [g.Grid_M, g.Grid_S, g.Grid_P]
	# coffinState = g.STATE_BANKRUPT
	# shockList = range(len(g.ParamSettings['probSpace']));		
	# def nextStateFn(x, shock_i):
		# return getNextStateVar(setupData, x[0], x[1], shock_i)
	# def isBankrupt(x):
		# (M, S) = (x[0], x[1])
		# return (M <= 0.0)
	# (isAbsorbedFn, nextIListFn) = markovChain.policyFn_to_transitionFns(gridList, nextStateFn, isBankrupt)
	# G = markovChain.createGraphFromPolicyFn(gridList, coffinState, shockList, isAbsorbedFn, nextIListFn)
	# return G

# plot locus of nodes that are destinations, conditional on shock
# def plotLocus(G, shock):
	# def isSuccessor(G, node):
		# if (node == g.STATE_BANKRUPT):
			# return False		
		# predecessors = G.predecessors(node)
		# if (len(predecessors) == 0):
			# return False
		# #print(node, predecessors)
		# for p in predecessors:
			# s = G.edge[p][node]['shock']
			# if (s == shock):
				# return True
		# return False
	# markovChain.plotGraphNodes2D(G, isSuccessor, title="locus | %d" % shock, xlabel="M", ylabel="S")

# def plotPrunedNodes(G):
	# S = markovChain.pruneZeroIndegree(G)
	# def in_S(G, node):
		# return (node in S)
	# markovChain.plotGraphNodes2D(G, in_S, title="after pruning zero indegree", xlabel="M", ylabel="S")
		
class MarkovChain:
	def __init__(self, initialM, initialS, initialP, VArray, optDArray, optInFracArray):
		(self.initialM, self.initialS, self.initialP) = (initialM, initialS, initialP)
		(self.currentM, self.currentS, self.currentP) = (initialM, initialS, initialP)
		self.VArray = VArray
		self.optDArray = optDArray
		self.optInFracArray = optInFracArray
		self.setupData = setupGetNextState(n=None, currentVArray=VArray, optControl_d=optDArray, optControl_inFrac=optInFracArray)
		self.fnObj_V = linterp.GetLinterpFnObj([g.Grid_M, g.Grid_S, g.Grid_P], VArray)
		#self.fnObj_nextM = lambda x, shock_i: getNextStateVar(self.setupData, x[0], x[1], shock_i)[0]
		#self.fnObj_nextS = lambda x, shock_i: getNextStateVar(self.setupData, x[0], x[1], shock_i)[1]
	
	def applyShock(self, shock_i):		
		((nextM, nextS, nextP), (d, inFrac)) =  getNextStateVar(self.setupData, self.currentM, self.currentS, self.currentP, shock_i)
		(self.currentM, self.currentS, self.currentP) = (nextM, nextS, nextP)
		return ((nextM, nextS, nextP), (d, inFrac))

	def reset(self):
		(self.currentM, self.currentS, self.currentP) = (self.initialM, self.initialS, self.initialP)
	
# probSpace is an array that sums to 1.0
# x is a random number drawn from Uniform[0,1]
# return the index of probSpace corresponding to x
def drawFromProbSpace(probSpace, x):
	c = scipy.cumsum(probSpace)
	return scipy.sum(c < x)	

M_ERROR_MAX = 0.0001;		# if M is within this much of 0, truncate to 0

def simulate(n, initialM, initialS, initialP, nSteps=100, bPrint=True, markovChainObj=None):
	def myprint(*args, **kwargs):
		if (bPrint): print(*args, **kwargs)		
	(currentM, currentS, currentP) = (initialM, initialS, initialP)	
	# simulate directly using policies and transition equations
	if (markovChainObj == None):
		currentVArray = g.IterList[n]['V']		
		optControl_d = g.IterList[n]['d']
		optControl_inFrac = g.IterList[n]['fracIn']		
		markovChainObj = MarkovChain(initialM, initialS, initialP, currentVArray, optControl_d, optControl_inFrac)
	# shocks
	rvs = scipy.stats.uniform.rvs(size=nSteps)
	[beta, grid_M, grid_S, grid_P, rFast, rSlow, popGrowth, slowOut, fastOut, fastIn, probSpace] = [g.ParamSettings[x] for x in 
	  ['beta', 'grid_M', 'grid_S', 'grid_P', 'rFast', 'rSlow', 'popGrowth', 'slowOut', 'fastOut', 'fastIn', 'probSpace']]
	shockList = [drawFromProbSpace(probSpace, rv) for rv in rvs]
	# initialize stats list
	stats = defaultdict(list)
	stats['M_outside']=[False]
	stats['S_outside']=[False]
	stats['P_outside']=[False]
	
	for (i, shock) in enumerate(shockList):
		myprint("initial: M=%.4f, S=%.4f, P=%.4f" % (currentM, currentS, currentP), end="")		
		# apply shock to initial state		
		myprint(" -> shock: %d, " % shock, end="")
		((nextM, nextS, nextP), (d, slow_in_frac)) = markovChainObj.applyShock(shock)
		myprint("controls: d=%.4f, frac=%.4f" % (d, slow_in_frac), end="")

		# record statistics
		stats['M'].append(currentM)
		stats['S'].append(currentS)
		stats['P'].append(currentP)
		stats['shock'].append(shock)
		stats['d'].append(d)
		stats['in_frac'].append(slow_in_frac)
		
		# calculate next period's state variables
		(fast_in_frac, fast_out_frac, slow_out_frac, pop_growth) = (fastIn[shock], fastOut[shock], slowOut[shock], popGrowth)
		(M, S, P) = (currentM, currentS, currentP)
		fast_growth = (1.0 + fast_in_frac*P - fast_out_frac) * (1.0 + rFast);
		nextM_2 = (M - d + slow_out_frac*S - slow_in_frac*S - fast_out_frac + fast_in_frac*P)/fast_growth;
		nextS_2 = (1.0 + slow_in_frac - slow_out_frac) * S * (1.0 + rSlow) / fast_growth;	
		nextP_2 = (pop_growth * P) / fast_growth;
		
		myprint("  fast_growth = (1.0 + %f*%f - %f) * (%0.3f) = %0.4f" % (fast_in_frac, P, fast_out_frac, 1.0+rFast, fast_growth))
		myprint("  nextM = %f/%f = %f" % ((M - d + slow_out_frac*S - slow_in_frac*S - fast_out_frac + fast_in_frac*P), fast_growth, nextM_2))
		myprint("  nextS = %0.4f/%0.4f = %0.4f" % ((1.0 + slow_in_frac - slow_out_frac) * S * (1.0 + rSlow), fast_growth, nextS_2))
		myprint("  nextP = %0.4f/%0.4f = %0.4f" % (pop_growth*P, fast_growth, nextS_2))
		myprint("  next: M=%0.4f, S=%0.4f, P=%0.4f" % (nextM, nextS, nextP))		
		# check if absorbed or outside grid
		if (nextM <= 0.0):
			myprint("  -> absorbed")
			break
		(M_outside, S_outside, P_outside) = (False, False, False)
		if (nextM > g.Grid_M[-1]):
			myprint("  -> M outside grid, set to %0.4f" % g.Grid_M[-1])
			nextM = g.Grid_M[-1]
			M_outside = True
		if (nextS > g.Grid_S[-1]):
			myprint("  -> S outside grid, set to %0.4f" % g.Grid_S[-1])
			nextS = g.Grid_S[-1]
			S_outside = True
		if (nextP > g.Grid_P[-1]):
			myprint("  -> P outside grid, set to %0.4f" % g.Grid_P[-1])
			nextP = g.Grid_P[-1]
			P_outside = True			
		# record stats that are conditional on bankruptcy
		stats['M_outside'].append(M_outside)
		stats['S_outside'].append(S_outside)
		stats['P_outside'].append(P_outside)
		# update state
		(currentM, currentS, currentP) = (nextM, nextS, nextP)
	# print statistics
	stats['nSteps'] = i+1
	stats['V'] = markovChainObj.fnObj_V([initialM, initialS, initialP])
	myprint ("number of steps: %d" % i)
	myprint ("average M: %0.4f" % scipy.mean(stats['M']))
	myprint ("average S: %0.4f" % scipy.mean(stats['S']))
	myprint ("average P: %0.4f" % scipy.mean(stats['P']))
	myprint ("average shock: %0.4f" % scipy.mean(stats['shock']))
	myprint ("M outside grid: %d/%d = %f" % (scipy.sum(stats['M_outside']), i, scipy.mean(stats['M_outside'])))
	myprint ("S outside grid: %d/%d = %f" % (scipy.sum(stats['S_outside']), i, scipy.mean(stats['S_outside'])))
	myprint ("P outside grid: %d/%d = %f" % (scipy.sum(stats['P_outside']), i, scipy.mean(stats['P_outside'])))
	myprint ("average d: %0.4f" % scipy.mean(stats['d']))
	myprint ("average in_frac: %0.4f" % scipy.mean(stats['in_frac']))
	myprint ("V: %0.4f" % scipy.mean(stats['V']))
	return (i, stats)

def plot_one_sim(initialM, initialS, initialP, nSteps=100):
	(i, stats) = simulate(-1, initialM, initialS, initialP, nSteps=nSteps, bPrint=False)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	M_series = stats['M']
	S_series = stats['S']
	in_frac_series=stats['in_frac']
	lines = ax.plot(range(len(S_series)), S_series, range(len(M_series)), M_series, 'r--', range(len(in_frac_series)), in_frac_series, 'g-.')
	ax.set_xlabel("time")
	ax.legend(lines, (r'$\bar{L}$', r'$\bar{M}$', r'$\gamma^L_t$'))
	
g_outcome_to_label = {'avg_M': r'mean $m$', 'avg_S': r'mean $l$', 'V': 'V', 'nSteps': 'mean lifetime', 'avg_in_frac': r'mean $\gamma^L_t$'}

def simulate_multiple(initialM, initialS, initialP, nSteps=1000, nRuns=1000):
	t1 = time.time()
	nIter = -1
	runStats = defaultdict(list)
	currentVArray = g.IterList[nIter]['V']		
	optControl_d = g.IterList[nIter]['d']
	optControl_inFrac = g.IterList[nIter]['fracIn']		
	markovChainObj = MarkovChain(initialM, initialS, initialP, currentVArray, optControl_d, optControl_inFrac)	
	for i in range(nRuns):
		markovChainObj.reset()
		(n, stats) = simulate(nIter, initialM, initialS, initialP, nSteps=nSteps, bPrint=False, markovChainObj=markovChainObj)
		# mean
		avg_M = scipy.mean(stats['M'])
		avg_S = scipy.mean(stats['S'])
		avg_P = scipy.mean(stats['P'])
		avg_shock = scipy.mean(stats['shock'])
		avg_M_outside = scipy.mean(stats['M_outside'])
		avg_S_outside = scipy.mean(stats['S_outside'])
		avg_P_outside = scipy.mean(stats['P_outside'])
		avg_d = scipy.mean(stats['d'])
		avg_in_frac = scipy.mean(stats['in_frac'])		
		V = scipy.mean(stats['V'])
		
		runStats['nSteps'].append(n)
		runStats['V'].append(V)
		runStats['avg_M'].append(avg_M)
		runStats['avg_S'].append(avg_S)
		runStats['avg_P'].append(avg_P)
		runStats['avg_shock'].append(avg_shock)
		runStats['avg_M_outside'].append(avg_M_outside)
		runStats['avg_S_outside'].append(avg_S_outside)	
		runStats['avg_P_outside'].append(avg_P_outside)	
		runStats['avg_d'].append(avg_d)
		runStats['avg_in_frac'].append(avg_in_frac)		
		
	print("runs=%d, steps=%d" % (nRuns, nSteps))
	for key in ['nSteps', 'avg_M', 'avg_S', 'avg_P', 'avg_shock', 'avg_M_outside', 'avg_S_outside', 'avg_P_outside', 'avg_d', 'avg_in_frac', 'V']:	            
		print("mean %s: %f (%f)" % (key, scipy.mean(runStats[key]), scipy.std(runStats[key])))
	t2 = time.time()
	print("total time for %d runs: %f avg: %f" % (nRuns, t2-t1, (t2-t1)/nRuns))
	return runStats

def simulateTestCase(arg):
	(test, newParams, outPath, initialM, initialS, initialP, nSteps, nRuns, xName, yName, x, y, recreate) = arg
	(testName, overrideParams) = test
	if (not os.path.exists(outPath)):
		print ("not found: %s" % outPath)
		return (None, None, None)
	try:
		loadRun(outPath)
		print("%s=%f, %s=%f" % (xName, x, yName, y))
		simFilename = outPath + ".sim"
		if (recreate or not os.path.exists(simFilename)):
			runStats = simulate_multiple(initialM, initialS, initialP, nSteps=nSteps, nRuns=nRuns)
			f = gzip.open(simFilename, 'wb')
			pickle.dump((x, y, runStats), f)
			f.close()
		else:
			f = gzip.open(simFilename, 'rb')
			(x, y, runStats) = pickle.load(f)
			f.close()
	except AssertionError as err:
		print("exception on %s: " % outPath, err)
	return (x, y, runStats)
	
def simulate_summary(dirname, testGenObj, recreate=True, initialM=1, initialS=1, initialP=1, nSteps=1000, nRuns=1000, multiprocess=True, nProcesses=6):	
	(xList, yList) = ([], [])	
	outcomes = defaultdict(list)
	args = []
	for (testName, overrideParams) in testGenObj.getOverrideParamsList():			
		newParams = dict(testGenObj.getDefaultParamsDict())
		newParams.update(overrideParams)
		prefix = testGenObj.getFilenamePrefix(testName, newParams)
		outPath = os.path.join(dirname, prefix) + ".out"
		(xName, yName) = testGenObj.getXYName()
		(x, y) = testGenObj.getXY(newParams)
		args.append(((testName, overrideParams), newParams, outPath, initialM, initialS, initialP, nSteps, nRuns, xName, yName, x, y, recreate))
	if (multiprocess):
		p = multiprocessing.Pool(nProcesses)
		# run simulations in parallel
		result = p.map(simulateTestCase, args)
		p.close()
	else:
		# run simulations
		result = map(simulateTestCase, args)
	for (x, y, runStats) in result:
		xList.append(x)
		yList.append(y)
		# store sample averages from simulations as outcomes
		for (key, value) in runStats.items():
			mean_val_across_runs = scipy.mean(value)
			outcomes[key].append(mean_val_across_runs)
		# store 
	sum_dict = {}	
	g_dict = g.to_dict()
	sum_dict['g'] = g_dict
	sum_dict['initialState'] = (initialM, initialS)
	sum_dict['xName'] = xName
	sum_dict['yName'] = yName
	sum_dict['xList'] = xList
	sum_dict['yList'] = yList
	sum_dict['outcomes'] = outcomes
	filename = "sim_M_%0.2f_S_%0.2f_summary.out" % (initialM, initialS)
	path = os.path.join(dirname, filename)
	f = gzip.open(path, 'wb')
	pickle.dump(sum_dict, f)
	f.close()
		
	return (filename, xName, yName, xList, yList, outcomes)

import fnmatch	
def convert_to_gzip():
	files = os.listdir(".")
	for file in files:
		if fnmatch.fnmatch(file, '*.sim'):
			print(file)
			f = open(file, 'r')
			(x, y, runStats) = pickle.load(f)			
			f.close()		
			f = gzip.open(file, 'wb')
			pickle.dump((x, y, runStats), f)			
			f.close()

def test_sim1(dirname, initialM=1, initialS=1, recreate=False, beta=0.9, multiprocess=True):
	x = simulate_summary(dirname, "rSlow", "rFast", initialM=initialM, initialS=initialS, recreate=recreate, generateTestListFn=generateTestList_test_r, generateFnArgs={'beta':beta}, multiprocess=multiprocess)

def test_sim2(dirname, initialM=1, initialS=1, recreate=False, beta=0.9, multiprocess=True, nProcesses=2):
	x = simulate_summary(dirname, initialM=initialM, initialS=initialS, recreate=recreate, multiprocess=multiprocess, nProcesses=nProcesses, **test_var_args())

def test_delta_sim(dirname, **kwargs):
	x = simulate_summary(dirname, g_TestDelta, initialM=1.1, initialS=1.1, **kwargs)

def test_delta_sim2(dirname, **kwargs):
	x = simulate_summary(dirname, g_TestDelta2, initialM=1.1, initialS=1.1, **kwargs)
	
def plot_simulation_summary(dirname, testGenObj, filename="sim_M_1.00_S_1.00_summary.out"):
	(xlabel, ylabel) = testGenObj.getXYGraphLabel()
	simFilename = os.path.join(dirname, filename)
	f = gzip.open(simFilename, 'rb')
	sim_dict = pickle.load(f)
	f.close()
	[sim_xName, sim_yName, sim_xList, sim_yList, sim_outcomes, sim_initialState] = [sim_dict[x] for x in ['xName', 'yName', 'xList', 'yList', 'outcomes', 'initialState']]

	# load summary of solutions
	solutionFilename = os.path.join(dirname, "solutions.out")
	f = gzip.open(solutionFilename, 'rb')
	solution_dict = pickle.load(f)
	f.close()
	[sol_xName, sol_yName, sol_xList, sol_yList, sol_outcomeList] = [solution_dict[x] for x in ['xName', 'yName', 'xList', 'yList', 'outcomeList']]	
	# keep track of points where iteration did not converge
	nonconverge_points = {}
	for (x, y, outcome) in zip(sol_xList, sol_yList, sol_outcomeList):
		if (outcome == SOLUTION_CATEGORY_NONCONVERGENCE):	# did not converge
			nonconverge_points[(x,y)] = 1
	sol_colors = [outcome_to_color(o) for o in sol_outcomeList]
	sol_markers_labels = [outcome_to_marker(o) for o in sol_outcomeList]
	# return (xList, yList, valList) with nonconvergence points removed
	def filter_nonconverge(*args):			
		result = [item for item in zip(*args) if (not (item[0],item[1]) in nonconverge_points)]
		if (len(result) > 0):
			return zip(*result)
		else:
			return [] * len(args)
		
	page = markup.page()
	page.init(title="simulation summary, initial state=%f,%f" % sim_initialState)
	page.p("initial state=%f,%f" % sim_initialState)
	page.br( )		

	# text summary of simulations
	txtList = []	
	for i in range(len(sim_xList)):
		(x, y) = (sim_xList[i], sim_yList[i])
		txtList.append("%s: %f, %s: %f" % (sim_xName, x, sim_yName, y))
		for (name, sim_valList) in sim_outcomes.items():
			txtList.append("  %s: %f" % (name, sim_valList[i]))
		txtList.append("")
	(prefix, ext) = os.path.splitext(filename)
	filename = os.path.join(dirname, prefix + ".txt")		
	f = open(filename, 'w')                             
	[f.write(line + '\n') for line in txtList]
	f.close()
	
	writeShowImgScript(page)
	plt.ioff()
	for (name, sim_valList) in sim_outcomes.items():		
		page.p(name)		
		(xList, yList, zList, colorList, markers_labels) = (sim_xList, sim_yList, sim_valList, sol_colors, sol_markers_labels)
		if (name == 'V'):			
			(xList, yList, zList, colorList, markers_labels) = filter_nonconverge(sim_xList, sim_yList, sim_valList, sol_colors, sol_markers_labels)
		# for each col and row, a 2d graph		
		xValues = list(sorted(set(xList)))
		yValues = list(reversed(sorted(set(yList))))
		xy_to_z = {}
		xy_to_i = {}
		xy_to_color = {}
		for (i, (x, y, z, color)) in enumerate(zip(xList, yList, zList, colorList)):
			xy_to_z[(x,y)] = z
			xy_to_i[(x,y)] = i
			xy_to_color[(x,y)] = color

		# 3d graph, one line plot per row
		fig = plt.figure()
		ax = Axes3D(fig)
		if (xlabel == None): xlabel = sim_xName
		if (ylabel == None): ylabel = sim_yName
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)		
		if (name in g_outcome_to_label):
			ax.set_zlabel(g_outcome_to_label[name])
		else:
			ax.set_zlabel(name)		
		markerStyles = itertools.cycle(['o', '^', 's', 'x'])
#		c2 = scipy.array([0.25, 0.25, 0.25]);		# gray
#		c1 = scipy.array([0.9, 0.0, 0.0])		
		c2 = scipy.array([0.1, 0.1, 0.1]);		# gray
		c1 = scipy.array([0.7, 0.7, 0.7])				
		colors = itertools.cycle([c1 + (c2-c1)*f for f in scipy.linspace(0, 1, len(yValues))])
		for y in yValues:
			xs = [x for x in xValues if ((x,y) in xy_to_z)]
			zs = [xy_to_z[(x,y)] for x in xValues if ((x,y) in xy_to_z)]
			ys = [y] * len(xs)
			label = testGenObj.getXYShortLabel()[1] + " = %0.3f" % y
			color=colors.next()
			marker = markerStyles.next()
			ax.plot(xs, ys, zs, linestyle='-', marker=marker, label=label, color=color, markeredgecolor=scipy.array([0.25, 0.25, 0.25]), markersize=5)
		ax.legend(loc='upper left', prop={"size":11})
		imgFilename = "%s_%s_lines_3d.png" % (prefix, name)
		imgPath = os.path.join(dirname, imgFilename)
		try:                                        
			os.remove(imgPath)
		except OSError:        
			pass   
		plt.savefig(imgPath, format='png', dpi=120)	
		plt.close(fig)
		page.img(src=imgFilename)             	
			
		# 3d plot
		fig = plt.figure()
		ax = Axes3D(fig)
		for (key, val) in groupby(zip(xList, yList, zList, colorList, markers_labels), lambda x: x[4]).items():
			(marker, label) = key
			(x_list, y_list, z_list, color_list, m_l_list) = zip(*val)
			ax.scatter(x_list, y_list, z_list, color=color_list, marker=marker, label=label, facecolor='none')
		if (xlabel == None): xlabel = sim_xName
		if (ylabel == None): ylabel = sim_yName
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_zlabel(name)
		imgFilename = "%s_%s_sim.png" % (prefix, name)
		imgPath = os.path.join(dirname, imgFilename)
		try:                                        
			os.remove(imgPath)
		except OSError:        
			pass   
		plt.savefig(imgPath, format='png', dpi=120)					
		page.img(src=imgFilename)             
		plt.close(fig)
				
		xImgDict = {}
		for x in xValues:		
			fig = plt.figure()
			ax = fig.add_subplot(111)	
			ax.set_title("%s = %f" % (xlabel, x))
			ax.set_xlabel(ylabel)
			ax.set_ylabel(name)			
			ax.scatter([y for y in yValues if ((x,y) in xy_to_z)],
			  [xy_to_z[(x,y)] for y in yValues if ((x,y) in xy_to_z)], 
			  color=[xy_to_color[(x,y)] for y in yValues if((x,y) in xy_to_z)])
			imgFilename = "%s_%s_x_%.3f_sim.png" % (prefix, name, x)
			imgPath = os.path.join(dirname, imgFilename)
			# try:                                        
				# os.remove(imgPath)
			# except OSError:        
				# pass   
			# plt.savefig(imgPath, format='png', dpi=120)								
			xImgDict[x] = imgFilename
			plt.close(fig)

		yImgDict = {}	
		for y in yValues:		
			fig = plt.figure()
			ax = fig.add_subplot(111)	
			ax.set_title("%s = %f" % (ylabel, y))
			ax.set_xlabel(xlabel)
			ax.set_ylabel(name)		
			#ax.scatter(xValues, [xy_to_z[(x,y)] for x in xValues], color=[xy_to_color[(x,y)] for x in xValues])
			ax.scatter([x for x in xValues if ((x,y) in xy_to_z)],
			  [xy_to_z[(x,y)] for x in xValues if ((x,y) in xy_to_z)], 
			  color=[xy_to_color[(x,y)] for x in xValues if((x,y) in xy_to_z)])			
			imgFilename = "%s_%s_y_%.3f_sim.png" % (prefix, name, y)
			imgPath = os.path.join(dirname, imgFilename)
			# try:                                        
				# os.remove(imgPath)
			# except OSError:        
				# pass   
			# plt.savefig(imgPath, format='png', dpi=120)							
			yImgDict[y] = imgFilename			
			plt.close(fig)

		# 2d graph, one line per row
		fig = plt.figure()
		ax = fig.add_subplot(111)	
		# ax.set_title("%s = %f" % (ylabel, y))
		ax.set_xlabel(xlabel)
		if (name in g_outcome_to_label):
			ax.set_ylabel(g_outcome_to_label[name])
		else:
			ax.set_ylabel(name)
		markerStyles = itertools.cycle(['o', '^', 's', 'x'])
		c2 = scipy.array([0.1, 0.1, 0.1]);		# gray
		c1 = scipy.array([0.7, 0.7, 0.7])				
		colors = itertools.cycle([c1 + (c2-c1)*f for f in scipy.linspace(0, 1, len(yValues))])
		for y in yValues:
			xs = [x for x in xValues if ((x,y) in xy_to_z)]
			zs = [xy_to_z[(x,y)] for x in xValues if ((x,y) in xy_to_z)]
			label = testGenObj.getXYShortLabel()[1] + " = %0.3f" % y
			color=colors.next()
			marker = markerStyles.next()
			ax.plot(xs, zs, linestyle='-', marker=marker, label=label, color=color, markeredgecolor=scipy.array([0.25, 0.25, 0.25]), markersize=5)
		ax.legend(loc='upper left', prop={"size":11})
		imgFilename = "%s_%s_lines.png" % (prefix, name)
		imgPath = os.path.join(dirname, imgFilename)
		# try:                                        
			# os.remove(imgPath)
		# except OSError:        
			# pass   
		plt.savefig(imgPath, format='png', dpi=120, bbox_inches='tight', pad_inches=0.25)	
		plt.close(fig)
		page.img(src=imgFilename)             
						
		# table
		writeHTMLTable(name, page, xList, yList, zList, xImgDict, yImgDict)
		# latex table
		writeLatexTable(name, page, xList, yList, zList)
		
	plt.ion()	
	# write plot to html
	filename = os.path.join(dirname, prefix + ".html")
	f = open(filename, 'w')                             
	f.write(str(page))
	f.close()

def plotSimulationSummary_1():
	plotSimulationSummary("test_r", "sim_M_1.00_S_1.00_summary", xlabel="loan interest rate", ylabel="deposit interest rate")	
	
def plotSimulationSummary_2():
	#origGrid = g.getGridSize()
	#g.setGridSize(M=(2, 80))
	plotSimulationSummary("test_var", "sim_M_1.00_S_1.00_summary", xlabel=r'low realization of $\alpha^D_t$', ylabel=r'probability of high $\alpha^D_t$')	
	#g.setGridSize(**origGrid)

def plotSimulationSummary_3():
	plotSimulationSummary("test_delta", "sim_M_1.10_S_1.10_summary", xlabel='delta', ylabel="interest on loans")
		
	
def plot2DSummary_1():
	plot_test_summary_2d("test_r", xlabel="loan interest rate", ylabel="deposit interest rate", addLines1=True)

def plot2DSummary_2():
	plot_test_summary_2d("test_var", xlabel=r'probability of high $\alpha^D_t$', ylabel=r'low realization of $\alpha^D_t$')
	
# write a 2D table to a markup page object.
# if outputFn is None, assume z values are floats and print them.
# otherwise, callback outputFn with (page, x, y, z, i) as args. i is a count that starts from 0.
def writeHTMLTable(name, page, xList, yList, zList, xImgDict, yImgDict, outputFn=None):
	# go from xList, yList to table.
	xValues = sorted(list(set(xList)))
	yValues = reversed(sorted(list(set(yList))))
	xy_to_z = {}
	xy_to_i = {}
	for (i, (x, y, z)) in enumerate(zip(xList, yList, zList)):
		xy_to_z[(x,y)] = z
		xy_to_i[(x,y)] = i	
	page.table.open()
	# header row
	page.tr.open()
	page.th("")
	for x in xValues:
		page.th.open()
		page.a("%0.4f" % x, href="javascript:setDisplayedGraph(\"%s\", \"%s\")" % (name, xImgDict[x]))
		page.th.close()
	page.tr.close()	
	for y in yValues:
		page.tr.open()
		# header col
		page.th.open()
		page.a("%0.4f" % y, href="javascript:setDisplayedGraph(\"%s\", \"%s\")" % (name, yImgDict[y]))
		page.th.close()
		for x in xValues:
			if ((x,y) in xy_to_z):
				z = xy_to_z[(x,y)]			
				if (outputFn != None):
					page.td.open()
					outputFn(x, y, z, i)
					page.td.close()
				else:
					page.td("%0.4f" % z)
			else:
				page.td()
		page.tr.close()
	page.table.close()
	page.img(id='graph2d_%s' % name)

def writeShowImgScript(page):
	fnTxt = "function setDisplayedGraph (name,filename) {d=document.getElementById(\"graph2d_\" + name); d.setAttribute(\"src\",filename) }"
	page.script(fnTxt, type="text/javascript")		
	
def writeLatexTable(name, page, xList, yList, zList):
	xValues = sorted(list(set(xList)))
	yValues = reversed(sorted(list(set(yList))))
	xy_to_z = {}	
	for (i, (x, y, z)) in enumerate(zip(xList, yList, zList)):
		xy_to_z[(x,y)] = z
	data = []
	for y in yValues:
		data.append([])
		for x in xValues:
			if ((x,y) in xy_to_z):
				data[-1].append("%0.3f" % xy_to_z[(x,y)])
			else:
				data[-1].append("")
	myheaders = ["%0.2f" % x for x in xValues]
	mystubs = ["%0.2f" % y for y in yValues]
	tbl = table.SimpleTable(data, myheaders, mystubs)
	page.tt(tbl.as_latex_tabular())				

############################################################################
# generate the graphs in the paper
output_path = "../../banks/doc1/bank_paper"

# show M, L over time of numerical example 1 
def paper_graph_1():
	loadRun("bp_example1.out")
	(i, stats) = simulate(-1, 1, 1, 1, nSteps=100, bPrint=False)
	plt.ioff()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#ax = fig.add_axes([0.1, 0.3, 0.4, 0.4])
	lines = ax.plot(range(len(stats['S'])), stats['S'], range(len(stats['M'])), stats['M'], 'r--')
	ax.set_xlabel("time")
	ax.legend(lines, (r'$l_t$ (assets/liabilities)', r'$m_t$ (cash/liabilities)'), loc='right')
	#fig.legend(lines, (r'$l$ (assets/liabilities)', r'$m$ (cash/liabilities)'), loc=(0.75, 0.5), prop={"size":11})
	plt.savefig("bp_example1.png", format='png', dpi=120, bbox_inches='tight', pad_inches=0.25)
	plt.ion()
	
def paper_graph_2():
	loadRun("bp_example2.out")
	(i, stats) = simulate(-1, 1, 1, 1, nSteps=100, bPrint=False)
	plt.ioff()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	lines = ax.plot(range(len(stats['S'])), stats['S'], range(len(stats['M'])), stats['M'], 'r--', range(len(stats['in_frac'])), stats['in_frac'], 'g-.')
	ax.set_xlabel("time")
	ax.legend(lines, (r'$l_t$ (assets/liabilities)', r'$m_t$ (cash/liabilities)', r'$\gamma^L_t$ (lending)'), loc='right', prop={"size":11})	
	plt.savefig("bp_example2.png", format='png', dpi=120, bbox_inches='tight', pad_inches=0.25)
	plt.ion()
	
# varying interest rates: g_TestR5
# varying rD, delta^D: g_TestDelta7
# varying delta^D, delta^L: g_TestL, g_TestL2 for the big graph
# testR_nSteps_3d.png
# testR_in_frac_3d.png
# testR_avg_M_lines.png
# testR_avg_S_lines.png
# testR_V_lines.png
# testDelta_nSteps_3d.png
# testDelta_in_frac_3d.png
# testDelta_avg_M_lines.png
# testDelta_avg_S_lines.png
# testDelta_V_lines.png
# testL_nSteps_3d.png
# testL_in_frac_3d.png
# testL_avg_M_lines.png
# testL_avg_S_lines.png
# testL_V_lines.png
# testL2_avg_M_3d.png
# testL2_avg_S_3d.png

def paper_graph_3():
	plot_simulation_summary("new_test_r_5", g_TestR5)	
	plot_simulation_summary("new_test_delta_7", g_TestDelta7)
	plot_simulation_summary("new_test_L", g_TestL)
	plot_simulation_summary("new_test_L_2", g_TestL2)

def copy_paper_graph_files(destDir="C:/My Dropbox/banks/doc1/bank_paper"):	
	file_map = {
		"bp_example1.png" : "num_example1.png",
		"bp_example2.png" : "num_example2.png",
		"new_test_r_5/sim_M_1.00_S_1.00_summary_nSteps_lines_3d.png" : "testR_nSteps_3d.png",
		"new_test_r_5/sim_M_1.00_S_1.00_summary_avg_in_frac_lines_3d.png" : "testR_in_frac_3d.png",
		"new_test_r_5/sim_M_1.00_S_1.00_summary_avg_M_lines.png" : "testR_avg_M_lines.png",
		"new_test_r_5/sim_M_1.00_S_1.00_summary_avg_S_lines.png" : "testR_avg_S_lines.png",
		"new_test_r_5/sim_M_1.00_S_1.00_summary_V_lines.png" : "testR_V_lines.png",
		"new_test_delta_7/sim_M_1.00_S_1.00_summary_nSteps_lines_3d.png" : "testDelta_nSteps_3d.png",
		"new_test_delta_7/sim_M_1.00_S_1.00_summary_avg_in_frac_lines_3d.png" : "testDelta_in_frac_3d.png",
		"new_test_delta_7/sim_M_1.00_S_1.00_summary_avg_M_lines.png" : "testDelta_avg_M_lines.png",
		"new_test_delta_7/sim_M_1.00_S_1.00_summary_avg_S_lines.png" : "testDelta_avg_S_lines.png",
		"new_test_delta_7/sim_M_1.00_S_1.00_summary_V_lines.png" : "testDelta_V_lines.png",
		"new_test_L/sim_M_1.00_S_1.00_summary_nSteps_lines_3d.png" : "testL_nSteps_3d.png",
		"new_test_L/sim_M_1.00_S_1.00_summary_avg_in_frac_lines_3d.png" : "testL_in_frac_3d.png",
		"new_test_L/sim_M_1.00_S_1.00_summary_avg_M_lines.png" : "testL_avg_M_lines.png",
		"new_test_L/sim_M_1.00_S_1.00_summary_avg_S_lines.png" : "testL_avg_S_lines.png",
		"new_test_L/sim_M_1.00_S_1.00_summary_V_lines.png": "testL_V_lines.png",
		"new_test_L_2/sim_M_1.00_S_1.00_summary_nSteps_lines_3d.png" : "testL2_avg_M_3d.png",
		"new_test_L_2/sim_M_1.00_S_1.00_summary_avg_in_frac_lines_3d.png" : "testL2_avg_S_3d.png"
	}
	for (src, dest) in file_map.items():
		shutil.copy(src, os.path.join(destDir, dest))
		print(src, os.path.join(destDir, dest))
	
def do_all_graphs():
	paper_graph_1()
	paper_graph_2()
	paper_graph_3()
	copy_paper_graph_files()
	
				