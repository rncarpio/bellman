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
  

// data structures specific to the ponzi problem

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>

#include "ponziProblem.h"
#include "maximizer.h"
#include "debugMsg.h"

namespace bpl = boost::python;
using namespace boost;
using namespace python;
using namespace pyublas;
using namespace std;
  
static GlobalParams g_Params;

// utility functions
// exponential
double U_exponential(double c) {
  double theta = g_Params.m_theta;
  return 1 - exp(-theta * c);
}
double Uprime_exponential(double c) {
  double theta = g_Params.m_theta;
  return theta * exp(-theta * c);
}
// linear
double U_linear(double c) {
  return c;
}
double Uprime_linear(double c) {
  return 1;
}
// CRRA
double U_crra(double c) {
  double gamma = g_Params.m_gamma;
  return pow(c, 1-gamma) / (1-gamma);
}
double Uprime_crra(double c) {
  double gamma = g_Params.m_gamma;
  return pow(c, -gamma);
}

// f only
// depositors have linear mean-SD preferences, i.e. U(c) = E(c) - m*SD(c)
// k = d + D - M
// assume Z has 2 possible values
// r is gross, i.e. > 1
double f(double k, double r) {
  assert(g_Params.m_ZVals.size() == 2);
//  double zLow = g_Params.m_ZVals[0];
//  double zHigh = g_Params.m_ZVals[1];
//  double pHigh = g_Params.m_ZProbs[1];
  double zLow = *ARRAYPTR1D(g_Params.m_pZVals, 0);
  double zHigh = *ARRAYPTR1D(g_Params.m_pZVals, 1);
  double pHigh = *ARRAYPTR1D(g_Params.m_pZProbs, 1);  
  assert(zLow < zHigh);
  if (k > zHigh) {
    return 0.0;
  } else if (k <= zHigh && k > zLow) {
    // slope of feasible region line
	// in this case, bank survives if zHigh occurs, so probability = pHigh
    double slope = (r*pHigh - 1.0)/(r*sqrt(pHigh * (1.0-pHigh)));
	if (g_Params.m_depositorSlope > slope) {
	  return 0.0;
	} else {
	  return 1.0;
	}
  } else if (k <= zLow && k > 0.0) {
    return 1.0;
  } else {
    // k < 0
    return 1.0;
  }
}

// calculate expected value for next period
// w_array is a 3D array
// grid_M, grid_D, grid_N are 1d arrays
// M, D, N are the state values
// d, r are the controls
// z_vals, z_probs are shock distribution values
// fs is a function that takes r, returns the fraction of income invested in bank (a double)
double expected_next_v (PonziParams const *pParams,
    double M, double D, double d, double r, double bankruptcyPenalty,
    PyArrayObject const *pZVals, PyArrayObject const *pZProbs,
    ddFn2 *pFSFn, bool bPrint) {
  int zi, zlen;
  double *pZ, fs, probZ, nextM, nextD;
  double sum;

  assert(pZVals->nd == pZProbs->nd);
  assert(pZVals->dimensions[0] == pZProbs->dimensions[0]);
  zlen = pZVals->dimensions[0];
  // calculate expectation -> for each possible shock z...
  if (bPrint) {
    DebugMsg("expected_next_v M=%f D=%f d=%f r=%f\n", M,D,d,r);
  }
  sum = 0.0;
  
  for (zi=0; zi<zlen; zi++) {
    pZ = (double*) ARRAYPTR1D(pZVals, zi);
    probZ = * (double*) ARRAYPTR1D(pZProbs, zi);
    //fs = (*pFSFn)(d + D - M, r);   
	fs = f(d + D - M, r);   
    nextM = (M + fs * (*pZ) - D - d) / (*pZ);
    nextD = r * fs;
    if (nextM <= 0.0) {
      //sum += 0.0;	  
	  // bankruptcyPenalty should be a positive number.
	  assert(bankruptcyPenalty >= 0.0);
	  double incr = probZ * (*pZ) * bankruptcyPenalty * (-nextM);				// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
	  //double incr = probZ * bankruptcyPenalty * (-nextM);				// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
	  sum += incr;
	  if (bPrint) {
	    DebugMsg("  probZ=%f z=%f nextM=%f nextD=%f: +%f\n", probZ, *pZ, nextM, nextD, incr);
	  }
    } else {
	  //double interpVal = interp2d_grid(pMGrid, pDGrid, pWArray, nextM, nextD);	// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
	  double interpVal = pParams->m_pWInterp->interp(nextM, nextD);	// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
      double incr = probZ * (*pZ) * interpVal;
	  //double incr = probZ * interpVal;
	  sum += incr;
	  if (bPrint) {
	    DebugMsg("  probZ=%f z=%f nextM=%f nextD=%f: w=%f +%f\n", probZ, *pZ, nextM, nextD, interpVal, incr);
	  }
    }
  }
  return sum;
}
  
// default values. can be overridden later
void GlobalParams::initGlobalParams() {
  m_beta = 0.9;
  m_theta = 0.5;
  m_gamma = 2.0;
  m_pU = &U_linear;
  m_pFS = &f;
  m_depositorSlope = 1.0;
  m_bankruptcyPenalty = 0.0;
};

/*
*  PonziParams methods
*/

// the function to be maximized.  this will be repeatedly called for each value of the control grid in controlVars.
// state variables (M, D) will be members of PonziParams object
double PonziParams::objectiveFunction(DoubleVector const &controlVars) const {
//double PonziParams::objectiveFunction2(double d, double r) const {
  double d = controlVars[0];
  double r = controlVars[1];
  double M = m_M;
  double D = m_D;  
  bool bPrint = m_bPrint;
  
  double ev = expected_next_v(this,
      M, D, d, r, g_Params.m_bankruptcyPenalty, g_Params.m_pZVals, g_Params.m_pZProbs, g_Params.m_pFS, bPrint);	  
  double result = (*g_Params.m_pU)(d) + g_Params.m_beta * ev;
  if (bPrint) {
    DebugMsg("calc_exp_util: M=%f D=%f d=%f r=%f\n", M, D, d, r);
	DebugMsg("  %f + %f * %f = %f\n", (*g_Params.m_pU)(d), g_Params.m_beta, ev, result);
  }
  return result;
}

void PonziParams::setStateVars(boost::python::list const &stateVars) {
  m_M = bpl::extract<double>(stateVars[0]);
  m_D = bpl::extract<double>(stateVars[1]);
}
/*
bpl::list PonziParams::getControlGridList(bpl::list const &stateVars) const {
  bpl::list result;
  return result;
}
*/
int PonziParams::getNControls() const {
  return 2;
}
void PonziParams::setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray) {
  m_StateGrid1 = bpl::extract<DoublePyArray>(stateGridList[0]);
  m_StateGrid2 = bpl::extract<DoublePyArray>(stateGridList[1]);  
  m_pStateGrid1 = (PyArrayObject const*) m_StateGrid1.data().handle().get();
  m_pStateGrid2 = (PyArrayObject const*) m_StateGrid2.data().handle().get(); 
  m_W = WArray;	
  m_pW = (PyArrayObject const*) m_W.data().handle().get();
  m_pWInterp.reset(new PyInterp2D(m_StateGrid1, m_StateGrid2, DoublePyMatrix(m_StateGrid1.size(), m_StateGrid2.size(), WArray)));
}	

	
static char *g_fnNames[] = {"crra", "exponential", "linear"};
static ddFn *g_u_functions[] = {&U_crra, &U_exponential, &U_linear};
ddFn *uFnNameToFnPtr(const char *pcUFnName) {  
  // set utility function  
  for (int i=0; i<CARRAYLEN(g_fnNames); i++) {
    if (strcmp(pcUFnName, g_fnNames[i]) == 0) {	  	  
	  return g_u_functions[i];
    }	
  }
  return NULL;
}

const char *uFnPtrToFnName(ddFn *pFn) {
  for (int i=0; i<CARRAYLEN(g_u_functions); i++) {
    if (pFn == g_u_functions[i]) {	  	  
	  return g_fnNames[i];
    }	
  }
  return NULL;
}

void setGlobalParams(double theta, double beta, double gamma, const char* pcUFnName, 
  DoublePyArray const &grid1, DoublePyArray const &grid2, DoublePyArray const &zVals, DoublePyArray const &zProbs,
  double depositorSlope, double bankruptcyPenalty) {

  g_Params.m_beta = beta;
  g_Params.m_theta = theta;
  g_Params.m_gamma = gamma;
    
  ddFn *pUFn = uFnNameToFnPtr(pcUFnName);
  if (pUFn == NULL) {
    char pcTemp[1024];
    sprintf_s(pcTemp, "unknown utility fn name: %s", pcUFnName);
    PyErr_SetString(PyExc_ValueError, pcTemp);
    throw_error_already_set();
  }
  g_Params.m_pU = pUFn;
  DebugMsg("setGlobalParams: setting utility to %s\n", pcUFnName);
    
  g_Params.m_StateGrid1 = grid1;
  g_Params.m_pStateGrid1 = (PyArrayObject const*) g_Params.m_StateGrid1.data().handle().get();
  g_Params.m_StateGrid2 = grid2;
  g_Params.m_pStateGrid2 = (PyArrayObject const*) g_Params.m_StateGrid2.data().handle().get();
  g_Params.m_ZVals = zVals;
  g_Params.m_pZVals = (PyArrayObject const*) g_Params.m_ZVals.data().handle().get();
  g_Params.m_ZProbs = zProbs;
  g_Params.m_pZProbs = (PyArrayObject const*) g_Params.m_ZProbs.data().handle().get();
  
  // check that zProbs sum up to 1
  double sum = std::accumulate(zProbs.begin(), zProbs.end(), 0.0);
  if (sum != 1.0) { throw std::invalid_argument("zProb doesn't sum to 1.0"); }
  
  g_Params.m_depositorSlope = depositorSlope;
  g_Params.m_bankruptcyPenalty = bankruptcyPenalty;
  
  return;
}

dict getGlobalParams() {
  dict result;
  result["beta"] = g_Params.m_beta;
  result["theta"] = g_Params.m_theta;
  result["gamma"] = g_Params.m_gamma;
  const char *pcFnName = uFnPtrToFnName(g_Params.m_pU);
  if (pcFnName != NULL) {
    result["utilityFn"] = pcFnName;
  } else {
    result["utilityFn"] = object();		// if pcFnName is NULL, return a None object
  }  
  result["stateGrid1"] = g_Params.m_StateGrid1;
  result["stateGrid2"] = g_Params.m_StateGrid2;
  result["zVals"] = g_Params.m_ZVals;
  result["zProbs"] = g_Params.m_ZProbs;
  result["depositorSlope"] = g_Params.m_depositorSlope;
  result["bankruptcyPenalty"] = g_Params.m_bankruptcyPenalty;
  return result;
}


void test1(const char *pcString) {
  printf("%s\n", pcString);
}

DoublePyArray test2(DoublePyArray const &x) {
  return 2*x;
}

BOOST_PYTHON_MODULE(_ponziProblem)
{                              
  g_Params.initGlobalParams();
  bpl::def("setGlobalParams", setGlobalParams);
  bpl::def("getGlobalParams", getGlobalParams);
  bpl::def("U_exp", U_exponential);
  bpl::def("U_linear", U_linear);
  bpl::def("U_crra", U_crra);
  bpl::def("f", f);  
  bpl::def("test1", test1);
  bpl::def("test2", test2);

//  bpl::class_<MaximizerCallParams, boost::noncopyable>("MaximizerCallParams", bpl::no_init)
//    ;
//  bpl::class_<BellmanParams, bpl::bases<MaximizerCallParams>, boost::noncopyable>("BellmanParams", bpl::no_init)
//    ;
  bpl::class_<PonziParams, bpl::bases<BellmanParams>>("PonziParams", bpl::init<>())      
    ;
  
}                                          

