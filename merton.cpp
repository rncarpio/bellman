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
  

// code for Merton problem

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/random.hpp>
#include <boost/random/lognormal_distribution.hpp>

#include "merton.h"
#include "maximizer.h"
#include "debugMsg.h"
#include "myfuncs.h"

namespace bpl = boost::python;
using namespace boost;
using namespace python;
using namespace pyublas;
using namespace std;

MertonParams::MertonParams(DoublePyArray stateGrid, double gamma, double delta, double riskfree_r, double mu, double sigma, double dt, bool bUseMonteCarlo)
: m_CDFFn(mu*dt, sqrt(dt)*sigma), m_PDFFn(mu*dt, sqrt(dt)*sigma)
{
	  m_StateGrid = stateGrid;
	  m_pStateGrid = (PyArrayObject const*) m_StateGrid.data().handle().get();
	  m_gamma = gamma;
	  assert(gamma >= 1.0);
	  if (gamma == 1.0) {	// log utility
	    m_uFn = LogUtil();
	  } else {
	    m_uFn = CRRA(gamma);
	  }
	  m_delta = delta;
	  m_riskfree_r = riskfree_r;
	  m_mu = mu;
	  m_sigma = sigma;
	  m_dt = dt;
	  m_beta = exp(-delta*dt);
	  	  
	  m_nDraws = 10000;
	  boost::mt19937 rng;
	  boost::lognormal_distribution<> lognormal(mu*dt, sqrt(dt)*sigma);
	  boost::variate_generator<boost::mt19937&, boost::lognormal_distribution<> > die(rng, lognormal);
	  m_RandomDraws.resize(m_nDraws);
	  std::generate(m_RandomDraws.begin(), m_RandomDraws.end(), die);
	  m_bUseMonteCarlo = bUseMonteCarlo;
	}

// Z is a lognormal shock
// nextW = f(Z)
// what if s=0, cf=1
double nextW_to_Z(double s, double cf, double r, double dt, double W, double nextW) {
  double Z = (nextW - (1.0-s)*(1.0-cf)*W*exp(r * dt)) / s*(1.0-cf)*W;
  return Z;
}
// inverse: Z = f_inv(nextW)
double Z_to_nextW(double s, double cf, double r, double dt, double W, double Z) {
  double nextW = (1.0-s)*(1.0-cf)*W*exp(r * dt) + s*(1.0-cf)*W*Z;
  return nextW;
}

double truncateIfLessThanZero(ddFnObj fn, double x) {
  if (x < 0.0) {
    return 0.0;
  } else {
    return fn(x);
  }
}

double MertonParams::calcEV_grid (double cf, double s, double W) const {
  ddFnObj inverseFn = boost::bind(nextW_to_Z, s, cf, m_riskfree_r, m_dt, W, _1);
  ddFnObj cdfFn = boost::bind(truncateIfLessThanZero, m_CDFFn, _1);		// boost lognormal cdf won't take arg < 0
  ddFnObj pdfFn = boost::bind(truncateIfLessThanZero, m_PDFFn, _1);
  double EV = -DBL_MAX;
  if (s*(1.0-cf)*W == 0.0) {   // zero invested in risky asset
    EV = interp1d_grid(m_pStateGrid, m_pPrevIterationArray, Z_to_nextW(s, cf, m_riskfree_r, m_dt, W, 0.0));
  } else {
    EV = calculateEV_grid(m_pStateGrid, m_pPrevIterationArray, cdfFn, pdfFn, inverseFn, m_PrevIteration[0], m_PrevIteration[m_PrevIteration.size()-1]);
  }
  return EV;
}

double calcEV_helper(double cf, double s, double r, double dt, double W, PyArrayObject const *pGrid, PyArrayObject const *pPrev, double z) {
  double nextW = Z_to_nextW(s, cf, r, dt, W, z);
  double nextV = interp1d_grid(pGrid, pPrev, nextW);
  return nextV;
}

double MertonParams::calcEV_montecarlo (double cf, double s, double W) const {
  ddFnObj nextVFn = boost::bind(calcEV_helper, cf, s, m_riskfree_r, m_dt, W, m_pStateGrid, m_pPrevIterationArray, _1);
  double EV = calculateEV_montecarlo_1d(nextVFn, m_RandomDraws);
  return EV;
}

double MertonParams::objectiveFunction(DoubleVector const &controlVars) const {
  double cf = controlVars[0];		// fraction of wealth consumed
  double s = controlVars[1];		// fraction of wealth invested in risky asset
  double W = m_wealth;				// wealth
  double EV = -DBL_MAX;
  if (m_bUseMonteCarlo == true) {  
    EV = calcEV_montecarlo(cf, s, W);
  } else {
    EV = calcEV_grid(cf, s, W);
  }
  double result = m_uFn(cf*W) + m_beta * EV;
  return result;
}

double MertonParams::u (double cf, double s) const {
  double W = m_wealth;				// wealth
  return m_uFn(cf*W);
}

double MertonParams::EV (double cf, double s) const {
  double W = m_wealth;				// wealth
  double EV = -DBL_MAX;
  if (m_bUseMonteCarlo == true) {  
    EV = calcEV_montecarlo(cf, s, W);
  } else {
    EV = calcEV_grid(cf, s, W);
  }
  return m_beta * EV;
}

// calculate the expected value of PrevIteration.
double MertonParams::EV_raw () const {
  double EV = -DBL_MAX;
  if (m_bUseMonteCarlo == true) {  
    ddFnObj Vfn = boost::bind(interp1d_grid, m_pStateGrid, m_pPrevIterationArray, _1);
	EV = calculateEV_montecarlo_1d(Vfn, m_RandomDraws);
  } else {
    ddFnObj cdfFn = boost::bind(truncateIfLessThanZero, m_CDFFn, _1);		// boost lognormal cdf won't take arg < 0
    ddFnObj pdfFn = boost::bind(truncateIfLessThanZero, m_PDFFn, _1);
    EV = calculateEV_grid(m_pStateGrid, m_pPrevIterationArray, cdfFn, pdfFn, my_identity<double>(), m_PrevIteration[0], m_PrevIteration[m_PrevIteration.size()-1]);
  }
  return EV;
}

BOOST_PYTHON_MODULE(_merton)
{                              
  bpl::class_<MertonParams, bpl::bases<BellmanParams>>("MertonParams", bpl::init<DoublePyArray, double, double, double, double, double, double, bool>())
		.def("u", &MertonParams::u)
		.def("EV", &MertonParams::EV)
		.def("EV_raw", &MertonParams::EV_raw)
		.def_readwrite("useMonteCarlo", &MertonParams::m_bUseMonteCarlo)		
    ;  
}                                          
