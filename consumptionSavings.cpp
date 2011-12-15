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
  

// consumption-savings problem (Phelps 1962, Hakansson 1970)

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <random>
#include <boost/timer.hpp>

#include "consumptionSavings.h"
#include "maximizer.h"
#include "debugMsg.h"
#include "myfuncs.h"
#include "cudaMonteCarlo.h"

namespace bpl = boost::python;
using namespace boost;
using namespace python;
using namespace pyublas;
using namespace std;

static double g_TotalElapsedTime;
static int g_nEVCalls;

ConsumptionSavingsParams::ConsumptionSavingsParams(DoublePyArray const &stateGrid, double gamma, double beta, double mean1, double mean2, double var2,
  EVMethodT evMethod)
: m_StateGrid(stateGrid), m_gamma(gamma), m_beta(beta), m_mean1(mean1), m_mean2(mean2), m_var2(var2)
{
  m_pStateGrid = (PyArrayObject const*) m_StateGrid.data().handle().get();
  assert(gamma >= 1.0);
  if (gamma == 1.0) {	// log utility
	m_uFn = LogUtil();
  } else {
	m_uFn = CRRA(gamma);
  }	  
  
  // EV stuff
  m_EVMethod = evMethod;
  m_nDraws = 20000;
  std::mt19937 rng;
  // note: lognormal_distribution<>, the RNG, takes mean and sd as args, not loc and scale!
  std::normal_distribution<> normal(mean2, sqrt(var2));  
  m_RandomDrawsSorted.resize(m_nDraws);
  std::generate(m_RandomDrawsSorted.begin(), m_RandomDrawsSorted.end(), 
    [&] () -> double { return exp(normal(rng)); });
  std::sort(m_RandomDrawsSorted.begin(), m_RandomDrawsSorted.end());
  // timing EV
  g_TotalElapsedTime = 0.0;
  g_nEVCalls = 0;  
}

// given lognormal shock Z (distributed with mean2, var2), calculate next period's wealth.
double Z_to_nextW(double s1, double s2, double W, double expMean1, double Z) {
  double nextW = s1*W*expMean1 + s2*W*Z;
  return nextW;
}
// s2*W must be nonzero
double nextW_to_Z(double s1, double s2, double W, double expMean1, double nextW) {
  double Z = (nextW - s1*W*expMean1) / (s2*W);  
  return Z;
}

void ConsumptionSavingsParams::setPrevIteration(DoublePyArray const &WArray) {
  m_PrevIteration = WArray;	
  m_pPrevIterationArray = (PyArrayObject const*) m_PrevIteration.data().handle().get();
  m_pPrevIterationInterp.reset(new PyInterp1D(m_StateGrid, m_PrevIteration));

  printf("%d calls, avg time per EV call: %f\n", g_nEVCalls, g_TotalElapsedTime/g_nEVCalls);
  g_TotalElapsedTime = 0.0;
  g_nEVCalls = 0;
  if (m_EVMethod == EV_CUDA_MONTECARLO) {
    const double *pFGridBegin = &m_StateGrid[0];
    const double *pFGridEnd = &m_StateGrid[0] + m_StateGrid.size();
    const double *pFValsBegin = &WArray[0];
    const double *pFValsEnd = &WArray[0] + WArray.size();
    const double *pRandomDrawsBegin = &m_RandomDrawsSorted[0];
    const double *pRandomDrawsEnd = &m_RandomDrawsSorted[0] + m_RandomDrawsSorted.size();  
    //cuda_setup(pFGridBegin, pFGridEnd, pFValsBegin, pFValsEnd, pRandomDrawsBegin, pRandomDrawsEnd);
  }
}

double ConsumptionSavingsParams::calcEV(double s1, double s2, double W, double expMean1) const {
  double result = -DBL_MAX;
  boost::timer t1;
  switch (m_EVMethod) {
    case EV_MONTECARLO:
      result = calculateEV_montecarlo_1d(
        [=](double Z) -> double { return (*(this->m_pPrevIterationInterp))(Z_to_nextW(s1, s2, W, expMean1, Z)); }, 
	    m_RandomDrawsSorted
	  );
	  break;
	case EV_MONTECARLO2: {
	    // pre-apply Z_to_nextW to random draws.  must be monotonic
	    DoubleVector draws2(m_RandomDrawsSorted.size());
	    std::transform(m_RandomDrawsSorted.begin(), m_RandomDrawsSorted.end(), draws2.begin(), [=] (double Z) -> double 
		  { return Z_to_nextW(s1, s2, W, expMean1, Z); });
	    double sum = 0.0;
		if (s2*W >= 0.0) {	// if the coefficient of Z is positive, then applying Z_to_nextW to an ascending sequence will give an ascending seq
		  sum = m_pPrevIterationInterp->apply_sum_sorted(draws2.begin(), draws2.end());
		} else {			// otherwise, the sequence will become descending -> pass the reverse iterator
		  sum = m_pPrevIterationInterp->apply_sum_sorted(draws2.rbegin(), draws2.rend());
		}
	    result = sum / draws2.size();
	  }
	  break;
	case EV_CUDA_MONTECARLO:
	  //result = cuda_calcEV(s1, s2, W, expMean1);
	  break;
	case EV_PARTIAL_EXP: {
	    // TODO: doesn't give the same answers. figure it out later
	    // use the formula for partial expectations of a lognormal variable	  		
        if (s2*W == 0.0) {
		  double nextW = s1*W*expMean1;
          result = (*m_pPrevIterationInterp)(nextW);
		} else {
	      result = lognormal_EV_lininterp(m_StateGrid, m_PrevIteration, m_mean2, sqrt(m_var2), [=] (double nextW) -> double
	        { return nextW_to_Z(s1, s2, W, expMean1, nextW); });	   
		}
	  }
	  break;
	default:	  
	  assert(false);
	  result = -DBL_MAX;
  }
  double elapsed = t1.elapsed();
  g_TotalElapsedTime += elapsed;
  g_nEVCalls++;
  
  return result;
}
    	  
double ConsumptionSavingsParams::objectiveFunction(DoubleVector const &controlVars) const {
  double W = m_wealth;				// wealth
  double c = controlVars[0];		
  double cf = c/W;
  double s1 = controlVars[1];		// fraction of wealth invested in asset 1
  double s2 = 1.0 - cf - s1;		// fraction of wealth invested in asset 2
  
  double expMean1 = exp(m_mean1);
  double EV = calcEV(s1, s2, W, expMean1);  
  double result = m_uFn(cf*W) + m_beta * EV;
  return result;
}

double ConsumptionSavingsParams::u (double cf, double s1) const {
  double W = m_wealth;				// wealth
  return m_uFn(cf*W);
}

double ConsumptionSavingsParams::EV (double cf, double s1) const {
  double s2 = 1.0 - cf - s1;		// fraction of wealth invested in asset 2
  double W = m_wealth;				// wealth
  double expMean1 = exp(m_mean1);
  return calcEV(s1, s2, W, expMean1);    
}


BOOST_PYTHON_MODULE(_consumptionSavings)
{                              
  bpl::class_<ConsumptionSavingsParams, bpl::bases<BellmanParams>>("ConsumptionSavingsParams", bpl::init<DoublePyArray, 
    double, double, double, double, double, EVMethodT>())
		.def_readonly("wealth", &ConsumptionSavingsParams::m_wealth)
		.def("u", &ConsumptionSavingsParams::u)
		.def("EV", &ConsumptionSavingsParams::EV)
    ;  
  enum_<EVMethodT>("EVMethodT")
        .value("EV_MONTECARLO", EV_MONTECARLO)
		.value("EV_MONTECARLO2", EV_MONTECARLO2)
        .value("EV_CUDA_MONTECARLO", EV_CUDA_MONTECARLO)
		.value("EV_PARTIAL_EXP", EV_PARTIAL_EXP)
    ;
}                                          
