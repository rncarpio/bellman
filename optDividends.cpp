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

#include "optDividends.h"
#include "maximizer.h"
#include "debugMsg.h"
#include "myfuncs.h"

namespace bpl = boost::python;
using namespace boost;
using namespace python;
using namespace pyublas;
using namespace std;

OptDividendsParams::OptDividendsParams(double beta, DoublePyArray const &randomDrawsSorted)
: m_beta(beta)
{
  m_RandomDrawsSorted.resize(randomDrawsSorted.size());
  std::copy(randomDrawsSorted.begin(), randomDrawsSorted.end(), m_RandomDrawsSorted.begin());
}
	
double OptDividendsParams::objectiveFunction(DoubleVector const &controlVars) const {
  double d = controlVars[0];
  double M = m_M;
  // pre-apply Z_to_nextM to random draws.  must be monotonic
  DoubleVector draws2(m_RandomDrawsSorted.size());
  std::transform(m_RandomDrawsSorted.begin(), m_RandomDrawsSorted.end(), draws2.begin(), [=] (double Z) -> double 
		  { return M - d + Z; });
  // find the first nextM that is >= 0
  auto firstNonNeg = std::find_if(draws2.begin(), draws2.end(), [] (double x) {return (x>=0.0);} );
  double sum = m_pPrevIterationInterp->apply_sum_sorted(firstNonNeg, draws2.end());
  double EV = sum / draws2.size();
  return d + m_beta * EV;
}

BOOST_PYTHON_MODULE(_optDividends)
{                              
  bpl::class_<OptDividendsParams, bpl::bases<BellmanParams>>("OptDividendsParams", bpl::init<double, DoublePyArray>())
    ;  
}                                          
