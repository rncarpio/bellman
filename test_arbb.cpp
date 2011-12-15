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
#include <string>
#include <vector>
#include <algorithm>

#include <boost/python.hpp>
#include <arbb.hpp>
#include "myTypes.h"
#include <numpy/arrayobject.h>

using namespace arbb;
namespace bpl = boost::python;

typedef void dd_arbb_fn(f64 const &, f64 const &, f64 &);

void arbb_crra(f64 const &c, f64 const &gamma, f64 &result) {
  result = pow(c, 1.0-gamma) / (1.0-gamma);
}

// How do we want to do this?
// - V function is in C++, templated to operate either on doubles or arbb::f64
//   - must be of type: void(T, T, T&) (result is last arg)
//   - e.g. identity(), stl functions, etc
// - draws is a DoubleVector


void do_montecarlo(f64 const &fnArg, arbb::dense<f64> const &draws, f64 &result) {
  arbb::dense<f64> fArray(draws.size());
  arbb::map(arbb_crra)(draws, fnArg, fArray);	// fArray = fFn(draws)
  //arbb::map(*pFn)(draws, fnArg, fArray);	// fArray = fFn(draws)
  f64 mysum = arbb::sum(fArray);
  result = mysum / (f64) (draws.size()[0]);
}

double arbb_EV_montecarlo_wrapper(double fnArg, DoublePyArray const &draws) {
  arbb::dense<f64> draws2(draws.size());
  f64 fnArg2 = fnArg;
  f64 result;
  arbb::range<arbb::f64> w_range = draws2.write_only_range();
  std::copy(draws.begin(), draws.end(), w_range.begin());  
  //arbb::call(do_montecarlo)(&arbb_crra, fnArg2, draws2, result);
  arbb::call(do_montecarlo)(fnArg2, draws2, result);
  double result2 = arbb::value(result);
  return result2;
}

void doublex (f64 in, f64 &out) {
  out = 2.0 * in;
}

double test1(double x) {
  f64 in = x;
  f64 out;
  arbb::call(doublex)(x, out);
  double result = arbb::value(out);
  return result;
}

BOOST_PYTHON_MODULE(_test_arbb)
{   
  bpl::def("arbb_EV", arbb_EV_montecarlo_wrapper);
  bpl::def("test1", test1);
}
  
