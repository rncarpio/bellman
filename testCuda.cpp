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
#include <boost/random.hpp>
#include <boost/random/lognormal_distribution.hpp>

#include "myTypes.h"
#include "myfuncs.h"
#include "cudaMonteCarlo.h"

namespace bpl = boost::python;
using namespace boost;
using namespace python;
using namespace pyublas;
using namespace std;

void setup_wrap(DoublePyArray const &fGrid, DoublePyArray const &fVals, DoublePyArray const &randomDraws) {
  const double *pFGridBegin = &fGrid[0];
  const double *pFGridEnd = &fGrid[0] + fGrid.size();
  const double *pFValsBegin = &fVals[0];
  const double *pFValsEnd = &fVals[0] + fVals.size();
  const double *pRandomDrawsBegin = &randomDraws[0];
  const double *pRandomDrawsEnd = &randomDraws[0] + randomDraws.size();
  
  cuda_setup(pFGridBegin, pFGridEnd, pFValsBegin, pFValsEnd, pRandomDrawsBegin, pRandomDrawsEnd);
}

DoublePyArray interp1d_vector_wrap(DoublePyArray const &xArray) {  
  DoublePyArray result(xArray.size());
  const double *pArgBegin = &xArray[0];
  const double *pArgEnd = &xArray[0] + xArray.size();
  double *pResultBegin = &result[0];
  interp1d_vector(pArgBegin, pArgEnd, pResultBegin);
  return result;
}

bpl::list test_setup_wrap() {
  DoubleVector fGrid, fVals, fSlopes;
  test_setup(fGrid, fVals, fSlopes);
  bpl::list result;
  DoublePyArray grid(fGrid.size());
  DoublePyArray vals(fVals.size());
  DoublePyArray slopes(fSlopes.size());
  std::copy(fGrid.begin(), fGrid.end(), grid.begin());
  std::copy(fVals.begin(), fVals.end(), vals.begin());
  std::copy(fSlopes.begin(), fSlopes.end(), slopes.begin());  
  result.append(grid);
  result.append(vals);
  result.append(slopes);
  return result;
}

double test2(DoublePyArray x) {
  return x[0];
}

BOOST_PYTHON_MODULE(_testCuda)
{                  
  deviceReset();
  bpl::def("setup", setup_wrap);
  bpl::def("interp1d_vector", interp1d_vector_wrap);
  bpl::def("test_setup", test_setup_wrap);
  bpl::def("device_reset", deviceReset);
  bpl::def("test_interp2", test_interp2);
  bpl::def("calcEV", cuda_calcEV);
  
  bpl::def("test2", test2);
  bpl::def("test1", mytest1);
}
