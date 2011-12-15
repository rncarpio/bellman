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
  

#ifndef _maximizer_h
#define _maximizer_h

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <string>
#include <vector>
#include <limits>

#include <pyublas/numpy.hpp>
#include "myTypes.h"
#include "myFuncs.h"

// maximize a function over a grid of control variables.

namespace bpl = boost::python;

// this object holds parameters of the maximization operation. e.g. state variables
// specific problems should inherit from this class
class MaximizerCallParams {
public:
  virtual double objectiveFunction(DoubleVector const &args) const = 0;				// this calculates the objective function.  it must be MT-safe, so it doesn't take python objects as args
  double objectiveFunction_wrap(boost::python::list const &args) const; 			// this wraps objectiveFunction() for python
  
  virtual ~MaximizerCallParams() {}
};

// this object is for use in solving Bellman equations with value or policy iteration.
// these methods are exposed to python
class BellmanParams : public MaximizerCallParams {
public:
  virtual double objectiveFunction(DoubleVector const &args) const { return std::numeric_limits<double>::quiet_NaN(); }
  virtual void setStateVars(boost::python::list const &stateVars) {		// set the state variables used in objectiveFunction()
  }
  virtual bpl::list getControlGridList(bpl::list const &stateVars) const {		// return a list of arrays that hold the grid for the control variables
    bpl::list result;
	return result;
  }
  virtual int getNControls() const {												// return the number of control variables
    return 0;
  }
  virtual void setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray) {		// set the previous value function.
  }
  virtual ~BellmanParams() {}
};

int gridSearch(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params, double &rMaxVal, DoubleVector &rArgMaxArray);
int gridSearchParallel(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params, double &rMaxVal, DoubleVector &rArgMaxArray);

// maximize an objective function over a 2-dimensional grid of control variables.
// return values: count (multiplicity), control1, control2, maxval (value of objective function)
void maximizer2d(DoublePyArray const &controlGrid1, DoublePyArray const &controlGrid2, MaximizerCallParams &params, int &rCount, double &rControl1, double &rControl2, double &rMaxval,
  bool bUseC, bool bParallel);

// same as maximizer2d, but for an arbitrary number of dimensions
// controlGrids is a std::vector of DoublePyArrays
void my_maximizer(DoublePyArrayVector const &controlGrids, MaximizerCallParams &params, int &rCount, DoubleVector &rArgmax, double &rMaxval);




	
#endif //_maximizer_h
