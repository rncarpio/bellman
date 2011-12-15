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
#include <tuple>

#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <numpy/arrayobject.h>
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range2d.h"

#include "myTypes.h"
#include "maximizer.h"

#define foreach         BOOST_FOREACH
#define reverse_foreach BOOST_REVERSE_FOREACH

namespace bpl = boost::python;
// Intel TBB multi-threaded library
using namespace tbb;


// 2d grid search.  returns # of values found
int gridSearch2D(DoublePyArray const &controlGrid1, DoublePyArray const &controlGrid2, MaximizerCallParams &params,  double &rMaxVal, double &rArgmax1, double &rArgmax2) {
  assert(controlGrid1.ndim() == controlGrid2.ndim() && controlGrid1.ndim() == 1); 
  
  //(pA)->data + (i)*(pA)->strides[0]
  char *pData1 = NULL;
  char *pData2 = NULL;
  int len1 = controlGrid1.dims()[0];
  int len2 = controlGrid2.dims()[0];
  int stride1 = controlGrid1.strides()[0];
  int stride2 = controlGrid2.strides()[0];  
  double result, max;
  int i=0, count=0;
  // evaluate params.objectiveFunction at every point in the control grid
  DoublePyArray::const_iterator iter1, iter2;
  for (iter1 = controlGrid1.begin(); iter1 != controlGrid1.end(); iter1++) {
    for (iter2 = controlGrid2.begin(); iter2 != controlGrid2.end(); iter2++) {
	  DoubleVector args(2);
      args[0] = *iter1;
	  args[1] = *iter2;
      //result = (*pFn)(arg1, arg2, pArgs);
	  result = params.objectiveFunction(args);
	  if (i == 0) {
        max = result;
	    rArgmax1 = args[0];
	    rArgmax2 = args[1];
	    count = 1;
      } else {
        if (result > max) {
          max = result;
		  rArgmax1 = args[0];
	      rArgmax2 = args[1];
		  count = 1;
	    } else if (result == max) {
	      count++;
	    }
	  }
	  i++;
	}
  }
  rMaxVal = max;
  return count;
}

// 2d parallel version
// object to be used with TBB
class MaxIndexFnObj {  
public:
  double m_value_of_max;
  double m_argmax1, m_argmax2;
  MaximizerCallParams &m_params;
  //DoublePyArray const &m_controlGrid1;
  //DoublePyArray const &m_controlGrid2;
  PyArrayObject *m_pGrid1, *m_pGrid2;
  
  void operator()( const blocked_range2d<size_t, size_t>& r ) {   
    double value;
    DoubleVector args(2);		
    // r is a 2-dimensonal range
    for( size_t i=r.rows().begin(); i!=r.rows().end(); ++i ){
	  //pData1 = (m_pGrid1->data) + (m_stride1 * i);
      for( size_t j=r.cols().begin(); j!=r.cols().end(); ++j) {        
        //pData2 = (m_pGrid2->data) + (m_stride2 * j);	
		double arg1, arg2;
		arg1 = * (double*) PyArray_GETPTR1(m_pGrid1, i);
		arg2 = * (double*) PyArray_GETPTR1(m_pGrid2, j);
		args[0] = arg1;
		args[1] = arg2;
        //value = (*m_pFn)(arg1, arg2, m_pArgs);
		value = m_params.objectiveFunction(args);
        if (m_value_of_max == -DBL_MAX || value > m_value_of_max) {
          m_value_of_max = value;
          m_argmax1 = args[0];
		  m_argmax2 = args[1];
		}
      }
    }
  }
  MaxIndexFnObj( MaxIndexFnObj& x, split ) :  // split constructor from tbb    
    m_value_of_max(-DBL_MAX), // -DBL_MAX from <climits>
    m_argmax1(-DBL_MAX), m_argmax2(-DBL_MAX),
	m_params(x.m_params), 
	//m_controlGrid1(x.m_controlGrid1), m_controlGrid2(x.m_controlGrid2)	
	m_pGrid1(x.m_pGrid1), m_pGrid2(x.m_pGrid2)
  {
  }
  // this is where results from different threads will be compared
  void join( const MaxIndexFnObj& y ) {
    if (m_value_of_max == -DBL_MAX || y.m_value_of_max > m_value_of_max) {
      m_value_of_max = y.m_value_of_max;
	  m_argmax1 = y.m_argmax1;
	  m_argmax2 = y.m_argmax2;
    }
  }  
  MaxIndexFnObj(DoublePyArray const &controlGrid1, DoublePyArray const &controlGrid2, MaximizerCallParams &params) :    
    m_value_of_max(-DBL_MAX), // -DBL_MAX from <climits>
    m_argmax1(-DBL_MAX), m_argmax2(-DBL_MAX),
	m_params(params), 
	//m_controlGrid1(controlGrid1), m_controlGrid2(controlGrid2)
	m_pGrid1((PyArrayObject*) controlGrid1.data().handle().get()), m_pGrid2((PyArrayObject*) controlGrid2.data().handle().get())
  {
  }
};

// TBB will divide up the task range among threads with a blocked_range.
// for N arbitrary dimensions, we need to use a 1d blocked_range, and then translate the index given in each range into the correct N-dimensional coordinates.
// wrap a gridArray with something that maps indices to a 1d range (obviously total size can't exceed the max value of an int)
inline unsigned int IndexListTo1D(IntVector const &lenArray, IntVector const &indexArray) {
  unsigned int i;  
  assert(lenArray.size() == indexArray.size());
  if (lenArray.size() == 0) {
    return 0;
  }
  unsigned int result = indexArray[0];
  for (i=1; i<lenArray.size(); i++) {
    result *= lenArray[i];
    result += indexArray[i];
  }
  return result;
}

inline void Index1DToArray(unsigned int index, IntVector const &lenArray, IntVector &dest) {
  int i;
  //dest.resize(lenArray.size());
  for (i=lenArray.size()-1; i>=0; i--) {
    dest[i] = index % lenArray[i];
	index /= lenArray[i];
  }
  assert(index == 0);
  return;
}


// multidimensional version
class MaxIndexFnObj2 {  
public:
  double m_value_of_max;
  DoubleVector m_argmax;
  MaximizerCallParams &m_params;
  DoublePyArrayVector const &m_ControlGridArray;
  IntVector m_lenArray;  
  
  void operator_any( const blocked_range<size_t>& r ) {   
    char *pData = NULL;    
	double value, arg;
	unsigned int j;
	IntVector indexArray(m_lenArray.size());
	DoubleVector argArray(m_lenArray.size());
    
    for( size_t index=r.begin(); index!=r.end(); ++index ){
	  Index1DToArray(index, m_lenArray, indexArray);
	  for (j=0; j<indexArray.size(); j++) {
	    //pData = (m_gridPtrArray[j]->data) + (m_gridPtrArray[j]->strides[0] * indexArray[j]);
		pData = ((char*) (m_ControlGridArray[j].array().data())) + (m_ControlGridArray[j].strides()[0] * indexArray[j]);
		arg = * (double*) pData;
		argArray[j] = arg;
	  }
      value = m_params.objectiveFunction(argArray);	  
      if (value > m_value_of_max) {
        m_value_of_max = value;
		m_argmax = argArray;
      }
    }
  }

  void operator()( const blocked_range<size_t>& r ) {
    int nArgs = m_lenArray.size();
	switch (nArgs) {
	  case 1:
	    operator_1(r);
		break;
	  case 2:
	    operator_2(r);
		break;
	  case 3:
	    operator_3(r);
		break;
	  default:
	    operator_any(r);
		break;
	}
  }

  void operator_1( const blocked_range<size_t>& r ) {   
	double value;
	IntVector indexArray(m_lenArray.size());
	DoubleVector argArray(m_lenArray.size());
    Index1DToArray(r.begin(), m_lenArray, indexArray);
	
	int nStride0 = m_ControlGridArray[0].strides()[0];
	const char *pBegin0 = (const char*) (m_ControlGridArray[0].array().data());
	const char *pEnd0 = pBegin0 + (nStride0 * m_ControlGridArray[0].size());
	const char *pData0 = pBegin0 + (nStride0 * indexArray[0]);
	
    for( size_t index=r.begin(); index!=r.end(); ++index ){
      argArray[0] = * (double*) pData0;

  	  value = m_params.objectiveFunction(argArray);	  
      if (value > m_value_of_max) {
        m_value_of_max = value;
		m_argmax = argArray;
      }
	  
	  pData0 += nStride0;	  
    }
  }
  void operator_2( const blocked_range<size_t>& r ) {   
	double value;
	IntVector indexArray(m_lenArray.size());
	DoubleVector argArray(m_lenArray.size());	
    Index1DToArray(r.begin(), m_lenArray, indexArray);

	int nStride0 = m_ControlGridArray[0].strides()[0];
	const char *pBegin0 = (const char*) (m_ControlGridArray[0].array().data());
	const char *pEnd0 = pBegin0 + (nStride0 * m_lenArray[0]);
	const char *pData0 = pBegin0 + (nStride0 * indexArray[0]);
	int nStride1 = m_ControlGridArray[1].strides()[0];
	const char *pBegin1 = (const char*) (m_ControlGridArray[1].array().data());
	const char *pEnd1 = pBegin1 + (nStride1 * m_lenArray[1]);
	const char *pData1 = pBegin1 + (nStride1 * indexArray[1]);
	
    for( size_t index=r.begin(); index!=r.end(); ++index ){
      argArray[0] = * (double*) pData0;
	  argArray[1] = * (double*) pData1;

  	  value = m_params.objectiveFunction(argArray);	  
      if (value > m_value_of_max) {
        m_value_of_max = value;
		m_argmax = argArray;
      }

	  pData1 += nStride1;
	  if (pData1 == pEnd1) {
	    pData1 = pBegin1;
	    pData0 += nStride0;
	  }	  
    }
  }
  void operator_3( const blocked_range<size_t>& r ) {   
	double value;
	IntVector indexArray(m_lenArray.size());
	DoubleVector argArray(m_lenArray.size());
    Index1DToArray(r.begin(), m_lenArray, indexArray);

	int nStride0 = m_ControlGridArray[0].strides()[0];
	const char *pBegin0 = (const char*) (m_ControlGridArray[0].array().data());
	const char *pEnd0 = pBegin0 + (nStride0 * m_ControlGridArray[0].size());
	const char *pData0 = pBegin0 + (nStride0 * indexArray[0]);
	int nStride1 = m_ControlGridArray[1].strides()[0];
	const char *pBegin1 = (const char*) (m_ControlGridArray[1].array().data());
	const char *pEnd1 = pBegin1 + (nStride1 * m_ControlGridArray[1].size());
	const char *pData1 = pBegin1 + (nStride1 * indexArray[1]);
	int nStride2 = m_ControlGridArray[2].strides()[0];
	const char *pBegin2 = (const char*) (m_ControlGridArray[2].array().data());
	const char *pEnd2 = pBegin2 + (nStride2 * m_ControlGridArray[2].size());
	const char *pData2 = pBegin2 + (nStride2 * indexArray[2]);
	
    for( size_t index=r.begin(); index!=r.end(); ++index ){
      argArray[0] = * (double*) pData0;
	  argArray[1] = * (double*) pData1;
	  argArray[2] = * (double*) pData2;

  	  value = m_params.objectiveFunction(argArray);	  
      if (value > m_value_of_max) {
        m_value_of_max = value;
		m_argmax = argArray;
      }

	  pData2 += nStride2;
	  if (pData2 == pEnd2) {
	    pData2 = pBegin2;
	    pData1 += nStride1;
		if (pData1 == pEnd1) {
		  pData1 = pBegin1;
		  pData0 += nStride0;
		}
	  }	  
    }
  }

  void operator_cp( const blocked_range<size_t>& r ) {   
    char *pData = NULL;    
	double value;

    CartesianProductIterator iter(m_ControlGridArray, r.begin());    
	CartesianProductIterator end(m_ControlGridArray, r.end());    
    for( size_t index=r.begin(); index!=r.end(); ++index, iter++ ){
	  value = m_params.objectiveFunction(*iter);
      if (value > m_value_of_max) {
        m_value_of_max = value;
		m_argmax = *iter;
      }
    }
  }
  
  MaxIndexFnObj2( MaxIndexFnObj2& x, split ) :  // split constructor from tbb    
    m_value_of_max(-DBL_MAX), // -DBL_MAX from <climits>
    m_argmax(x.m_argmax),
	m_params(x.m_params), m_ControlGridArray(x.m_ControlGridArray), m_lenArray(x.m_lenArray)	
  {
  }
  // this is where results from different threads will be compared
  void join( const MaxIndexFnObj2& y ) {
    if (y.m_value_of_max > m_value_of_max) {
      m_value_of_max = y.m_value_of_max;
	  m_argmax = y.m_argmax;	  
    }
  }  
  MaxIndexFnObj2(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params) :    
    m_value_of_max(-DBL_MAX), // -DBL_MAX from <climits>
    m_argmax(controlGridArray.size(), -DBL_MAX), m_lenArray(controlGridArray.size()),
	m_params(params), m_ControlGridArray(controlGridArray)
  {
    for (unsigned int i=0; i<controlGridArray.size(); i++) {
	  m_lenArray[i] = controlGridArray[i].dims()[0];
	}
  }
};

int gridSearch2DParallel(DoublePyArray const &controlGrid1, DoublePyArray const &controlGrid2, MaximizerCallParams &params, double &rMaxVal, double &rArgmax1, double &rArgmax2) {
  assert(controlGrid1.ndim() == controlGrid2.ndim() &&  controlGrid1.ndim() == 1); 
  
  MaxIndexFnObj fnObj(controlGrid1, controlGrid2, params);
  int len1 = controlGrid1.dims()[0];
  int len2 = controlGrid2.dims()[0];  
  //printf("calling parallel_reduce\n");
  parallel_reduce( blocked_range2d<size_t>(0, len1, 16, 0, len2, 16), fnObj);  
  //printf("returned from parallel_reduce\n");
  rMaxVal = fnObj.m_value_of_max;
  rArgmax1 = fnObj.m_argmax1;
  rArgmax2 = fnObj.m_argmax2;
  return 1;
}

// parallel version with arbitrary dimensions
int gridSearchParallel(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params, double &rMaxVal, DoubleVector &rArgMaxArray) {  
  int nGrids = controlGridArray.size();
  int i;
  IntVector lenArray(nGrids);
  
  for (i=0; i<nGrids; i++) {
    // check that all grids are 1-dimensional
    assert(controlGridArray[i].ndim() == 1);  
	lenArray[i] = controlGridArray[i].dims()[0];
  }

  unsigned int totalGridSize = 1;
  double totalGridSize2 = 1.0;
  for (i=0; i<nGrids; i++) {
	totalGridSize *= lenArray[i];
	totalGridSize2 *= double(lenArray[i]);
  }
  // check that total grid space isn't too large to be indexed
  assert(totalGridSize2 < double(UINT_MAX));
  
  MaxIndexFnObj2 fnObj(controlGridArray, params);  
  parallel_reduce( blocked_range<size_t>(0, totalGridSize), fnObj);  

  rMaxVal = fnObj.m_value_of_max;
  rArgMaxArray.resize(fnObj.m_argmax.size());
  copy(fnObj.m_argmax.begin(), fnObj.m_argmax.end(), rArgMaxArray.begin());
  return 1;
}
  
// single-threaded grid search with an arbitrary number of dimensions
int gridSearch(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params, double &rMaxVal, DoubleVector &rArgMaxArray) {  
  int nGrids = controlGridArray.size();
  int i;
  IntVector lenArray(nGrids), strideArray(nGrids), dataIndexArray(nGrids);
  
  for (i=0; i<nGrids; i++) {
    // check that all grids are 1-dimensional
    assert(controlGridArray[i].ndim() == 1);  
	lenArray[i] = controlGridArray[i].dims()[0];
	strideArray[i] = controlGridArray[i].strides()[0];
	dataIndexArray[i] = 0;
  }

  DoubleVector argArray(nGrids);  
  int nIter = 0;
  int nMaxMultiplicity = 0;  
  bool bDone = false;
  
  // check for zero size, if one grid has zero size then there's nothing to do
  for (i=0; i<nGrids; i++) {
    if (lenArray[i] == 0) {
	  bDone = true;
	  break;
	}
  }
  double max = -DBL_MAX;
  while (!bDone) {
    double result;
	// get double array args from char* data
	for (i=0; i<nGrids; i++) {
	  char *pData = (char*) (controlGridArray[i].array().data());
	  char *pArg = pData + (strideArray[i] * dataIndexArray[i]);
	  argArray[i] = *(double*) pArg;
	}
	result = params.objectiveFunction(argArray);
	if (nIter == 0) {			// first iteration
      max = result;
	  rArgMaxArray.resize(argArray.size());
	  copy(argArray.begin(), argArray.end(), rArgMaxArray.begin());		
	  nMaxMultiplicity = 1;
    } else {
      if (result > max) {
        max = result;
		rArgMaxArray.resize(argArray.size());
		copy(argArray.begin(), argArray.end(), rArgMaxArray.begin());		
	    nMaxMultiplicity = 1;
	  } else if (result == max) {
	    nMaxMultiplicity++;
	  }
	}
	nIter++;

    // increment indices for next iteration
	bDone = true;
	for (i=nGrids-1; i>=0; i--) {
	  // increment index i
	  dataIndexArray[i] += 1;
	  if (dataIndexArray[i] == lenArray[i]) {
	    // cycle this index
		dataIndexArray[i] = 0;
	  } else {
	    bDone = false;
		break;
	  }
	  // if we make it here, all indices have cycled, therefore we are done
	}
	
  }
  rMaxVal = max;
  return nMaxMultiplicity;
}

bpl::tuple maximizer2d_wrapper(DoublePyArray const &controlGrid1, DoublePyArray const &controlGrid2, bpl::object const &params, bool bUseC, bool bParallel) {
  int count = 0;
  double argmax1, argmax2, maxval;
  MaximizerCallParams& p = bpl::extract<MaximizerCallParams&>(params);   
  maximizer2d(controlGrid1, controlGrid2, p, count, argmax1, argmax2, maxval, bUseC, bParallel);
  return bpl::make_tuple(count, argmax1, argmax2, maxval);
}

void maximizer2d(DoublePyArray const &controlGrid1, DoublePyArray const &controlGrid2, MaximizerCallParams &params, int &rCount, double &rArgmax1, double &rArgmax2, double &rMaxval, bool bUseC, bool bParallel) {
  rCount = 0;
  if (bUseC) {
	if (!bParallel) {      
	  rCount = gridSearch2D(controlGrid1, controlGrid2, params, rMaxval, rArgmax1, rArgmax2);	
	} else {	  
	  rCount = gridSearch2DParallel(controlGrid1, controlGrid2, params, rMaxval, rArgmax1, rArgmax2);	
	}
  } else {
    // use c++ version
    DoubleVector argmaxArray(2);  
    DoublePyArrayVector controlGridArray(2);
    controlGridArray[0] = controlGrid1;
    controlGridArray[1] = controlGrid2;    
	if (!bParallel) {
      rCount = gridSearch(controlGridArray, params, rMaxval, argmaxArray);
	} else {
	  rCount = gridSearchParallel(controlGridArray, params, rMaxval, argmaxArray);
	}
    rArgmax1 = argmaxArray[0];
    rArgmax2 = argmaxArray[1];
	// printf("result: maxval=%f argmax_d=%f argmax_r=%f\n", maxval, argmax1, argmax2);
  }
  return;
}

// grid search maximizer that takes an arbitrary dimension of state vars, control vars
// 3 args: gridList, argList, w.f
// gridList is a sequence of grids for control vars
// argList is a sequence of doubles, for the state vars
// w.f is a multi-dimensional array
void my_maximizer(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params, int &rCount, DoubleVector &rArgmaxArray, double &rMaxval, bool bParallel) {
  rCount = 0;
   // if (controlGridArray.size() == 2) {
     // double argmax1, argmax2;
	 // rCount = gridSearch2DParallel(controlGridArray[0], controlGridArray[1], params, rMaxval, argmax1, argmax2);	
	 // rArgmaxArray[0] = argmax1;
	 // rArgmaxArray[1] = argmax1;
   // } else {
	  if (!bParallel) {
		rCount = gridSearch(controlGridArray, params, rMaxval, rArgmaxArray);
	  } else {
		rCount = gridSearchParallel(controlGridArray, params, rMaxval, rArgmaxArray);
	  }  
 // }
  return;
}

void my_maximizer2(DoublePyArrayVector const &controlGridArray, MaximizerCallParams &params, int &rCount, DoubleVector &rArgmaxArray, double &rMaxval, bool bParallel) {
  int nGrids = controlGridArray.size();
  unsigned int totalGridSize = 1;
  double totalGridSize2 = 1.0;
  for (int i=0; i<nGrids; i++) {
	totalGridSize *= controlGridArray[i].size();
	totalGridSize2 *= double(controlGridArray[i].size());
  }
  // check that total grid space isn't too large to be indexed
  assert(totalGridSize2 < double(UINT_MAX));
  assert(nGrids > 0);
  
  // calculate the objective function over every grid point
  std::vector<double> fArray(totalGridSize);
}

bpl::tuple my_maximizer_wrapper(bpl::list const &controlGridArrayList, bpl::object const &params, bool bParallel) {
  int count = 0;
  int i;
  double maxval;
  // first arg should be a sequence of DoubleArrays
  DoublePyArrayVector controlGridArrays(bpl::len(controlGridArrayList));
  DoubleVector argmax(bpl::len(controlGridArrayList));  
  for (i=0; i<bpl::len(controlGridArrayList); i++) {
    controlGridArrays[i] = bpl::extract<DoublePyArray>(controlGridArrayList[i]);
  }
  MaximizerCallParams& p = bpl::extract<MaximizerCallParams&>(params);  
  
  my_maximizer(controlGridArrays, p, count, argmax, maxval, bParallel);
  bpl::list argmaxList;
  for (i=0; i<bpl::len(controlGridArrayList); i++) {
    argmaxList.append(argmax[i]);
  }
  return bpl::make_tuple(count, argmaxList, maxval);
}

double MaximizerCallParams::objectiveFunction_wrap(bpl::list const &args) const {
  DoubleVector args2(bpl::len(args));
  for (int i=0; i<bpl::len(args); i++) {
	args2[i] = bpl::extract<double>(args[i]);
  }	
  return this->objectiveFunction(args2);
}

// maximizer test call params
// objective function is an array that will be interpolated
class TestParamsArray: public MaximizerCallParams {
public:
  // must be mt-safe. don't use boost::python handles!
  double objectiveFunction(DoubleVector const &args) const {    
    double result=DBL_MAX, arg1, arg2, arg3;
    if (args.size() == 1) {
  	  arg1 = args[0];
  	  result = interp1d_grid(m_pGrid1, m_pF, arg1);
    } else if (args.size() == 2) {
  	  arg1 = args[0];
	  arg2 = args[1];		
	  result = interp2d_grid(m_pGrid1, m_pGrid2, m_pF, arg1, arg2);		
	  //printf("objectiveFn: %f %f -> %f\n", arg1, arg2, result);
    } else if (args.size() == 3) {
	  arg1 = args[0];
	  arg2 = args[1];
	  arg3 = args[2];
	  result = interp3d_grid(m_pGrid1, m_pGrid2, m_pGrid3, m_pF, arg1, arg2, arg3);		
    } else {
	  assert(false);
    }
	return result;
  }

  DoublePyArray getFunctionArray() {
    return m_F;
  }
  void setFunctionArray1d(DoublePyArray const &grid1, DoublePyArray const &F) {
    if (!(grid1.ndim() == 1 && F.ndim() == 1 && grid1.size() == F.size())) {
      PyErr_SetString(PyExc_ValueError, "setFunctionArray1d: input args have wrong size/dimension");
      bpl::throw_error_already_set();
	}
    m_grid1 = grid1;
	m_F = F;
	m_pGrid1 = (PyArrayObject const*) m_grid1.data().handle().get();
	m_pF = (PyArrayObject const*) m_F.data().handle().get();
  }
  void setFunctionArray2d(DoublePyArray const &grid1, DoublePyArray const &grid2, DoublePyArray const &F) {
    if (!(grid1.ndim() == 1 && grid2.ndim() == 1 && F.ndim() == 2 && grid1.size() * grid2.size() == F.size())) {
      PyErr_SetString(PyExc_ValueError, "setFunctionArray2d: input args have wrong size/dimension");
      bpl::throw_error_already_set();
	}
    m_grid1 = grid1;
	m_grid2 = grid2;
	m_F = F;
	m_pGrid1 = (PyArrayObject const*) m_grid1.data().handle().get();
	m_pGrid2 = (PyArrayObject const*) m_grid2.data().handle().get();
	m_pF = (PyArrayObject const*) m_F.data().handle().get();	
  }
  void setFunctionArray3d(DoublePyArray const &grid1, DoublePyArray const &grid2, DoublePyArray const &grid3, DoublePyArray const &F) {
    if (!(grid1.ndim() == 1 && grid2.ndim() == 1 && grid3.ndim() == 1 && F.ndim() == 3 && grid1.size() * grid2.size() * grid3.size() == F.size())) {
      PyErr_SetString(PyExc_ValueError, "setFunctionArray3d: input args have wrong size/dimension");
      bpl::throw_error_already_set();
	}
    m_grid1 = grid1;
	m_grid2 = grid2;
	m_grid3 = grid3;
	m_F = F;
	m_pGrid1 = (PyArrayObject const*) m_grid1.data().handle().get();
	m_pGrid2 = (PyArrayObject const*) m_grid2.data().handle().get();
	m_pGrid3 = (PyArrayObject const*) m_grid3.data().handle().get();
	m_pF = (PyArrayObject const*) m_F.data().handle().get();	
  }
    
  // the DoublePyArray objects will handle the reference counting.
  // the PyArrayObject pointers are what will actually get used, for speed and thread safety reasons
  DoublePyArray m_grid1, m_grid2, m_grid3;  
  DoublePyArray m_F;
  PyArrayObject const *m_pGrid1, *m_pGrid2, *m_pGrid3, *m_pF;
};

// objectiveFunction will call a python function
// obviously, this is not MT-safe
class TestParamsFn: public MaximizerCallParams {
public:
  // must be mt-safe. don't use boost::python handles!
  double objectiveFunction(DoubleVector const &args) const {    
    bpl::list args2;
	for (DoubleVector::const_iterator iter=args.begin(); iter != args.end(); iter++) {
	  args2.append(*iter);
	}
	bpl::object result = m_CallbackFn(args2);
	return bpl::extract<double>(result);
  }

  void setObjFn(bpl::object const &obj) {
    m_CallbackFn = obj;
  }
  bpl::object m_CallbackFn;
};

// testing stuff
double test1(DoublePyArray Grid1, DoublePyArray pGrid2, DoublePyArray F, double xi, double yi) {
  return 0.0;
}

void test2(MaximizerCallParams &params, double &rVal) {
  rVal = 1.0;
  return;
}

  class hello
  {
    public:
      hello(const std::string& country) { this->country = country; }
      std::string greet() const { return "Hello from " + country; }
    private:
      std::string country;
  };
  
bpl::list cartesianProduct(bpl::list const &gridArrayList) {
  int i;
  // first arg should be a sequence of DoubleArrays
  DoublePyArrayVector gridArrays(bpl::len(gridArrayList));  
  for (i=0; i<bpl::len(gridArrayList); i++) {
    gridArrays[i] = bpl::extract<DoublePyArray>(gridArrayList[i]);
  }
  CartesianProductIterator end = CartesianProduct_end(gridArrays);
  bpl::list result;
  for (CartesianProductIterator iter = CartesianProduct_begin(gridArrays); iter != end; iter++) {
    bpl::list item;
    DoubleVector prod = (*iter);
	foreach(double x, prod) {
	  //printf("%f\n", x);
	  item.append(x);
	}
	result.append(item);
  }
  return result;
}

void test3() {
  int ints[] = {1, 2, 3, 4};
  foreach(int i, ints) {
    printf("%d\n", i);
  }
}

BOOST_PYTHON_MODULE(_maximizer)
{ 
  boost::python::def("test1", test1);
  boost::python::def("test2", test2);
  boost::python::def("test3", test3);
  boost::python::def("cartesianProduct", cartesianProduct);
  
  boost::python::def("maximizer2d", maximizer2d_wrapper);
  boost::python::def("maximizer", my_maximizer_wrapper);  
  
  bpl::class_<MaximizerCallParams, boost::noncopyable>("MaximizerCallParams", bpl::no_init)
		.def("objectiveFunction", &MaximizerCallParams::objectiveFunction_wrap)
	;
  bpl::class_<TestParamsArray, bpl::bases<MaximizerCallParams>>("TestParamsArray", bpl::init<>())
        .def("getFunctionArray", &TestParamsArray::getFunctionArray)
		.def("setFunctionArray1d", &TestParamsArray::setFunctionArray1d)
		.def("setFunctionArray2d", &TestParamsArray::setFunctionArray2d)
		.def("setFunctionArray3d", &TestParamsArray::setFunctionArray3d)				
    ;
  bpl::class_<TestParamsFn, bpl::bases<MaximizerCallParams>>("TestParamsFn", bpl::init<>())
        .def("setObjFn", &TestParamsFn::setObjFn)
    ;
  bpl::class_<BellmanParams, bpl::bases<MaximizerCallParams>, boost::noncopyable>("BellmanParams", bpl::init<>())
		.def("setStateVars", &BellmanParams::setStateVars)
		.def("getControlGridList", &BellmanParams::getControlGridList)
		.def("getNControls", &BellmanParams::getNControls)
		.def("setPrevIteration", &BellmanParams::setPrevIteration)
	;	
  
  bpl::class_<hello>("hello", bpl::init<std::string>())
        .def("greet", &hello::greet)  // Add a regular member function.        
    ;	
}                                          

 

