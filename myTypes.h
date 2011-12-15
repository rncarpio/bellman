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
  


#ifndef _myTypes_h
#define _myTypes_h

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <vector>
#include <functional>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/lambda/lambda.hpp>
#include <pyublas/numpy.hpp>

#define DLLEXPORT __declspec(dllexport)

#define CARRAYLEN(a) (sizeof(a) / sizeof(a[0]))
typedef PyArrayObject *PyArrayPtr;
typedef PyArrayPtr *PyArrayPtrPtr;
typedef unsigned int uint;

typedef std::vector<int> IntVector;
typedef std::vector<double> DoubleVector;
typedef pyublas::numpy_vector<double> DoublePyArray;
typedef pyublas::numpy_matrix<double> DoublePyMatrix;
typedef std::vector<DoublePyArray> DoublePyArrayVector;
// a function that takes a double, returns a double
typedef double (ddFn) (double arg);
// takes 2 doubles, returns double
typedef double (ddFn2) (double arg1, double arg2);
typedef double (DoublePyArrayFn) (DoublePyArray const &x, void *pArgs);
// C++ std versions
typedef std::tr1::function<double (double)> ddFnObj;
typedef std::tr1::function<double (double, double)> ddFn2Obj;
typedef std::tr1::function<double (DoublePyArray const &x, void *pArgs)> DoublePyArrayFnObj;

// access PyArrayObject elements as doubles
#define ARRAYLEN1D(pA)		((pA)->dimensions[0])
inline double* ARRAYPTR1D(PyArrayObject const *pA, int i) {
  assert(pA->nd == 1);
  assert(i >= 0);
  assert(i < pA->dimensions[0]);
  return (double*) PyArray_GETPTR1(pA, i);
}

inline double* ARRAYPTR2D(PyArrayObject const *pA, int i, int j) {
  assert(pA->nd == 2);
  assert(i >= 0);
  assert(j >= 0);
  assert(i < pA->dimensions[0]);
  assert(j < pA->dimensions[1]);
  return (double*) PyArray_GETPTR2(pA, i, j);
}

inline double* ARRAYPTR3D(PyArrayObject const *pA, int i, int j, int k) {
  assert(pA->nd == 3);
  assert(i >= 0);
  assert(j >= 0);
  assert(k >= 0);
  assert(i < pA->dimensions[0]);
  assert(j < pA->dimensions[1]);
  assert(k < pA->dimensions[2]);
  return (double*) PyArray_GETPTR3(pA, i, j, k);
}

// iterator wrapper around a PyArrayObject.
// pyublas wraps more functionality but is not thread-safe
class PyArrayIterator
  : public boost::iterator_facade<
        PyArrayIterator
      , double
      , boost::bidirectional_traversal_tag
    >
{
 public:
    PyArrayIterator()
      : m_pArrayObj(NULL), m_pData(NULL), m_nItemSize(0)
	  {}

    explicit PyArrayIterator(PyArrayObject const *pArrayObj, int offset)
      : m_pArrayObj(pArrayObj), m_pData(NULL), m_nItemSize(0)
	  {
	    assert(PyArray_ISCONTIGUOUS(pArrayObj));
		m_pData = (char*) PyArray_DATA(pArrayObj);
		m_nItemSize = PyArray_ITEMSIZE(pArrayObj);
		m_pData += m_nItemSize * offset;
	  }

 private:
    friend class boost::iterator_core_access;

    void increment() {
	  m_pData += m_nItemSize;
	}
    void decrement() {
	  m_pData -= m_nItemSize;
	}

    bool equal(PyArrayIterator const& other) const
    {
      assert(m_pArrayObj == other.m_pArrayObj);
	  return (m_pData == other.m_pData);
    }

    double& dereference() const { 
	  return * (double*) m_pData; 
	}

    PyArrayObject const* m_pArrayObj;
	char *m_pData;
	int m_nItemSize;
};

PyArrayIterator PyArray_begin(PyArrayObject const *pArrayObj) {
  return PyArrayIterator(pArrayObj, 0);
}
PyArrayIterator PyArray_end(PyArrayObject const *pArrayObj) {
  return PyArrayIterator(pArrayObj, PyArray_SIZE(pArrayObj));
}

// an iterator that will iterate through all combinations of elements in N PyArrays.
// used for evaluating a function at every grid point, given N grids.
class CartesianProductIterator
  : public boost::iterator_facade<
        CartesianProductIterator
      , DoubleVector const
      , boost::forward_traversal_tag
    >
{
 public:
    explicit CartesianProductIterator(DoublePyArrayVector const &grids, int offset=0, bool bEnd=false)
	: m_Grids(grids), m_CurrentValue(grids.size()), m_BeginIters(grids.size()), m_EndIters(grids.size()), m_CurrentIters(grids.size()), m_bReachedEnd(bEnd)
	{
	  // m_BeginIters will contain each grid's begin iter, same for m_EndIters
	  DoublePyArrayVector::const_iterator gridIter;
	  std::vector<DoublePyArray::const_iterator>::iterator itersIter;
	  for (gridIter=grids.begin(), itersIter=m_BeginIters.begin(); gridIter != grids.end(); gridIter++, itersIter++) {
		   (*itersIter) = (*gridIter).begin();
	  }
	  for (gridIter=grids.begin(), itersIter=m_EndIters.begin(); gridIter != grids.end(); gridIter++, itersIter++) {
		(*itersIter) = (*gridIter).end();
	  }
	  m_CurrentIters = m_BeginIters;
	  // jump to offset
	  int index = offset;		
	  for (int i=grids.size()-1; i>=0; i--) {
		m_CurrentIters[i] += index % grids[i].size();
		index /= grids[i].size();
	  }
	  // m_CurrentValues will store the values pointed to by m_CurrentIters
	  DoubleVector::iterator valsIter;
	  for (valsIter=m_CurrentValue.begin(), itersIter=m_CurrentIters.begin(); valsIter != m_CurrentValue.end(); valsIter++, itersIter++) {
		(*valsIter) = *(*itersIter);
	  }
	}
	
 private:
    friend class boost::iterator_core_access;

    void increment() {
	  assert(m_bReachedEnd == false);
	  std::vector<DoublePyArray::const_iterator>::reverse_iterator iter_current, iter_begin, iter_end;
	  DoubleVector::reverse_iterator iter_val;
	  // go from right to left
	  for (iter_current=m_CurrentIters.rbegin(), iter_begin = m_BeginIters.rbegin(), iter_end = m_EndIters.rbegin(), iter_val = m_CurrentValue.rbegin(); 
	       iter_current!=m_CurrentIters.rend(); 
		   iter_current++, iter_begin++, iter_end++, iter_val++) {
	    (*iter_current)++;						// increment the currently rightmost iterator
		if ((*iter_current) != (*iter_end)) {	// if it hasn't reached the end,
		  (*iter_val) = *(*iter_current);  // update the current value
		  return;                          // exit loop
		} else {                           // otherwise, we need to cycle this position back to the beginning, and move left
		  (*iter_current) = (*iter_begin);
		  (*iter_val) = *(*iter_current);
		}
	  }
	  m_bReachedEnd = true;					// if we've reached here, we've gone all the way around
	}
	
    bool equal(CartesianProductIterator const& other) const
    {      
	  assert(m_BeginIters == other.m_BeginIters);
	  return (m_CurrentIters == other.m_CurrentIters && m_bReachedEnd == other.m_bReachedEnd);
    }

    DoubleVector const& dereference() const { 
	  return m_CurrentValue;
	}

	DoublePyArrayVector const &m_Grids;
	DoubleVector m_CurrentValue;
	std::vector<DoublePyArray::const_iterator> m_BeginIters, m_EndIters, m_CurrentIters;
	bool m_bReachedEnd;
};

CartesianProductIterator CartesianProduct_begin(DoublePyArrayVector const &grids) {
  return CartesianProductIterator(grids);
}
CartesianProductIterator CartesianProduct_end(DoublePyArrayVector const &grids) {
  return CartesianProductIterator(grids, 0, true);
}

#endif //_myTypes_h
