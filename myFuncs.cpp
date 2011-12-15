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
#include <functional>

#include <boost/python.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <numpy/arrayobject.h>

#include "myTypes.h"
#include "myFuncs.h"


using boost::tuples::tuple;
using boost::math::normal;
namespace bpl = boost::python;

double mysum(DoublePyArray arr) {
  PyArrayObject *pArr = (PyArrayObject *) arr.data().handle().get();
  double sum = std::accumulate(PyArray_begin(pArr), PyArray_end(pArr), 0.0);
  return sum;
}

double mytrapz2(DoublePyArray y, DoublePyArray x) {
  return trapezoid_integrate2(y, x);
}

double mytrapz(DoublePyArray y, DoublePyArray x) {
  PyArrayObject *pX = (PyArrayObject *) x.data().handle().get();
  PyArrayObject *pY = (PyArrayObject *) y.data().handle().get();
  return trapezoid_integrate(PyArray_begin(pY), PyArray_end(pY), PyArray_begin(pX));
}

// calculate expected value on a grid
// cdfFn, pdfFn are boost function objects
// if there is a change of variables, inverseFn is the inverse transformation
double calculateEV_grid(PyArrayObject const *pGrid, PyArrayObject const *pFArray, ddFnObj cdfFn, ddFnObj pdfFn, ddFnObj inverseFn, double leftK, double rightK) {
  assert(ARRAYLEN1D(pGrid) == ARRAYLEN1D(pFArray));
  assert(pGrid->nd == 1);
  // below is the integral to the left of the grid
  double below = leftK * cdfFn(inverseFn(*ARRAYPTR1D(pGrid, 0)));
  // above is the integral to the right of the grid
  double above = rightK * (1.0 - cdfFn(inverseFn(*ARRAYPTR1D(pGrid, ARRAYLEN1D(pGrid)-1))));
  // between is the integral on the grid.  evaluate f*pdf on the grid points and integrate
  DoubleVector betweenArray(ARRAYLEN1D(pGrid));
  PyArrayIterator iter1, iter2, last1;
  DoubleVector::iterator out;
  last1 = PyArray_end(pGrid);
  for (iter1=PyArray_begin(pGrid), iter2=PyArray_begin(pFArray), out=betweenArray.begin(); 
       iter1 != last1; 
	   iter1++, iter2++, out++) {
    *out = pdfFn(inverseFn(*iter1)) * (*iter2);
  }	
  double between = trapezoid_integrate(betweenArray.begin(), betweenArray.end(), PyArray_begin(pGrid));
  return below + above + between;
}

// calculate expected value of f(z) using grids for f() and the PDF of z.
template<class Range>
double calculateEV_grid2(PyArrayObject const *pFGrid, PyArrayObject const *pFVals, 
                        Range const &pdfGrid, Range const &pdfVals, ddFnObj inverseFn) {
  assert(pdfGrid.size() == pdfVals.size());						
  DoubleVector x(pdfGrid.size());
  DoubleVector::iterator xi;
  Range::const_iterator iter1, iter2;
  for (iter1=pdfGrid.begin(), iter2=pdfVals.begin(), xi=x.begin(); iter1 != pdfGrid.end(); iter1++, iter2++, xi++) {
    (*xi) = (*iter2) * interp1d_grid(pFGrid, pFVals, inverseFn(*iter1));
  }
  double result = trapezoid_integrate(x.begin(), x.end(), pdfGrid.begin());
  return result;
}
						

// calculate expected value using monte carlo
double calculateEV_montecarlo_1d(ddFnObj fFn, DoubleVector const &draws) {
  // DoubleVector fArray(draws.size());
  // std::transform(draws.begin(), draws.end(), fArray.begin(), fFn);
  // double sum = std::accumulate(fArray.begin(), fArray.end(), 0.0);
  // return sum / draws.size();
  double sum=0.0;
  std::for_each(draws.begin(), draws.end(), [&] (double x) { sum += fFn(x); } );
  return sum / draws.size();
}

double calculateEV_grid_wrapper(DoublePyArray x, DoublePyArray y, bpl::object const &fn1, bpl::object const &fn2, double offset, double leftK, double rightK) {
  PyArrayObject *pX = (PyArrayObject *) x.data().handle().get();
  PyArrayObject *pY = (PyArrayObject *) y.data().handle().get();
  MyDDFnObj& cdfFn = bpl::extract<MyDDFnObj&>(fn1);
  MyDDFnObj& pdfFn = bpl::extract<MyDDFnObj&>(fn2);
  ddFnObj offsetFn = boost::bind(std::plus<double>(), -offset, _1);
  return calculateEV_grid(pX, pY, cdfFn, pdfFn, offsetFn, leftK, rightK);
}

double calculateEV_grid2_wrapper(DoublePyArray const &fGrid, DoublePyArray const &fVals, DoublePyArray const &pdfGrid, DoublePyArray const &pdfVals) {
  PyArrayObject *pFGrid = (PyArrayObject *) fGrid.data().handle().get();
  PyArrayObject *pFVals = (PyArrayObject *) fVals.data().handle().get();
  return calculateEV_grid2(pFGrid, pFVals, pdfGrid, pdfVals, my_identity<double>());
}

// convert Python sequences of doubles <-> DoubleVector
struct DoubleVector_to_list
{
    static PyObject* convert(DoubleVector const &vec) {
	  bpl::list result;
	  DoubleVector::const_iterator iter;
	  for (iter=vec.begin(); iter != vec.end(); iter++) {
	    result.append(*iter);
	  }
      return boost::python::incref(result.ptr());
    }
};

double lognormal_EV_lininterp_wrap(DoublePyArray const &fGrid, DoublePyArray const &fVals, double mean, double sd) {
  return lognormal_EV_lininterp(fGrid, fVals, mean, sd, std::identity<double>());
}

BOOST_PYTHON_MODULE(_myfuncs)
{ 
  bpl::to_python_converter<DoubleVector, DoubleVector_to_list>();		// register conversion
  
  bpl::def("mysum", mysum);
  bpl::def("mytrapz", mytrapz);
  bpl::def("mytrapz2", mytrapz2);
  bpl::def("mycalcEV_grid", calculateEV_grid_wrapper);
  bpl::def("mycalcEV_grid2", calculateEV_grid2_wrapper);
  bpl::def("lognormal_EV_lininterp", lognormal_EV_lininterp_wrap);
  boost::python::def("getCellIndex", getCellIndex_wrap);
  boost::python::def("forceToGrid", forceToGrid_wrap);
  boost::python::def("interp1d_single", interp1d);
  boost::python::def("interp2d_single", interp2d);
  boost::python::def("interp1d", interp1d_grid_wrap);
  boost::python::def("interp2d", interp2d_grid_wrap);  
  boost::python::def("interp3d", interp3d_grid_wrap);

  bpl::class_<PyInterp1D>("Interp1D", bpl::init<DoublePyArray, DoublePyArray>())
		.def("__call__", &PyInterp1D::interp)  
		.def("__call__", &PyInterp1D::interp_vector<DoublePyArray::const_iterator, DoublePyArray>)
		.def("applySorted", &PyInterp1D::apply_sum_sorted_seq<DoublePyArray>)
	;  
  bpl::class_<PyInterp2D>("Interp2D", bpl::init<DoublePyArray, DoublePyArray, DoublePyMatrix>())
		.def("__call__", &PyInterp2D::interp_tuple)  
		.def("__call__", &PyInterp2D::interp_list)
	;  
  
  bpl::class_<MyDDFnObj, boost::noncopyable>("MyDDFnObj", bpl::no_init)
		.def("__call__", &MyDDFnObj::operator())  
	;  
  bpl::class_<NormalCDFObj, bpl::bases<MyDDFnObj>>("NormalCDFObj", bpl::init<double, double>()) 
    ;	  
  bpl::class_<NormalPDFObj, bpl::bases<MyDDFnObj>>("NormalPDFObj", bpl::init<double, double>())  
    ;	  
  bpl::class_<LognormalCDFObj, bpl::bases<MyDDFnObj>>("LognormalCDFObj", bpl::init<double, double>()) 
    ;	  
  bpl::class_<LognormalPDFObj, bpl::bases<MyDDFnObj>>("LognormalPDFObj", bpl::init<double, double>())  
    ;	  
	
}
