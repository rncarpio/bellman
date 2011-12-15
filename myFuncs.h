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
  

// some mathematical functions

#ifndef _myfuncs_h
#define _myfuncs_h

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <string>
#include <vector>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/python.hpp>

#include <pyublas/numpy.hpp>
#include "myTypes.h"

using boost::math::isnan;
namespace bpl = boost::python;

double DLLEXPORT calculateEV_grid(PyArrayObject const *pGrid, PyArrayObject const *pFArray, ddFnObj cdfFn, ddFnObj pdfFn, ddFnObj inverseFn, double leftK, double rightK);
double DLLEXPORT calculateEV_montecarlo_1d(ddFnObj fFn, DoubleVector const &draws);

// grid utility functions
 int getCellIndex(double value, PyArrayObject const *pGrid) {
  double dx = *ARRAYPTR1D(pGrid, 1) - *ARRAYPTR1D(pGrid, 0);
  if (value < *ARRAYPTR1D(pGrid, 0)) {
    return -1;
  } else if (value >= *ARRAYPTR1D(pGrid, ARRAYLEN1D(pGrid)-1)) {
    return ARRAYLEN1D(pGrid) - 2;
  } else {
    int result = (int) floor((value - *ARRAYPTR1D(pGrid, 0)) / dx);
	if (result == (ARRAYLEN1D(pGrid) - 1)) {
	  result--;
	}
	return result;
  }
}
 int getCellIndex_wrap(double value, DoublePyArray const &grid) {
  PyArrayObject const *pGrid = (PyArrayObject const*) grid.data().handle().get();
  return getCellIndex(value, pGrid);
}  

// if x is outside grid, force it to boundaries
  double forceToGrid(double x, PyArrayObject const *pGrid) {
  double first = *ARRAYPTR1D(pGrid, 0);
  if (x < first) {
    return first;
  }
  double last = *ARRAYPTR1D(pGrid, ARRAYLEN1D(pGrid)-1);
  if (x > last) {
    return last;
  }
  return x;
}
 double forceToGrid_wrap(double value, DoublePyArray const &grid) {
  PyArrayObject const *pGrid = (PyArrayObject const*) grid.data().handle().get();
  return forceToGrid(value, pGrid);
}  

// 1d interpolation between 2 points, x1,x2 with function values f1 = f(x1), f2 = f(x2)
 double interp1d(double z, double x1, double f1, double x2, double f2) {
  if (z <= x1) {
    return f1;
  }
  if (z >= x2) {
    return f2;
  }
  double result = f1 + (f2-f1)*(z-x1)/(x2-x1);
  return result;
}

 double interp1d_grid(PyArrayObject const *pGrid, PyArrayObject const *pF, double xi) {
  double a, x1, x2, f1, f2, result;
  int i;
  a = forceToGrid(xi, pGrid);
  i = getCellIndex(a, pGrid);
  x1 = *ARRAYPTR1D(pGrid, i);
  x2 = *ARRAYPTR1D(pGrid, i+1);
  f1 = *ARRAYPTR1D(pF, i);
  f2 = *ARRAYPTR1D(pF, i+1);
  result = interp1d(xi, x1, f1, x2, f2);
  return result;
}
 double interp1d_grid_wrap(DoublePyArray const &grid1, DoublePyArray const &F, double xi) {
  PyArrayObject const *pGrid1 = (PyArrayObject const*) grid1.data().handle().get();
  PyArrayObject const *pF = (PyArrayObject const*) F.data().handle().get();
  return interp1d_grid(pGrid1, pF, xi);
}

class Interp1D {
public:
  DoubleVector m_grid, m_vals, m_slope;
  double m_dx;

  template<class Array1D_T1, class Array1D_T2>
  Interp1D(Array1D_T1 const &gridArray, Array1D_T2 const &valsArray) {
    if (gridArray.size() != valsArray.size()) throw std::logic_error("x and y must be the same size");	
	if (!(gridArray.size() >= 2)) throw std::logic_error("size must be at least 2");	
	m_grid.resize(gridArray.size());
	m_vals.resize(valsArray.size());
	m_slope.resize(gridArray.size() - 1);
    std::copy(gridArray.begin(), gridArray.end(), m_grid.begin());
	std::copy(valsArray.begin(), valsArray.end(), m_vals.begin());
	for (unsigned int i=0; i<gridArray.size()-1; i++) {
	  m_slope[i] = (m_vals[i+1] - m_vals[i]) / (m_grid[i+1] - m_grid[i]);
	}
	m_dx = m_grid[1] - m_grid[0];
  }
  
  double interp(double xi) const {
    // check if outside grid
	if (xi < m_grid.front()) {
	  return m_vals.front();
	}
	if (xi > m_grid.back()) {
	  return m_vals.back();
	}
	// get cell
    int cell = (int) floor((xi - m_grid.front()) / m_dx);	
	if (cell == m_grid.size()-1) {
	  cell--;
	}
    // interp
    double result = m_vals[cell] + (xi - m_grid[cell]) * m_slope[cell];
	return result;
  }
  double operator() (double xi) {
    return interp(xi);
  }  
  template <typename Iter1, typename ResultT>
  ResultT interp_vector(Iter1 xBegin, Iter1 xEnd) const {
    ResultT result(xEnd-xBegin);
	std::transform(xBegin, xEnd, result.begin(), *this);
	return result;
  }
  // apply f to a sequence sorted in ascending order, then sum up.  used for monte carlo integration.
  template <typename Array1D>
  double apply_sum_sorted_seq(Array1D const &sorted) const {
    return apply_sum_sorted(sorted.begin(), sorted.end());    
  }
  template <typename Iter1>
  double apply_sum_sorted(Iter1 begin, Iter1 end) const {
    double sum = 0.0, incr, cell_x_sum;
	Iter1 i=begin;
	unsigned int currentCell = 0;
	int count = 0;
	while (*i < m_grid.front() && i != end) {	// skip over points below grid
	  count++;
	  i++;
	}
	sum += m_vals.front() * (double) count;	
	while (currentCell < m_grid.size()-1 && i != end) {
	  cell_x_sum = 0.0;
	  count = 0;
	  while (*i < m_grid[currentCell+1] && i != end) {	
	    //incr = m_vals[currentCell] + (*i - m_grid[currentCell]) * m_slope[currentCell];
		cell_x_sum += *i;
		count++;
		//sum += incr;		
	    i++;
	  }
	  incr = m_vals[currentCell]*count + (cell_x_sum - m_grid[currentCell]*count) * m_slope[currentCell];
	  sum += incr;
	  currentCell++;
	}
	count = 0;
	while (i != end) {
	  count++;
	  i++;
	}
	sum += m_vals.back() * (double) count;	
	return sum;
  }
};

typedef Interp1D PyInterp1D;

class Interp2D {
public:
  DoubleVector m_grid1, m_grid2;
  typedef boost::numeric::ublas::matrix<double> DoubleMatrix;
  typedef boost::numeric::ublas::matrix_row<DoubleMatrix> DoubleMatrixRow;
  DoubleMatrix m_vals;
  typedef std::shared_ptr<Interp1D> InterpPtr;
  //typedef Interp1D *InterpPtr;
  std::vector<InterpPtr> m_Interps;   // we'll keep a size1-length vector of 1d interps, each one interpolates over grid2
  double m_dx1;
  
  template<class Array1D_T1, class Array1D_T2, class Array2D>
  Interp2D(Array1D_T1 const &grid1, Array1D_T2 const &grid2, Array2D const &vals) {
    if (grid1.size() != vals.size1() || grid2.size() != vals.size2()) throw std::logic_error("grid does not match array size");	
	if (grid1.size() < 2 || grid2.size() < 2) throw std::logic_error("size must be at least 2");	
	// copy grids
	m_grid1.resize(grid1.size());
	m_grid2.resize(grid2.size());
	std::copy(grid1.begin(), grid1.end(), m_grid1.begin());
	std::copy(grid2.begin(), grid2.end(), m_grid2.begin());
	// copy vals
	m_vals.resize(grid1.size(), grid2.size());
	for (int i=0; i<vals.size1(); i++) {
	  for (int j=0; j<vals.size2(); j++) {
	    m_vals(i,j) = vals(i,j);
	  }
	}
	// construct interp1d vector
	for (int i=0; i<grid1.size(); i++) {
	  DoubleMatrixRow row (m_vals, i);
	  m_Interps.push_back(InterpPtr(new Interp1D(grid2, row)));
	}
	m_dx1 = m_grid1[1] - m_grid1[0];
  }
  double interp(double x1, double x2) const {
    // check if outside grid1
	if (x1 < m_grid1.front()) {
	  x1 = m_grid1.front();
	}
	if (x1 > m_grid1.back()) {
	  x1 = m_grid1.back();
	}
	// get cell
    int cell = (int) floor((x1 - m_grid1.front()) / m_dx1);	
	if (cell == m_grid1.size()-1) {
	  cell--;
	}
    // interp along grid2
	double i1 = m_Interps[cell]->interp(x2);
	double i2 = m_Interps[cell+1]->interp(x2);
	double result = interp1d(x1, m_grid1[cell], i1, m_grid1[cell+1], i2);
	return result;
  }
  double operator() (double x1, double x2) {
    return interp(x1, x2);
  }   
  double interp_tuple (bpl::tuple const &x) {
    double x1 = bpl::extract<double>(x[0]);
	double x2 = bpl::extract<double>(x[1]);
    return interp(x1, x2);
  }     
  double interp_list (bpl::list const &x) {
    double x1 = bpl::extract<double>(x[0]);
	double x2 = bpl::extract<double>(x[1]);
    return interp(x1, x2);
  }     
  
};
typedef Interp2D PyInterp2D;

// bilinear interpolation between 4 points of a rectangle, corners (x1,y1), (x2,y2)
// f_1_1 = f(x1,y1)
 double interp2d(double z1, double z2, double x1, double y1, double x2, double y2, double f_1_1, double f_1_2, double f_2_1, double f_2_2) {  
  // interp1d along y1 line, then along y2 line
  double f_y1 = interp1d(z1, x1, f_1_1, x2, f_2_1);
  double f_y2 = interp1d(z1, x1, f_1_2, x2, f_2_2);
  // then, interp1d again
  double result = interp1d(z2, y1, f_y1, y2, f_y2);
  return result;
}

// bilinear interploation of F, given grid points in grid1, grid2
 double interp2d_grid(PyArrayObject const *pGrid1, PyArrayObject const *pGrid2, PyArrayObject const *pF, double xi, double yi) {
  double a,b;
  int i,j;
  double x1,x2,y1,y2,u1,u2,u3,u4;
  // figure out which cell xi,yi is in
  a = forceToGrid(xi, pGrid1);
  b = forceToGrid(yi, pGrid2);
  i = getCellIndex(a, pGrid1);
  j = getCellIndex(b, pGrid2);
  // get values of x,y,f at corners
  x1 = *(double*) PyArray_GETPTR1(pGrid1, i);
  x2 = *(double*) PyArray_GETPTR1(pGrid1, i+1);
  y1 = *(double*) PyArray_GETPTR1(pGrid2, j);
  y2 = *(double*) PyArray_GETPTR1(pGrid2, j+1);
  u1 = *(double*) PyArray_GETPTR2(pF, i, j);
  u2 = *(double*) PyArray_GETPTR2(pF, i+1, j);
  u3 = *(double*) PyArray_GETPTR2(pF, i, j+1);
  u4 = *(double*) PyArray_GETPTR2(pF, i+1, j+1);
  // interpolate
  double result = interp2d(xi, yi, x1, y1, x2, y2, u1, u3, u2, u4);
  return result;
}
 double interp2d_grid_wrap(DoublePyArray const &grid1, DoublePyArray const &grid2, DoublePyArray const &F, double xi, double yi) {
  PyArrayObject const *pGrid1 = (PyArrayObject const*) grid1.data().handle().get();
  PyArrayObject const *pGrid2 = (PyArrayObject const*) grid2.data().handle().get();
  PyArrayObject const *pF = (PyArrayObject const*) F.data().handle().get();
  return interp2d_grid(pGrid1, pGrid2, pF, xi, yi);
}
  
// trilinear interpolation
// pF is a 3d array of doubles
// pGrid1-3 are 1d arrays with the grid coords (must be evenly spaced)
// return interpolated value f(x1, x2, x3)
 double interp3d_grid(PyArrayObject const *pGrid1, PyArrayObject const *pGrid2, PyArrayObject const *pGrid3, PyArrayObject const *pF,
    double xi, double yi, double zi) {
  double a, b, c;
  int i, j, k;
  double x1, x2, y1, y2, z1, z2;
  double u1, u2, u3, u4, u5, u6, u7, u8;
  double w1, w2, w3, w4, w5, w6, w7, u;

  a = forceToGrid(xi, pGrid1);
  b = forceToGrid(yi, pGrid2);
  c = forceToGrid(zi, pGrid3);
  i = getCellIndex(a, pGrid1);
  j = getCellIndex(b, pGrid2);
  k = getCellIndex(c, pGrid3);
  
  x1 = *ARRAYPTR1D(pGrid1, i);
  x2 = *ARRAYPTR1D(pGrid1, i+1);
  y1 = *ARRAYPTR1D(pGrid2, j);
  y2 = *ARRAYPTR1D(pGrid2, j+1);
  z1 = *ARRAYPTR1D(pGrid3, k);
  z2 = *ARRAYPTR1D(pGrid3, k+1);
  
  u1 = *ARRAYPTR3D(pF, i, j, k);
  u2 = *ARRAYPTR3D(pF, i+1, j, k);
  u3 = *ARRAYPTR3D(pF, i, j+1, k);
  u4 = *ARRAYPTR3D(pF, i+1, j+1, k);
  u5 = *ARRAYPTR3D(pF, i, j, k+1);
  u6 = *ARRAYPTR3D(pF, i+1, j, k+1);
  u7 = *ARRAYPTR3D(pF, i, j+1, k+1);
  u8 = *ARRAYPTR3D(pF, i+1, j+1, k+1);

  w1 = u2 + (u2-u1)/(x2-x1)*(a-x2);
  w2 = u4 + (u4-u3)/(x2-x1)*(a-x2);
  w3 = w2 + (w2-w1)/(y2-y1)*(b-y2);
  w4 = u5 + (u6-u5)/(x2-x1)*(a-x1);
  w5 = u7 + (u8-u7)/(x2-x1)*(a-x1);
  w6 = w4 + (w5-w4)/(y2-y1)*(b-y1);
  w7 = w3 + (w6-w3)/(z2-z1)*(c-z1);
  u = w7;

  return u;
}
double interp3d_grid_wrap(DoublePyArray const &grid1, DoublePyArray const &grid2, DoublePyArray const &grid3, 
    DoublePyArray const &F, double xi, double yi, double zi) {
  PyArrayObject const *pGrid1 = (PyArrayObject const*) grid1.data().handle().get();
  PyArrayObject const *pGrid2 = (PyArrayObject const*) grid2.data().handle().get();
  PyArrayObject const *pGrid3 = (PyArrayObject const*) grid3.data().handle().get();
  PyArrayObject const *pF = (PyArrayObject const*) F.data().handle().get();
  return interp3d_grid(pGrid1, pGrid2, pGrid3, pF, xi, yi, zi);
}

// integrate a given array of f(x) on a grid of x using the trapezoidal rule
template <class ContainerA, class ContainerB>
double trapezoid_integrate2(const ContainerA &y,  const ContainerB &x) {
    if (x.size() != y.size()) {
        throw std::logic_error("x and y must be the same size");
    }
    double sum = 0.0;
    for (unsigned int i = 1; i < x.size(); i++) {
        sum += (x[i] - x[i-1]) * (y[i] + y[i-1]);	
    }
    return sum * 0.5;
}

// first arg is f(x), second arg is x							 
template <class InputIterator1, class InputIterator2>
double trapezoid_integrate(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2) {
  double y0 = *first1;
  double x0 = *first2;
  InputIterator1 iy = first1;
  InputIterator2 ix = first2;
  ix++;
  iy++;
  double sum=0.0;
  for (; iy != last1; ix++, iy++) {
    sum += ((*ix) - x0) * ((*iy) + y0);
    x0 = *ix;	
    y0 = *iy;
  }
  return sum * 0.5;
}

// probability distributions
struct MyDDFnObj {
  virtual double operator() (double x) const = 0;
  virtual operator ddFnObj() = 0;
};

template <class Distribution>
struct CDFFnObj : public MyDDFnObj {
  CDFFnObj(double arg1, double arg2): m_dist(arg1, arg2) {}
  double operator() (double x) const { assert(!isnan(x)); return cdf(m_dist, x); }
  operator ddFnObj() { return *this; }
  Distribution m_dist;
};
template <class Distribution>
struct PDFFnObj : public MyDDFnObj {
  PDFFnObj(double arg1, double arg2): m_dist(arg1, arg2) {}
  double operator() (double x) const { assert(!isnan(x)); return pdf(m_dist, x); }
  operator ddFnObj() { return *this; }
  Distribution m_dist;
};

typedef CDFFnObj<boost::math::normal> NormalCDFObj;
typedef PDFFnObj<boost::math::normal> NormalPDFObj;
//typedef CDFFnObj<boost::math::lognormal> LognormalCDFObj;
// boost's lognormal CDF won't take args < 0.0
struct LognormalCDFObj : public MyDDFnObj {
  boost::math::lognormal m_dist;
  LognormalCDFObj(double mean, double sd) : m_dist(mean, sd) {}
  double operator() (double x) const {
    assert(!isnan(x));
    if (x <= 0.0) {
	  return 0.0;
	} else {
	  return cdf(m_dist, x);
	}
  }
  operator ddFnObj() { return *this; }
};
//typedef PDFFnObj<boost::math::lognormal> LognormalPDFObj;
struct LognormalPDFObj : public MyDDFnObj {
  boost::math::lognormal m_dist;
  LognormalPDFObj(double mean, double sd) : m_dist(mean, sd) {}
  double operator() (double x) const {
    assert(!isnan(x));
    if (x <= 0.0) {	  
	  return 0.0;
	} else {
	  return pdf(m_dist, x);
	}
  }
  operator ddFnObj() { return *this; }
};

double lognormal_PartialExp(double k, double mean, double sd) {
  NormalCDFObj stdnormCDF(0.0, 1.0);
  return exp(mean + 0.5*sd*sd) * stdnormCDF( (mean + (sd*sd) - log(k)) / sd );
}

// instead of a y=x line, take the expectation of Ax+B
double lognormal_PartialExp_Affine(double k, double A, double B, double mean, double sd) {
  LognormalCDFObj lognormCDF(mean, sd);
  return A*lognormal_PartialExp(k, mean, sd) + B*(1.0 - lognormCDF(k));
}

// calculate expectation of a linear fn on an interval using lognormal RV
double lognormal_EV_interval(double x0, double x1, double y0, double y1, double mean, double sd) {
  assert(x0 < x1);
  assert(x0 >= 0.0);
  double slope = (y1-y0)/(x1-x0);
  double y_intercept = y0 - slope*x0;
  return lognormal_PartialExp_Affine(x0, slope, y_intercept, mean, sd) - lognormal_PartialExp_Affine(x1, slope, y_intercept, mean, sd);
}

template <typename Range>
double lognormal_EV_lininterp(Range const &fGrid, Range const &fVals, double mean, double sd, ddFnObj inverseFn) {
  DoubleVector grid2(fGrid.size());
  DoubleVector vals2(fVals.begin(), fVals.end());
  std::transform(fGrid.begin(), fGrid.end(), grid2.begin(), inverseFn);
  // if grid2 is decreasing, reverse it and vals2
  if (grid2[1] < grid2[0]) {
    std::reverse(grid2.begin(), grid2.end());
	std::reverse(vals2.begin(), vals2.end());
  }
  // now, grid2 must be increasing
  LognormalCDFObj lognormCDF(mean, sd);  
  double below = vals2[0] * lognormCDF(grid2[0]);
  double above = vals2[vals2.size()-1] * (1.0 - lognormCDF(grid2[grid2.size()-1]));
  double between = 0.0;
  assert(grid2.size() >= 2);
  for (unsigned int i=0; i<grid2.size()-1; i++) {
    double x0 = grid2[i];
	double x1 = grid2[i+1];
	double y0 = vals2[i];
	double y1 = vals2[i+1];
    if (x1 <= 0.0) {	// since a lognormal rv is >= 0, this occurs with zero probability, skip it
	  continue;
	}
	if (x0 <= 0.0) {      // if the interval contains 0, cut it
	  y0 = interp1d(0.0, x0, y0, x1, y1);
	  x0 = 0.0;
	}
    between += lognormal_EV_interval(x0, x1, y0, y1, mean, sd);
  }
  return below + above + between;
}

// utility functions
// exponential
struct exponential {
  exponential(double theta) :m_theta(theta) {}
  double operator() (double c) const { return 1.0 - exp(-m_theta * c); }
  double m_theta;
};
struct linear {
  double operator() (double c) const { return c;}
};
struct CRRA {
  CRRA(double gamma) :m_gamma(gamma) {}
  double operator() (double c) const { return pow(c, 1-m_gamma) / (1-m_gamma); }
  double m_gamma;
};
struct LogUtil {
  double operator() (double c) const { return log(c); }
};

// misc

template <class T> 
class my_identity {
  public:
    T operator()(T const &t) const { return t; }
};

#endif //_myfuncs_h
