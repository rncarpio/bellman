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

#include <Python.h>
#include <numpy/arrayobject.h>

// access array elements as doubles
//#define ARRAYPTR1D(pA, i)	(double*) ((pA)->data + (i)*(pA)->strides[0])
inline double* ARRAYPTR1D(PyArrayObject *pA, int i) {
  assert(pA->nd == 1);
  assert(i >= 0);
  assert(i < pA->dimensions[0]);
  return (double*) ((pA)->data + (i)*(pA)->strides[0]);
}
  
#define ARRAYLEN1D(pA)		((pA)->dimensions[0])

#define ARRAYPTR2D(pA, i, j)	(double*) ((pA)->data + (i)*(pA)->strides[0] + (j)*(pA)->strides[1])

//#define ARRAYPTR3D(pA,i,j,k)	(double*) ((pA)->data + (i)*(pA)->strides[0] + (j)*(pA)->strides[1] + (k)*(pA)->strides[2])
inline double* ARRAYPTR3D(PyArrayObject *pA, int i, int j, int k) {
  assert(pA->nd == 3);
  assert(i >= 0);
  assert(j >= 0);
  assert(k >= 0);
  assert(i < pA->dimensions[0]);
  assert(j < pA->dimensions[1]);
  assert(k < pA->dimensions[2]);
  return (double*) ((pA)->data + (i)*(pA)->strides[0] + (j)*(pA)->strides[1] + (k)*(pA)->strides[2]);
}

#define CARRAYLEN(a) (sizeof(a) / sizeof(a[0]))

// a function that takes a double, returns a double
typedef double (ddFn) (double arg);
// takes 2 doubles and void ptr, returns double
typedef double (ddFn2) (double arg1, double arg2, void *pArgs);
typedef PyArrayObject *PyArrayPtr;
typedef PyArrayPtr *PyArrayPtrPtr;
typedef unsigned int uint;

// global parameters
typedef struct globalParams {
  double theta;
  double beta;
  double gamma;
  PyArrayObject *pGrid1, *pGrid2, *pGrid3;
  PyArrayObject *pZVals, *pZProbs;
  ddFn *pU;
  ddFn *pFS;
} _globalParams;

typedef struct eu_params {
  double M, D, N;
  PyArrayObject *pW;
  bool bPrint;
} _eu_params;

struct globalParams g_Params;

// utility functions
// exponential
double U_exponential(double c) {
  double theta = g_Params.theta;
  return 1 - exp(-theta * c);
}
double Uprime_exponential(double c) {
  double theta = g_Params.theta;
  return theta * exp(-theta * c);
}
// linear
double U_linear(double c) {
  return c;
}
double Uprime_linear(double c) {
  return 1;
}
// CRRA
double U_crra(double c) {
  double gamma = g_Params.gamma;
  return pow(c, 1-gamma) / (1-gamma);
}
double Uprime_crra(double c) {
  double gamma = g_Params.gamma;
  return pow(c, -gamma);
}

// fraction of income deposited into bank
double fs(double r) {
  return 1.0 - exp(1.0 - r);
}

// default values. can be overridden later
void initGlobalParams() {
  g_Params.beta = 0.9;
  g_Params.theta = 0.5;
  g_Params.gamma = 2.0;
  g_Params.pGrid1 = g_Params.pGrid2 = g_Params.pGrid3 = NULL;
  g_Params.pZVals = g_Params.pZProbs = NULL;
  g_Params.pU = &U_exponential;
  g_Params.pFS = &fs;
}

PyObject* setGlobalParams(PyObject *self, PyObject *args) {
  double theta, beta, gamma;
  PyArrayObject *pGrid1=NULL, *pGrid2=NULL, *pGrid3=NULL;
  PyArrayObject *pZVals=NULL, *pZProbs=NULL;

  PyArrayPtrPtr src[] = {&pGrid1, &pGrid2, &pGrid3, &pZVals, &pZProbs};
  PyArrayPtrPtr dest[] = {&g_Params.pGrid1, &g_Params.pGrid2, &g_Params.pGrid3,
     &g_Params.pZVals, &g_Params.pZProbs};

  if (!PyArg_ParseTuple(args, "dddO!O!O!O!O!:setGlobalParams", &theta, &beta, &gamma,
      &PyArray_Type, &pGrid1, &PyArray_Type, &pGrid2, &PyArray_Type, &pGrid3, 
      &PyArray_Type, &pZVals, &PyArray_Type, &pZProbs)) {
    return NULL;
  }
  g_Params.beta = beta;
  g_Params.theta = theta;
  g_Params.gamma = gamma;
  // set pointer values
  for (uint i=0; i<CARRAYLEN(src); i++) {
    if (*(dest[i]) != NULL) {
      Py_DECREF(*(dest[i]));
    }
    Py_INCREF(*(src[i]));
    *(dest[i]) = *(src[i]);
  }
  
  // check that zProbs sum up to 1
  double sum = 0;
  for (uint i=0; i<ARRAYLEN1D(pZProbs); i++) {
    sum += * ARRAYPTR1D(pZProbs, i);
  }
  if (sum != 1.0) {
    PyErr_SetString(PyExc_ValueError, "z probs don't add up to 1");
    return NULL;
  }
  return Py_BuildValue("i", 1);
}


// return the cell that contains value.
// if below grid min, return -1
// if above grid max, return len(pGrid)-2
int getCellIndex(double value, PyArrayObject *pGrid) {
  double dx = *ARRAYPTR1D(pGrid, 1) - *ARRAYPTR1D(pGrid, 0);
  if (value < *ARRAYPTR1D(pGrid, 0)) {
    return -1;
  } else if (value >= *ARRAYPTR1D(pGrid, ARRAYLEN1D(pGrid)-1)) {
    return ARRAYLEN1D(pGrid) - 2;
  } else {
    int result = floor((value - *ARRAYPTR1D(pGrid, 0)) / dx);
	if (result == (ARRAYLEN1D(pGrid) - 1)) {
	  result--;
	}
	return result;
  }
}

// if x is outside grid, force it to boundaries
double forceToGrid(double x, PyArrayObject *pGrid) {
  double first = *ARRAYPTR1D(pGrid, 0);
  if (x < first) {
    return first;
  }
//  int len = ARRAYLEN1D(pGrid);
//  double last = *ARRAYPTR1D(pGrid, len - 1);
  double last = *ARRAYPTR1D(pGrid, ARRAYLEN1D(pGrid)-1);
  if (x > last) {
    return last;
  }
  return x;
}

// trilinear interpolation
// pF is a 3d array of doubles
// pGrid1-3 are 1d arrays with the grid coords (must be evenly spaced)
// return interpolated value f(x1, x2, x3)
double interpTrilinear(PyArrayObject *pGrid1, PyArrayObject *pGrid2, PyArrayObject *pGrid3, PyArrayObject *pF,
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

// test trilinear interpolation
double f3(double x, double y, double z) {
  return x + y + z;
}

PyObject *test3(PyObject *self, PyObject *args) {
  double x1, x2, x3, f;
  PyArrayObject *pGrid1=NULL, *pGrid2, *pGrid3, *pF;

  if (!PyArg_ParseTuple(args, "O!O!O!O!ddd:test3", &PyArray_Type, &pGrid1, &PyArray_Type, &pGrid2, 
      &PyArray_Type, &pGrid3, &PyArray_Type, &pF,
      &x1, &x2, &x3)) {
    return NULL;
  }
  f = interpTrilinear(pGrid1, pGrid2, pGrid3, pF, x1, x2, x3);
  return Py_BuildValue("d", f);
}

// 2d grid search.  returns # of values found
int gridSearch2D(PyArrayObject *pGrid1, PyArrayObject *pGrid2, ddFn2* pFn, void* pArgs, double *pArgmax1, double *pArgmax2) {
  assert(pGrid1->nd == pGrid2->nd &&  pGrid1->nd == 1); 
  
  //(pA)->data + (i)*(pA)->strides[0]
  char *pData1 = NULL;
  char *pData2 = NULL;
  int len1 = pGrid1->dimensions[0];
  int len2 = pGrid2->dimensions[0];
  int stride1 = pGrid1->strides[0];
  int stride2 = pGrid2->strides[0];  
  double result, max, arg1, arg2;
  int i=0, count=0;
  char *pLast1 = (pGrid1->data) + (stride1 * len1);
  char *pLast2 = (pGrid2->data) + (stride2 * len2);
  for (pData1=pGrid1->data; pData1 != pLast1; pData1 += stride1) {
    for (pData2=pGrid2->data; pData2 != pLast2; pData2 += stride2) {   
      arg1 = * (double*) pData1;
	  arg2 = * (double*) pData2;
      result = (*pFn)(arg1, arg2, pArgs);
	  if (i == 0) {
        max = result;
	    *pArgmax1 = arg1;
	    *pArgmax2 = arg2;
	    count = 1;
      } else {
        if (result > max) {
          max = result;
  	      *pArgmax1 = arg1;
	      *pArgmax2 = arg2;		
		  count = 1;
	    } else if (result == max) {
	      count++;
	    }
	  }
	  i++;
	}
  }
  return count;
}

// w_array is a 3D array
// grid_M, grid_D, grid_N are 1d arrays
// M, D, N are the state values
// d, r are the controls
// z_vals, z_probs are shock distribution values
// fs is a function that takes r, returns the fraction of income invested in bank (a double)
double expected_next_v
(PyArrayObject *pWArray, 
    PyArrayObject *pMGrid, PyArrayObject *pDGrid, PyArrayObject *pNGrid,
    double M, double D, double N, double d, double r,
    PyArrayObject *pZVals, PyArrayObject *pZProbs,
    ddFn *pFSFn, bool bPrint=false) {
  int zi, zlen;
  double *pZ, fs, probZ, nextN, nextM, nextD;
  double sum;

  assert(pZVals->nd == pZProbs->nd);
  assert(pZVals->dimensions[0] == pZProbs->dimensions[0]);
  zlen = pZVals->dimensions[0];
  // calculate expectation -> for each possible shock z...
  if (bPrint) {
    printf("expected_next_v M=%f D=%f N=%f d=%f r=%f\n", M,D,N,d,r);
  }
  sum = 0.0;
  for (zi=0; zi<zlen; zi++) {
    pZ = (double*) ARRAYPTR1D(pZVals, zi);
    probZ = * (double*) ARRAYPTR1D(pZProbs, zi);
    fs = (*pFSFn)(r);
    nextN = N * (*pZ);
    nextM = M + nextN * fs - D - d;
    nextD = r * fs * nextN;
    if (nextM <= 0.0) {
      sum += 0.0;
	  if (bPrint) {
	    printf("  probZ=%f z=%f nextN=%f nextM=%f nextD=%f: +0\n", probZ, *pZ, nextN, nextM, nextD);
	  }
    } else {
	  double interpVal = interpTrilinear(pMGrid, pDGrid, pNGrid, pWArray, nextM, nextD, nextN);
      double incr = probZ * interpVal;
	  sum += incr;
	  if (bPrint) {
	    printf("  probZ=%f z=%f nextN=%f nextM=%f nextD=%f: w=%f +%f\n", probZ, *pZ, nextN, nextM, nextD, interpVal, incr);
	  }
    }
  }
  return sum;
}

// args points to double[3], containing pW,M,D,N
double calc_exp_util(double d, double r, void *pArgs) {
  struct eu_params *pParams = (struct eu_params *) pArgs;
  double M = pParams->M;
  double D = pParams->D;
  double N = pParams->N;
  PyArrayObject *pW = pParams->pW;
  bool bPrint = pParams->bPrint;
  
  double ev = expected_next_v(pW, g_Params.pGrid1, g_Params.pGrid2, g_Params.pGrid3,
      M, D, N, d, r, g_Params.pZVals, g_Params.pZProbs, g_Params.pFS, bPrint);
  double result = (*g_Params.pU)(d) + g_Params.beta * ev;
  if (bPrint) {
    printf("calc_exp_util: M=%f D=%f N=%f d=%f r=%f\n", M, D, N, d, r);
	printf("  %f + %f * %f\n", (*g_Params.pU)(d), g_Params.beta, ev);
  }
  return result;
}

PyObject *expUtil(PyObject *self, PyObject *args) {
  double M, D, N, d, r;
  PyArrayObject *pW=NULL;

  int bPrint = 0;
  if (!PyArg_ParseTuple(args, "O!ddddd|i:expUtil",
      &PyArray_Type, &pW, &M, &D, &N, &d, &r, &bPrint)) {
    return NULL;
  }

  if (pW->nd != 3 ||
      pW->dimensions[0] != g_Params.pGrid1->dimensions[0] ||
      pW->dimensions[1] != g_Params.pGrid2->dimensions[0] ||
      pW->dimensions[2] != g_Params.pGrid3->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError, "w dimensions don't match grid");
    return NULL;
  }

  struct eu_params params = {M, D, N, pW, (bPrint != 0)};
  double result = calc_exp_util(d, r, (void*) &params);

  return Py_BuildValue("d", result);
}

PyObject *maximizer2d(PyObject *self, PyObject *args) {
  double M, D, N;
  PyArrayObject *pGrid_d = NULL, *pGrid_r = NULL, *pW=NULL;

  if (!PyArg_ParseTuple(args, "O!O!O!ddd:maximizer2d", 
      &PyArray_Type, &pGrid_d, &PyArray_Type, &pGrid_r, &PyArray_Type, &pW, &M, &D, &N)) {
    return NULL;
  }

  if (pW->nd != 3 ||
      pW->dimensions[0] != g_Params.pGrid1->dimensions[0] ||
      pW->dimensions[1] != g_Params.pGrid2->dimensions[0] ||
      pW->dimensions[2] != g_Params.pGrid3->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError, "w dimensions don't match grid");
    return NULL;
  }
  
  struct eu_params params = {M, D, N, pW};
  double argmax1, argmax2;
  int count = gridSearch2D(pGrid_d, pGrid_r, &calc_exp_util, (void*) &params, &argmax1, &argmax2);
  return Py_BuildValue("idd", count, argmax1, argmax2);
}

PyObject *test1(PyObject *self, PyObject *args) {
  char *kwlist[] = {"bPrint", NULL};
  int arg1, arg2;
  int bPrint = 0;
  if (!PyArg_ParseTuple(args,  "ii|i:test1", &arg1, &arg2, &bPrint)) {
    return NULL;
  }

  return Py_BuildValue("iii", arg1, arg2, bPrint);
}

PyMethodDef methods[] = {
  {"test3", test3, METH_VARARGS},
  {"setGlobalParams", setGlobalParams, METH_VARARGS},
  {"expUtil", expUtil, METH_VARARGS},
  {"test1", test1, METH_VARARGS},
  {"maximizer2d", maximizer2d, METH_VARARGS},
  {NULL, NULL},                    
};             
  
#ifdef __cplusplus
extern "C" {      
#endif      

#define DLLEXPORT __declspec(dllexport)    
void DLLEXPORT init_ponzi2_fns() {
  import_array();
  initGlobalParams();
  (void)Py_InitModule("_ponzi2_fns", methods);
}                                      
 
#ifdef __cplusplus
}                 
#endif

