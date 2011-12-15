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
  

// data structures specific to the ponzi problem

// global parameters
typedef struct globalParams {
  PyFileObject *pOutputFileObj;
  FILE *pOutputFile;
  
  double theta;
  double beta;
  double gamma;
  PyArrayObject *pGrid1, *pGrid2;
  PyArrayObjectPtr_array gridPtrArray;		// for arbitrary dimensions, use a vector.  must update the ref count when reassigning!
  PyArrayObject *pZVals, *pZProbs;
  ddFn *pU;
  ddFn2 *pFS;
  double depositorSlope;
  double bankruptcyPenalty;
} _globalParams;

typedef struct eu_params0 {
  double M, D;
  PyArrayObject *pW;
  bool bPrint;
} _eu_params;

class eu_params {
  public:
    eu_params(double M, double D, PyArrayObject *w, int bPrint=0)	
   	{
	  m_M = M;
	  m_D = D;
	  m_stateVars.resize(2);
	  m_stateVars[0] = M;
	  m_stateVars[1] = D;
	  m_pW = w;
	  m_bPrint = (bPrint != 0);
	}
	  
    eu_params(PyObject *pArgList, PyArrayObject *w, int bPrint=0) {
	  m_pW = w;
      int result = getDoublesFromPySequence(pArgList, m_stateVars);
	  m_M = m_stateVars[0];
	  m_D = m_stateVars[1];
      assert(result >= 0);
	  m_bPrint = (bPrint != 0);
    }	  
	
	double m_M, m_D;
    DoubleArray m_stateVars;
    PyArrayObject *m_pW;
    bool m_bPrint;
};

struct globalParams g_Params;

// fraction of income deposited into bank
double fs(double r) {
  return 1.0 - exp(1.0 - r);
}
// f only
// depositors have linear mean-SD preferences, i.e. U(c) = E(c) - m*SD(c)
// k = d + D - M
// assume Z has 2 possible values
// r is gross, i.e. > 1
double f(double k, double r, void* pArg) {
  assert(ARRAYLEN1D(g_Params.pZVals) == 2);
  double zLow = *ARRAYPTR1D(g_Params.pZVals, 0);
  double zHigh = *ARRAYPTR1D(g_Params.pZVals, 1);
  double pHigh = *ARRAYPTR1D(g_Params.pZProbs, 1);
  assert(zLow < zHigh);
  if (k > zHigh) {
    return 0.0;
  } else if (k <= zHigh && k > zLow) {
    // slope of feasible region line
	// in this case, bank survives if zHigh occurs, so probability = pHigh
    double slope = (r*pHigh - 1.0)/(r*sqrt(pHigh * (1.0-pHigh)));
	if (g_Params.depositorSlope > slope) {
	  return 0.0;
	} else {
	  return 1.0;
	}
  } else if (k <= zLow && k > 0.0) {
    return 1.0;
  } else {
    // k < 0
    return 1.0;
  }
}

PyObject *testf(PyObject *self, PyObject *args) {
  double k, r;

  if (!PyArg_ParseTuple(args, "dd:testf", &k, &r)) { 
    return NULL;
  }
  double result = f(k,r, NULL);
  return Py_BuildValue("d", result);
}

// default values. can be overridden later
void initGlobalParams() {
  g_Params.pOutputFileObj = NULL;
  g_Params.pOutputFile = PySys_GetFile("stdout", stdout);
  
  g_Params.beta = 0.9;
  g_Params.theta = 0.5;
  g_Params.gamma = 2.0;
  g_Params.pGrid1 = g_Params.pGrid2 = NULL;
  g_Params.pZVals = g_Params.pZProbs = NULL;
  g_Params.pU = &U_linear;
  g_Params.pFS = &f;
  g_Params.depositorSlope = 1.0;
  g_Params.bankruptcyPenalty = 0.0;
}

PyObject* setGlobalParams(PyObject *self, PyObject *args) {
  double theta, beta, gamma;
  const char* pcUFnName = NULL;
  PyArrayObject *pGrid1=NULL, *pGrid2=NULL;
  PyArrayObject *pZVals=NULL, *pZProbs=NULL;
  double depositorSlope, bankruptcyPenalty;
  PyObject *pGridList = NULL;
  uint i;
  
  PyArrayPtrPtr src[] = {&pGrid1, &pGrid2, &pZVals, &pZProbs};
  PyArrayPtrPtr dest[] = {&g_Params.pGrid1, &g_Params.pGrid2,
     &g_Params.pZVals, &g_Params.pZProbs};

  if (!PyArg_ParseTuple(args, "dddsO!O!O!O!ddO:setGlobalParams", &theta, &beta, &gamma, &pcUFnName,
      &PyArray_Type, &pGrid1, &PyArray_Type, &pGrid2, 
      &PyArray_Type, &pZVals, &PyArray_Type, &pZProbs,
	  &depositorSlope, &bankruptcyPenalty,
	  &pGridList)) {
    return NULL;
  }
  g_Params.beta = beta;
  g_Params.theta = theta;
  g_Params.gamma = gamma;
  // set pointer values
  for (i=0; i<CARRAYLEN(src); i++) {
    if (*(dest[i]) != NULL) {
      Py_DECREF(*(dest[i]));
    }
    Py_INCREF(*(src[i]));
    *(dest[i]) = *(src[i]);
  }

  // for multidimensional grids, copy from list to array, take care of ref counts    
  int nGrids = PyObject_Length(pGridList);
  PyArrayObjectPtr_array tempArray(nGrids);
  if (nGrids < 0) {
    PyErr_SetString(PyExc_ValueError, "gridList has length < 0");
	return NULL;
  }
  if (getPyArrayFromPySequence(pGridList, tempArray) <= 0) {
    PyErr_SetString(PyExc_ValueError, "could not read items from gridList");
	return NULL;
  }
  // decr ref counts for old object
  for (i=0; i<g_Params.gridPtrArray.size(); i++) {
    Py_DECREF(g_Params.gridPtrArray[i]);
  }
  g_Params.gridPtrArray = tempArray;
  for (i=0; i<g_Params.gridPtrArray.size(); i++) {
    Py_INCREF(g_Params.gridPtrArray[i]);
  }
    
  char *fnNames[] = {"crra", "exponential", "linear"};
  ddFn *u_functions[] = {&U_crra, &U_exponential, &U_linear};
  
  // set utility function
  bool bFoundMatch = false;
  for (uint i=0; i<CARRAYLEN(fnNames); i++) {
    if (strcmp(pcUFnName, fnNames[i]) == 0) {
	  g_Params.pU = u_functions[i];
	  DebugMsg("setGlobalParams: setting utility to %s\n", pcUFnName);
	  bFoundMatch = true;
    }	
  }
  char pcTemp[1024];
  sprintf(pcTemp, "unknown utility fn name: %s", pcUFnName);
  if (bFoundMatch == false) {
    PyErr_SetString(PyExc_ValueError, pcTemp);
    return NULL;
  }
  
  // check that zProbs sum up to 1
  double sum = 0;
  for (uint i=0; i<ARRAYLEN1D(pZProbs); i++) {
    sum += * ARRAYPTR1D(pZProbs, i);
  }
  if (sum != 1.0) {    
    char pcTemp[1024];
	//printf("diff: %.30f", sum - 1.0);
    std::string sErr("z probs don't add up to 1: ");
	sprintf(pcTemp, "%.30f = ", sum);
	sErr += pcTemp;
    for (uint i=0; i<ARRAYLEN1D(pZProbs)-1; i++) {
      sprintf(pcTemp, "%f +", * ARRAYPTR1D(pZProbs, i));
	  sErr += pcTemp;
    }
    sprintf(pcTemp, "%f", * ARRAYPTR1D(pZProbs, ARRAYLEN1D(pZProbs)-1));
    sErr += pcTemp;	
	
    PyErr_SetString(PyExc_ValueError, sErr.c_str());
    return NULL;
  }
  
  g_Params.depositorSlope = depositorSlope;
  g_Params.bankruptcyPenalty = bankruptcyPenalty;
  
  return Py_BuildValue("i", 1);
}

// w_array is a 3D array
// grid_M, grid_D, grid_N are 1d arrays
// M, D, N are the state values
// d, r are the controls
// z_vals, z_probs are shock distribution values
// fs is a function that takes r, returns the fraction of income invested in bank (a double)
double expected_next_v
(PyArrayObject *pWArray, 
    PyArrayObject *pMGrid, PyArrayObject *pDGrid,
    double M, double D, double d, double r, double bankruptcyPenalty,
    PyArrayObject *pZVals, PyArrayObject *pZProbs,
    ddFn2 *pFSFn, bool bPrint=false) {
  int zi, zlen;
  double *pZ, fs, probZ, nextM, nextD;
  double sum;

  assert(pZVals->nd == pZProbs->nd);
  assert(pZVals->dimensions[0] == pZProbs->dimensions[0]);
  zlen = pZVals->dimensions[0];
  // calculate expectation -> for each possible shock z...
  if (bPrint) {
    DebugMsg("expected_next_v M=%f D=%f d=%f r=%f\n", M,D,d,r);
  }
  sum = 0.0;
  for (zi=0; zi<zlen; zi++) {
    pZ = (double*) ARRAYPTR1D(pZVals, zi);
    probZ = * (double*) ARRAYPTR1D(pZProbs, zi);
    fs = (*pFSFn)(d + D - M, r, NULL);   
    nextM = (M + fs * (*pZ) - D - d) / (*pZ);
    nextD = r * fs;
    if (nextM <= 0.0) {
      //sum += 0.0;	  
	  // bankruptcyPenalty should be a negative number.
	  double incr = probZ * (*pZ) * bankruptcyPenalty * (-nextM);				// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
	  //double incr = probZ * bankruptcyPenalty * (-nextM);				// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
	  sum += incr;
	  if (bPrint) {
	    DebugMsg("  probZ=%f z=%f nextM=%f nextD=%f: +%f\n", probZ, *pZ, nextM, nextD, incr);
	  }
    } else {
	  double interpVal = interp2d_grid(pMGrid, pDGrid, pWArray, nextM, nextD);	// multiply by *pZ because we changed the problem to use per-customer (divided by N_t) variables
      double incr = probZ * (*pZ) * interpVal;
	  //double incr = probZ * interpVal;
	  sum += incr;
	  if (bPrint) {
	    DebugMsg("  probZ=%f z=%f nextM=%f nextD=%f: w=%f +%f\n", probZ, *pZ, nextM, nextD, interpVal, incr);
	  }
    }
  }
  return sum;
}

// args points to double[3], containing pW,M,D,N
double calc_exp_util_orig(double d, double r, void *pArgs) {
  eu_params *pParams = (eu_params *) pArgs;
  double M = pParams->m_M;
  double D = pParams->m_D;
  PyArrayObject *pW = pParams->m_pW;
  bool bPrint = pParams->m_bPrint;
  
  double ev = expected_next_v(pW, g_Params.pGrid1, g_Params.pGrid2,
      M, D, d, r, g_Params.bankruptcyPenalty, g_Params.pZVals, g_Params.pZProbs, g_Params.pFS, bPrint);
  double result = (*g_Params.pU)(d) + g_Params.beta * ev;
  if (bPrint) {
    DebugMsg("calc_exp_util: M=%f D=%f d=%f r=%f\n", M, D, d, r);
	DebugMsg("  %f + %f * %f = %f\n", (*g_Params.pU)(d), g_Params.beta, ev, result);
  }
  return result;
}

double calc_exp_util(double d, double r, void *pArgs) {
  eu_params *pParams = (eu_params *) pArgs;
  double M = pParams->m_M;
  double D = pParams->m_D;
  PyArrayObject *pW = pParams->m_pW;
  bool bPrint = pParams->m_bPrint;
  
  double ev = expected_next_v(pW, g_Params.pGrid1, g_Params.pGrid2,
      M, D, d, r, g_Params.bankruptcyPenalty, g_Params.pZVals, g_Params.pZProbs, g_Params.pFS, bPrint);
  double result = (*g_Params.pU)(d) + g_Params.beta * ev;
  if (bPrint) {
    DebugMsg("calc_exp_util: M=%f D=%f d=%f r=%f\n", M, D, d, r);
	DebugMsg("  %f + %f * %f = %f\n", (*g_Params.pU)(d), g_Params.beta, ev, result);
  }
  return result;
}

double calc_exp_util2(DoubleArray const &argArray, void *pArgs) {
  return calc_exp_util(argArray[0], argArray[1], pArgs);
}

PyObject *expUtil(PyObject *self, PyObject *args) {
  double M, D, d, r;
  PyArrayObject *pW=NULL;

  int bPrint = 0;
  if (!PyArg_ParseTuple(args, "O!dddd|i:expUtil",
      &PyArray_Type, &pW, &M, &D, &d, &r, &bPrint)) {
    return NULL;
  }

  if (pW->nd != 2 ||
      pW->dimensions[0] != g_Params.pGrid1->dimensions[0] ||
      pW->dimensions[1] != g_Params.pGrid2->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError, "w dimensions don't match grid");
    return NULL;
  }

  eu_params params(M, D, pW);
  params.m_bPrint = (bPrint != 0);
  double result = calc_exp_util(d, r, (void*) &params);

  return Py_BuildValue("d", result);
}

