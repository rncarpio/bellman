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
  


#ifndef _ponziProblem_h
#define _ponziProblem_h

#include <vector>
#include <pyublas/numpy.hpp>
#include "myTypes.h"
#include "maximizer.h"
#include "myFuncs.h"

namespace bpl = boost::python;

class GlobalParams {
public:
  // default values. can be overridden later
  void initGlobalParams();
	
 // global parameters (not per-call)
  double m_theta;
  double m_beta;
  double m_gamma;
  DoublePyArray m_StateGrid1, m_StateGrid2; 	// for 2-dimensional state space
  PyArrayObject const *m_pStateGrid1, *m_pStateGrid2;
  DoublePyArrayVector m_StateGridArray;			// for arbitrary dimensional state space, use a vector.    
  DoublePyArray m_ZVals, m_ZProbs;				// distribution of the shock
  PyArrayObject const *m_pZVals, *m_pZProbs;
  ddFn *m_pU;									// utility function
  //ddFnObj m_UFn;
  ddFn2 *m_pFS;
  double m_depositorSlope;
  double m_bankruptcyPenalty;  
};

static void setGlobalParams(double theta, double beta, double gamma, const char* pcUFnName, 
    DoublePyArray const &grid1, 
	DoublePyArray const &grid2, 
	DoublePyArray const &zVals, 
	DoublePyArray const &zProbs, 
	double depositorSlope, 
	double bankruptcyPenalty);

class PonziParams: public BellmanParams {
  public:
	double objectiveFunction(DoubleVector const &args) const;
    PonziParams() {
	  m_bPrint = false;
	}
	// methods inherited from BellmanParams
	void setStateVars(bpl::list const &stateVars);
	int getNControls() const;
	// control grid list is implemented in python
	void setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray);	

	DoublePyArray m_StateGrid1, m_StateGrid2;
	PyArrayObject const *m_pStateGrid1, *m_pStateGrid2;	
	double m_M, m_D;    
    DoublePyArray m_W;
	PyArrayObject const *m_pW;
	std::shared_ptr<PyInterp2D> m_pWInterp;
    bool m_bPrint;
};

double expected_next_v (PonziParams const *pParams,
    double M, double D, double d, double r, double bankruptcyPenalty,
    PyArrayObject const *pZVals, PyArrayObject const *pZProbs,
    ddFn2 *pFSFn, bool bPrint=false);


#endif //_ponziProblem_h