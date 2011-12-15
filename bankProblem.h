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
  


#ifndef _bankProblem_h
#define _bankProblem_h

#include <vector>
#include <pyublas/numpy.hpp>
#include "myTypes.h"
#include "maximizer.h"
#include "myFuncs.h"

namespace bpl = boost::python;

// 2 assets: "fast" and "slow"
class BankParams3: public BellmanParams {
  public:
	double objectiveFunction(DoubleVector const &args) const;
    BankParams3(double beta, double rFast, double rSlow,
           DoublePyArray const &SlowOutFrac,
		   DoublePyArray const &FastOutFrac, DoublePyArray const &FastInFrac,
		   DoublePyArray const &ProbSpace, DoublePyArray const &BankruptcyPenalty);
	
	// methods inherited from BellmanParams
	void setStateVars(bpl::list const &stateVars) {
	  // M = M/F (per dollar in fast asset)
	  // S = S/F
      m_M = bpl::extract<double>(stateVars[0]);
      m_S = bpl::extract<double>(stateVars[1]);	  
	}
	int getNControls() const { return 2; }
	// control grid list is implemented in python
	void setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray); 

    double calc_EV(double d, double slowInFrac, DoubleVector *pNextM=NULL, DoubleVector *pNextS=NULL) const;
	bpl::tuple calc_EV_wrap(double d, double slowInFrac);
	
	double m_beta;				// discount factor	
	double m_M, m_S;		// state variables
	double m_rFast, m_rSlow;
	DoublePyArray m_StateGrid1, m_StateGrid2;
	PyArrayObject const *m_pStateGrid1, *m_pStateGrid2;
		
	// each period, a random fraction of F, S is realized as outflow, inflow
	// there is a single joint distribution over all random variables
	DoublePyArray m_SlowOutFrac, m_FastOutFrac, m_FastInFrac;
	DoublePyArray m_ProbSpace;
	double m_BankruptcyPenalty[3];

    DoublePyArray m_PrevIterArray;			// W will be a 2d array	
	PyArrayObject const *m_pPrevIter;
	std::shared_ptr<Interp2D> m_pPrevIterInterp;
	//Interp2D *m_pPrevIterInterp;
};

// with population
class BankParams4: public BellmanParams {
  public:
	double objectiveFunction(DoubleVector const &args) const;
    BankParams4(double beta, double rFast, double rSlow,
           DoublePyArray const &SlowOutFrac,
		   DoublePyArray const &FastOutFrac, DoublePyArray const &FastInFrac,
		   DoublePyArray const &ProbSpace, DoublePyArray const &BankruptcyPenalty, double PopGrowth);
	
	// methods inherited from BellmanParams
	void setStateVars(bpl::list const &stateVars) {
	  // M = M/F (per dollar in fast asset)
	  // S = S/F
	  // P = P/F
      m_M = bpl::extract<double>(stateVars[0]);
      m_S = bpl::extract<double>(stateVars[1]);
	  m_P = bpl::extract<double>(stateVars[2]);
	}
	int getNControls() const { return 2; }
	// control grid list is implemented in python
	void setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray); 

    double calc_EV(double d, double slowInFrac, DoubleVector *pNextM=NULL, DoubleVector *pNextS=NULL, DoubleVector *pNextP=NULL) const;
	bpl::tuple calc_EV_wrap(double d, double slowInFrac);
	
	double m_beta;				// discount factor	
	double m_M, m_S, m_P;		// state variables
	double m_rFast, m_rSlow;
	double m_PopGrowth;
	DoublePyArray m_StateGrid1, m_StateGrid2, m_StateGrid3;
	PyArrayObject const *m_pStateGrid1, *m_pStateGrid2, *m_pStateGrid3;
		
	// each period, a random fraction of F, S is realized as outflow, inflow
	// there is a single joint distribution over all random variables
	DoublePyArray m_SlowOutFrac, m_FastOutFrac, m_FastInFrac;
	DoublePyArray m_ProbSpace;
	double m_BankruptcyPenalty[3];

    DoublePyArray m_PrevIterArray;			// W will be a 3d array	
	PyArrayObject const *m_pPrevIter;
	//std::shared_ptr<Interp2D> m_pPrevIterInterp;
	//Interp2D *m_pPrevIterInterp;
};

#endif //_bankProblem_h