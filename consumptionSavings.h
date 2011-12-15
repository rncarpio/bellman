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
  


#ifndef _consumptionSavings_h
#define _consumptionSavings_h

#include <vector>
#include <pyublas/numpy.hpp>
#include "myTypes.h"
#include "myFuncs.h"
#include "maximizer.h"

namespace bpl = boost::python;

enum EVMethodT {EV_MONTECARLO, EV_MONTECARLO2, EV_CUDA_MONTECARLO, EV_PARTIAL_EXP};

// consumption-savings problem with CRRA utility, two lognormal assets
class ConsumptionSavingsParams: public BellmanParams {
  public:
    double objectiveFunction(DoubleVector const &args) const;
	
    ConsumptionSavingsParams(DoublePyArray const &stateGrid, double gamma, double beta, double mean1, double mean2, double var2, EVMethodT evMethod);
	double u (double cf, double s) const;
	double EV (double cf, double s) const;
	double calcEV (double s1, double s2, double W, double expMean1) const;
	
	// methods inherited from BellmanParams
	void setStateVars(bpl::list const &stateVars) {
	  m_wealth = bpl::extract<double>(stateVars[0]);
	}
	int getNControls() const {
	  return 2;
	}
	void setPrevIteration(DoublePyArray const &WArray); 
		
    DoublePyArray m_StateGrid;				// grid over wealth
	PyArrayObject const *m_pStateGrid;	
	double m_wealth;						// state variable: wealth
    DoublePyArray m_PrevIteration;			// store the previous iteration of the value function
	PyArrayObject const *m_pPrevIterationArray;	
	std::shared_ptr<PyInterp1D> m_pPrevIterationInterp;
	ddFnObj m_uFn;							// utility function for consumption
	double m_gamma;							// CRRA utility parameter (1 for log utility)	
	double m_beta;				// discrete discount factor
    double m_mean1, m_mean2, m_var2;		// asset return params (asset 1 is risk free)

	EVMethodT m_EVMethod;
	int m_nDraws;
	std::vector<double> m_RandomDrawsSorted;		// draws for monte carlo
};

#endif //_consumptionSavings_h