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
  


#ifndef _merton_h
#define _merton_h

#include <vector>
#include <pyublas/numpy.hpp>
#include "myTypes.h"
#include "myFuncs.h"
#include "maximizer.h"

namespace bpl = boost::python;

class MertonParams: public BellmanParams {
  public:
    double objectiveFunction(DoubleVector const &args) const;
	
	// gamma - CRRA utility parameter (1 for log utility)
	// delta - continuous discount factor
	// mu, sigma - drift & volatility of risky asset
	// riskfree_r - risk free interest rate
	// dt - time step
    MertonParams(DoublePyArray stateGrid, double gamma, double delta, double riskfree_r, double mu, double sigma, double dt, bool bUseMonteCarlo=true);
	double u (double cf, double s) const;
	double EV (double cf, double s) const;
	double EV_raw () const;
	
	// methods inherited from BellmanParams
	void setStateVars(bpl::list const &stateVars) {
	  m_wealth = bpl::extract<double>(stateVars[0]);
	}
	int getNControls() const {
	  return 2;
	}
	void setPrevIteration(DoublePyArray const &WArray) {
      m_PrevIteration = WArray;	
      m_pPrevIterationArray = (PyArrayObject const*) m_PrevIteration.data().handle().get();
	}
	
	// non-exposed methods
	double calcEV_grid (double cf, double s, double W) const;
	double calcEV_montecarlo (double cf, double s, double W) const;
	//double calcEV_helper(double cf, double s, double W, double z) const;
	
    DoublePyArray m_StateGrid;				// grid over wealth
	PyArrayObject const *m_pStateGrid;	
	double m_wealth;						// state variable: wealth
    DoublePyArray m_PrevIteration;			// store the previous iteration of the value function
	PyArrayObject const *m_pPrevIterationArray;	
	ddFnObj m_uFn;							// utility function for consumption
	double m_gamma;							// CRRA utility parameter (1 for log utility)
	double m_riskfree_r;					// risk-free interest rate
	double m_mu;							// risky asset drift
	double m_sigma;							// risky asset volatility
	double m_dt;							// time step
	double m_delta;							// continuous discount factor
	double m_beta;							// discrete discount factor
	LognormalCDFObj m_CDFFn;				// CDF function
    LognormalPDFObj m_PDFFn;				// PDF function
	DoubleVector m_RandomDraws;				// random draws for monte carlo EV
	bool m_bUseMonteCarlo;					// use monte carlo for EV
	int m_nDraws;							// number of draws for monte carlo
};

#endif //_merton_h