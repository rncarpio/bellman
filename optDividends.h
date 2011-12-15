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
  

#ifndef _optDividends_h
#define _optDividends_h

#include <vector>
#include <pyublas/numpy.hpp>
#include "myTypes.h"
#include "myFuncs.h"
#include "maximizer.h"

namespace bpl = boost::python;


// optimal dividends problem
class OptDividendsParams: public BellmanParams {
  public:
    double objectiveFunction(DoubleVector const &args) const;
	
    OptDividendsParams(double beta, DoublePyArray const &randomDrawsSorted);
	
	// methods inherited from BellmanParams
	void setStateVars(bpl::list const &stateVars) {
	  m_M = bpl::extract<double>(stateVars[0]);
	}
	int getNControls() const {
	  return 1;
	}
	void setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray) {
	  m_StateGrid = bpl::extract<DoublePyArray>(stateGridList[0]);
	  m_pStateGrid = (PyArrayObject const*) m_StateGrid.data().handle().get();
      m_PrevIteration = WArray;	
      m_pPrevIterationArray = (PyArrayObject const*) m_PrevIteration.data().handle().get();
      m_pPrevIterationInterp.reset(new PyInterp1D(m_StateGrid, m_PrevIteration));
	}
		
    DoublePyArray m_StateGrid;				// grid over wealth
	PyArrayObject const *m_pStateGrid;	
	double m_M;						// state variable: cash reserve
    DoublePyArray m_PrevIteration;			// store the previous iteration of the value function
	PyArrayObject const *m_pPrevIterationArray;	
	std::shared_ptr<PyInterp1D> m_pPrevIterationInterp;
	
	double m_beta;				// discrete discount factor
	std::vector<double> m_RandomDrawsSorted;		// draws for monte carlo
};

#endif //_optDividends_h